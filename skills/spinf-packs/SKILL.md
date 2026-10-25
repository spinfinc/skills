---
name: spinf-packs
description: Author, calibrate, validate and publish a spinf prompt pack - a versioned JSON file of calibrated questions that any spinf-serve (or the hosted API, for official packs) scores in the same pass as a caller's own questions and turns into category scores and a decision. Use when turning a question set into a reusable pack, building a pack repository (prompt-packs.json, fixtures, CI), reading or editing a pack file, re-deciding stored results, or running a pack locally with the spinf_serve.packs evaluator. Calling a pack from client code is the `spinf` skill.
---

# Authoring prompt packs

A **prompt pack** is one JSON file: the scoring request as the API takes it, how each answer becomes a *read*, how reads
become calibrated *category scores* (0–1, with a percentile on normal traffic), and how categories become one
*decision*. It is self-contained and **model-specific**: its constants were measured on the model it names. A server
started with `--packs <source>` offers it under `GET /v1/packs`, and a caller names it in `scoring.packs`.

The format specification is https://github.com/spinfinc/spinf-prompts/blob/main/docs/pack-format.md and the official
repository https://github.com/spinfinc/spinf-prompts is the reference example of a pack repository (files, fixtures,
samples, manifest, CI). The evaluator is the pure-Python `spinf_serve.packs` module (`pip install spinf-serve`, no GPU
dependencies). Fetch the specification before writing a field not covered here.

## Decide the shape first

- **A decision pack** (moderation, triage): categories with thresholds, a `decision` that says safe / unsafe or picks a
  class. Needs labelled data for the thresholds and, for the `micro_layer` mode, a training set.
- **A report-only pack** (an indicator, a news index's question set): every threshold `null`, no deciding category.
  The pack carries the questions and their per-read baselines (mean, sd on a public sample); the consumer aggregates.
  No labels needed. The decision is `per_category` and always safe.
- **A parameterised pack** (brand perception): a value the caller passes per input (`inputs[].parameters`), written into
  the context only (`{brand}`), never into the shared questions; the questions refer to it through a handle the context
  defines ("the brand of interest").

Rules that shape the questions: every query is a scored one (`{?}` with at least two options, 1 to 5 tokens each); one
`{F}` placeholder with `combinations` or `values` turns a template into a family of reads; the content is presented
through a **context** template with exactly one `{X}`; one pack, one model. Write and test the questions as in the
`spinf` skill (content label in the context, answers named in the question, several wordings per concept).

## The file

```json
{ "pack": "acme/support-triage", "version": "0.1.0", "model": "spinf/gemma-4-12b-fp8", "media": ["text"],
  "answers": "logprob",
  "contexts": { "ticket": "Customer support message:\n\n{X}" },
  "scoring": { "empty_floor": false, "case_insensitive": false,
               "queries": [ { "id": "urgent", "template": "\n\nIs this urgent?\nAnswer:{?}", "options": [" yes", " no"] },
                            { "id": "topic",  "template": "\n\nIs this about {F}?\nAnswer:{?}", "options": [" yes", " no"],
                              "values": { "F": ["billing", "a technical problem"] } } ] },
  "reads": [ { "id": "urgent", "query": "urgent", "kind": "pair", "sign": 1, "category": "urgent", "mean": -2.1, "sd": 1.3 },
             { "id": "topic:billing", "query": "topic", "F": "billing", "kind": "pair", "category": "billing", "mean": -1.4, "sd": 1.6 } ],
  "categories": { "urgent":  { "label": "Urgent", "platt": [1.0, 0.0], "threshold": 0.8,
                               "normal_quantiles": { "levels": [0, 0.5, 0.9, 0.99, 1], "values": [0.02, 0.1, 0.4, 0.8, 0.99] } },
                  "billing": { "label": "Billing", "platt": [1.0, 0.0], "threshold": null, "normal_quantiles": { "…": "…" } } },
  "decision": { "mode": "per_category", "fallback": "per_category", "critical": [], "per_category": {} } }
```

- **reads**: one per (query, combination) used. `pair` (two options): `sign · logit(p0 / (p0 + p1))`; `option`:
  `ln p[option] − mu`; `any` and `set` for one-of-many questions (see the specification). `mean` and `sd` are measured on
  normal traffic; the read's standardised value is `z = (value − mean) / sd`. `view` tags reads so a category can average
  per view (`"aggregate": "views"`) rather than per read.
- **categories**: `score = sigmoid(a · macro + b)` with `platt = [a, b]` and `macro` the mean `z` of its reads
  (`[1, 0]` when nothing is fitted on labels: then `logit(score)` is the mean z); `threshold` in [0, 1] or `null`;
  `normal_quantiles` for the percentile; `requires` (flagged only while the listed categories are flagged), `decides:
  false` (a gate that never decides).
- **decision**: `max` (highest deciding category against one threshold), `per_category` (any flagged), `micro_layer`
  (a small layer over category macros and read z's, with a `lock` over the questions and calibration; the `fallback`
  mode decides when the lock no longer matches). `critical` categories force unsafe whenever flagged. Thresholds,
  `critical`, labels, contexts, `media` and `parameters` are editable in place; anything else is a new version.
- `answers: "logprob"` takes option probabilities from the full-precision log-probabilities (preferred); `"p"` from the
  API's rounded `p`. The constants fitted for one do not fit the other.

## Build it

1. **Draft.** Write the file without `mean`, `sd`, `platt`, `normal_quantiles` and thresholds. `packs.load(path,
   draft=True)` validates everything but the constants and gives the request side.
2. **Score a sample** on the model the pack names, on the server that will run it (the hosted API or your own; scores
   depend on the engine, and the calibrated fp8 scheme makes them identical between an Ada card and the hosted service):

   ```python
   from spinf_serve import packs
   pack = packs.load("drafts/support-triage.json", draft=True)
   body = pack.request(texts, context="ticket")              # the /v1/score body: texts wrapped in the context,
   result = client.post("/v1/score", json=body).json()       # the queries in their own group, floors off
   ```

   For a report-only pack, a few thousand public texts of the right kind are the sample (keep their source and licence;
   a pack ships only questions and constants, never the sample's text unless it is cleared). For a decision pack,
   score the labelled set too.
3. **Calibrate.** Per read: the value from the specification's formula, then its `mean` and `sd` over the sample. Per
   category: `platt` fitted on labels (logistic regression of the label on the macro), or `[1, 0]`; `normal_quantiles`
   from the sample's scores at the chosen levels; `threshold` where the labelled errors are acceptable, or at the score
   that `rate` of normal traffic reaches, or `null`. The micro-layer is optional: a decision pack works with
   `per_category` or `max`.
4. **Validate.** `spinf-serve packs validate <folder>` (or `python -m spinf_serve.packs validate`) checks the file and
   the manifest and reports whether the lock matches. Fix every line it prints.
5. **Fixture.** Score a handful of sample texts and store, per sample, the raw `/v1/score` result and
   `pack.evaluate(result, mode=…)` for every mode the pack supports, with the engine fingerprint:
   `{"pack", "version", "engine", "cases": [{"id", "text", "context", "params", "result", "expected": {mode: …}}]}`.
   The fixture is what lets CI recompute the pack without a GPU and what pins a version forever.
6. **README** per pack: what it reads, the categories and their roles, the wording of the questions, the sample it was
   standardised on, how to read `score` and `percentile`. Readers look for the questions first: list them.

## The repository

Any GitHub repository with a `prompt-packs.json` at its root is a pack source for spinf-serve
(`--packs github:owner/repo@tag`). Use your own namespace (by convention your GitHub owner; `spinf/` is reserved), pin
a tag per release, and let CI validate:

```
prompt-packs.json                      the manifest: format 1, namespace, packs[] with id, name, description, tags, media,
                                       parameters, readme, samples, versions[] {version, file, model, status, fixture}
packs/<name>/<version>.json            the pack; a published version never changes
packs/<name>/README.md, samples/inputs.json, fixtures/<version>.json
LICENSE, ACKNOWLEDGEMENTS.md           the pack's licence; the data it was standardised or checked on, with terms
.github/workflows/ci.yml               uv pip install spinf-serve; spinf-serve packs validate .; packs list --packs . --no-official;
                                       a check that recomputes every fixture with spinf_serve.packs
```

The default version is the first `available` one in manifest order (newest first); `deprecated` is served but not the
default; `coming_soon` is listed, not served. Keep build material (drafts, calibration samples, scripts) in a clearly
named folder apart from `packs/`, so a visitor finds the questions at once. A folder without a manifest is scanned for
`*.json` files with `pack` and `scoring`, which is convenient locally and wrong for anything else.

## Running and re-deciding without a server

The evaluator runs anywhere Python runs. From stored raw results (scored with `include_queries: true` or from the
pack's own queries) another mode, other thresholds or another `critical` list give a new decision with no scoring:

```python
pack = packs.load("packs/moderation/0.1.3.json")
out = pack.evaluate(result, mode="per_category", thresholds={"spam": 0.7}, critical=[])
```

An edited pack (a changed question or constant) fails the micro-layer lock and decides with its fallback mode, with a
`warning` in the decision. spinf-client carries the same evaluator (`spinf_client.packs`, `Moderation.redecide()`), and
the hosted API runs it too, so the three agree to the fixture's rounding.

## Versioning and publication

- A version never changes once published. Refit thresholds or calibration: new patch version. Changed or added
  questions, reads or categories: new minor. A new model: a new pack (the constants do not transfer).
- The official `spinf/` packs are published in spinf-prompts and served by the hosted API; a community pack is served by
  any spinf-serve started with its source. A caller runs an edited copy of a pack through spinf-client's `pack_file=`
  (its questions go out as ordinary queries, the decision is made locally).
- Credit the data: a pack carries questions and constants, and the sample they were measured on is named with its
  licence in the repository, not bundled unless its terms allow it.
