---
name: spinf
description: Classify, score and extract from content with the spinf scoring API, hosted (api.spinf.com) or self-hosted with spinf-serve - many questions about one piece of content in one call, a probability for each allowed answer, no generated text. Use when writing, reviewing or debugging code that calls spinf (/v1/score, /v1/chat/completions with extra_body.scoring, or the spinf-client Python package), or when designing questions, answer lists, few-shot examples, thresholds, calibration and aggregation (macro cells) of spinf scores.
---

# spinf scoring API

spinf reads a piece of content once and answers many closed questions about it in the same pass. Each question is a
**template** that ends at a `{?}` slot plus the **options** (answers) allowed there; the result is the probability of each
option. Nothing is generated. The same request and response run against two servers:

| | hosted API | your own spinf-serve |
|---|---|---|
| base URL | `https://api.spinf.com` | the server's, e.g. `http://localhost:8000` |
| key | `Authorization: Bearer $SPINF_API_KEY` (`ssk-…`) | none, unless the server was started with `--token` |
| model name | `spinf-12b`, `spinf-31b` | `spinf/gemma-4-12b-fp8`, `spinf/gemma-4-31b-fp8`, `spinf/qwen3.8-27b-int4`; the hosted names are accepted as aliases, and the field may be left out (one server, one model) |
| cost | input tokens billed per model price, with a per-call minimum (`GET /v1/models`) | none: the GPU is yours |
| limits | the account's | the server's (`GET /v1/models` reports them) |

Everything below applies to both unless marked **hosted only**. Installing and running the server is the `spinf-serve`
skill; writing a prompt pack is the `spinf-packs` skill.

**Raw scores are not the result.** A single raw `p` carries the bias of its question's wording. Before scores are used
for decisions or analysis, calibrate them and, for measurements, combine several cells into a macro cell: see
[Calibrate, then aggregate](#calibrate-then-aggregate-macro-cells).

Reference, as Markdown (the same pages describe both servers): the docs index
https://github.com/spinfinc/spinf-serve/blob/main/docs/README.md, scoring API `docs/scoring-api.md`, writing questions
`docs/writing-questions.md`, calibration `docs/calibration.md`, prompt packs `docs/prompt-packs.md`. Hosted account,
prices and limits: https://spinf.com/llms.txt. Fetch the reference before relying on a field or limit not covered here.

## Setup

- **Hosted**: the key lives in the `SPINF_API_KEY` environment variable (created at https://spinf.com/console/api-keys,
  shown once). Never write a key into code, config, tests, logs or commits, and never ask the user to paste it into the
  chat. Check it with `curl -s https://api.spinf.com/v1/account -H "Authorization: Bearer $SPINF_API_KEY"`. The API does
  not accept browser (CORS) calls: call it from servers, scripts and jobs.
- **Self-hosted**: `SPINF_API_URL` (or the `base_url`) is the server; `GET /readyz` answers 200 when the model is loaded
  and warm; `GET /v1/models` lists the model, its capabilities and the effective limits. A token only if the server
  requires one.
- Three ways to call, same body: plain HTTP (`POST /v1/score`), the OpenAI SDK pointed at `<base>/v1`
  (`/v1/chat/completions`, scoring in `extra_body`), or the Python package `spinf-client` (`pip install spinf-client`).

## A call

```bash
curl $SPINF_API_URL/v1/score \
  -H "Authorization: Bearer $SPINF_API_KEY" -H "Content-Type: application/json" \
  -d '{
    "model": "spinf-12b",
    "messages": [ { "role": "user", "content": "Customer support message:\nI was charged twice for September." } ],
    "scoring": { "queries": [
      { "id": "topic",
        "template": "\n\nTopic (billing, technical, account, shipping, or cancellation):{?}",
        "options": [" billing", " technical", " account", " shipping", " cancellation"] },
      { "id": "urgent",
        "template": "\n\nIs this urgent (business blocked, security, legal threat, or data loss)?\nAnswer:{?}",
        "options": [" yes", " no"] }
    ] }
  }'
```

With spinf-client:

```python
from spinf_client import SpinfClient, query, text

client = SpinfClient()                                                                  # hosted: SPINF_API_KEY
client = SpinfClient(base_url="http://localhost:8000", model="spinf/gemma-4-12b-fp8")   # your own server
out = client.score(
    content=[text("Customer support message:\nI was charged twice for September.")],
    queries=[query("urgent", "\n\nIs this urgent?\nAnswer:{?}", options=[" yes", " no"])],
)
out["results"][0]["queries"][0]["combinations"][0]["options"]   # [{"text": " yes", "p": …, "floor_p": …}, …]
```

With the OpenAI SDK the same body goes to `/v1/chat/completions`, `scoring` in `extra_body`; `choices` is empty and the
results are in `resp.model_extra["results"]`.

Response shape (trimmed):

```json
{ "model": "spinf-12b", "system_fingerprint": "fp_…",
  "results": [ { "input_id": "0", "queries": [ { "id": "topic", "combinations": [ {
      "values": {},
      "options": [ { "text": " billing", "logprob": -0.9, "p": 0.97, "floor_p": 0.31 }, … ],
      "residual": 0.42, "floor_residual": 0.55 } ] } ] } ],
  "warnings": [],
  "usage": { "prompt_tokens": 131,
             "prompt_tokens_details": { "content_tokens": 14, "query_tokens": 60, "floor_tokens": 57 } } }
```

The hosted API adds `usage.billed_tokens`; a server of your own has no billing fields. Treat them as optional.

## Rules for a question (refused with a 400 otherwise)

- The template is read **right after the content**: start it with `"\n\n"`, and put exactly one slot at the very end:
  `{?}` for a scored question, or an extraction slot (below). Up to 5 `{name}` placeholders; `{{` / `}}` for literal
  braces. A filled template is at most 256 tokens by default.
- Write each option as it would continue the template, **with a leading space** (`" yes"`), and no space before `{?}`
  (`"…Answer:{?}"`, not `"…Answer: {?}"`, which fails with `option_boundary`).
- 1 to 1,000 options per question, 1 to 5 tokens each, no duplicates (case-insensitive by default).
- One content (`messages`, or one entry of `inputs`) is up to 4,096 tokens by default (a server of your own can raise
  it); roles are ignored, the text is read as it is. Up to 10,000 questions (templates × combinations) per content.
- Placeholders are filled from `combinations`: a list of `{name: value}` objects, or `"all"` with
  `"values": {name: [...]}` for the full grid (up to 1,000 per question).
- Request body up to 2 MB, up to 2M processed tokens per request, by default.

## Extraction and generation slots

A template may end with `{*}` (text), `{n}` (a number) or `{$}` (an amount with a currency) instead of `{?}`: the model
writes the answer, decoded from the same read of the content, with its probability. No `options` on such a query.

- `{*}`: write the template with an opening quote and name the thing: `'\n\nThe company in this news is "{*}"'`. The
  answer ends at the closing quote or a newline. `{n}` / `{$}`: no quotes and no filler, end right at the slot
  (`"\n\nUnits sold: {n}"`). Take a number's sign from `p_negative`, not from the top answer.
- Per query: `max_tokens` (1 to 64, default 16), `top_k` (1 to 20 answers, default 5), `calibration: {a, b}` fitted
  once per field on about 50 labelled examples (adds `p_calibrated`). Result: `answers` best first, each `{text, p}`
  (+ `value`, `currency`), `mass` (probability on all answers found), `p_negative`.
- `"generate": true` on a `{*}` slot writes free text instead (a one-sentence summary, a caption): greedy, or sample with
  `{"temperature", "top_p", "top_k", "seed", "n"}`; `max_tokens` up to 100; the answers are `{text, finish}` without a
  probability.
- Up to 10 slots per content by default. Extraction has no empty floor. When the answers are a known short list, a
  scored question is cheaper and has a floor. **Hosted only**: extraction and generation are account features
  (`403 feature_not_enabled` otherwise) and a slot is billed as a flat token amount; a server of your own has them on.

## Many contents in one call

Send `inputs` instead of `messages`: `[{ "id": "t1", "messages": [...] }, …]`. Every input is scored with all of
`scoring.queries`, or only the ids in its `query_ids`; an input can add its own `queries`. Results come back one per
input, in request order, with `input_id`. Batch whenever there is more than one item: the content is read once per
input but each question's floor is computed once per call, and on the hosted API the minimum is per call. Size batches
to stay under the body limit (2 MB by default; spinf-client's `moderate()` batches for you).

## Reading the scores

- `p`: the probability of each option among the options listed (they sum to 1). Decide on `p`, with a threshold.
- `floor_p`: the same question scored with **empty** content, the question's own bias. The lift `p - floor_p` is what
  the content says. Without few-shot examples, the lift (or the option with the largest lift) often classifies better
  than raw `p` on multi-choice questions. With examples in the content, use `p` as it is.
- `residual`: the probability mass on words other than the options. Mostly set by the template; close to 1 for
  multi-token options. Compare it only across contents under the same question. A `low_option_mass` warning means the
  options are rarely the model's next words: reword the template or the options.
- The floor depends only on the question: after a first call, keep its `floor_p` and send
  `"scoring": {"empty_floor": false, …}` to skip it. Recompute it when `system_fingerprint` changes.
- Scores are comparable over time only under the same template, options, examples and `system_fingerprint`. The
  fingerprint also changes with the engine version and the GPU architecture (an Ada card of your own and the hosted
  service give the same scores with the calibrated fp8 scheme; Ampere does not): calibrate on the server you will run on.

## Calibrate, then aggregate (macro cells)

A **cell** is one score: one question (a template with one combination of its placeholders) on one content. A single
cell is noisy and carries the bias of its wording, so do not report or threshold raw cells on their own.

1. **Calibrate each cell.** Score a baseline set of contents of the same kind (a month of articles from the same
   sources, a random sample of tickets). For each cell keep the distribution of its lift `p - floor_p` (or of `p` when
   the content has few-shot examples), and express every new score as a z-score or percentile of that distribution.
2. **Build macro cells.** Ask the thing you care about through several cells: 3 to 5 wordings of the question, the
   same question over related placeholder values, and related questions that point the same way (flip the sign of
   those that point the opposite way). Average their calibrated scores into one macro cell. Each cell's wording bias
   and noise partly cancel. Extra cells are cheap: the content is read once, each cell costs its own tokens.
3. **Aggregate.** Average macro cells over contents (per day, entity, product or segment), then standardise each
   aggregate on its own trailing window. This is how news indexes turn hundreds of questions over millions of
   articles into daily reads.
4. **For per-item decisions**, calibrate on labelled items instead: pick the threshold on `p`, or on a macro cell's
   score, that gives the precision and recall you need, and send the uncertain band to review.

Keep a macro cell's cells, templates, options, examples and baseline fixed once in use, and recalibrate when
`system_fingerprint` changes. Design the macro cells with the user when a project starts. A question set that is
calibrated and reused is a **prompt pack**: the `spinf-packs` skill turns it into a file any server loads.

## Designing a classifier

1. **Label the content.** Start it with what it is: `Customer review:`, `User post:`, `News article:`.
2. **Name the answers in the question** and make the options those words: short, distinct, one or two words each.
   Put definitions in the question. For a long label list, ask the broad category first, then the label within it.
3. **Add 4 to 8 labelled examples** in the content, before the item, in the same format as the question, separated
   by `---`; cover every answer and the hard cases. This is the biggest lever on accuracy. Examples are content, so
   they are read (and, hosted, billed) with every input.
4. **Measure before scaling.** Score 100–300 items with known labels (never the ones used as examples), try two or
   three wordings and 0 / 4 / 8 examples, keep the best, and pick thresholds (act above 0.9, send 0.5–0.9 to review).
   Then freeze the template, options and examples.
5. **Ask everything in one read.** Each extra question costs only its own tokens. Use that for the checks you would
   otherwise skip, and for the extra wordings that make up macro cells.

## Prompt packs

A pack is a ready-made, calibrated set of questions for one job, named in the call and scored in the same pass as the
user's own questions. `GET /v1/packs` lists what the server offers (hosted: the official packs; a server of your own:
the official packs of its model plus whatever `--packs` sources it was started with). The official packs for the 12B
are `spinf/moderation`, `spinf/triage`, `spinf/entertainment`, `spinf/products`, `spinf/reviews` and `spinf/brand`.

```json
"scoring": {
  "packs": [ { "id": "spinf/moderation", "context": "forum" } ],
  "queries": [ { "id": "on_topic", "template": "\n\nIs this post about cooking?\nAnswer:{?}", "options": [" yes", " no"] } ]
}
```

- The listing gives each pack's versions, `contexts` (moderation: `ai` for messages to an assistant, `forum` for posts),
  categories with their default `threshold` (`null` = reported, never decides), `parameters` and decision modes. Read
  thresholds from there; don't hardcode them. Pin `version` in production.
- Response: `results[].packs[]` = `{id, version, context, categories: {<id>: {score, percentile, flagged}}, decision}`.
  `score` is calibrated 0–1 and comparable across categories; `percentile` places it on normal traffic; `flagged` =
  over the category's threshold. `decision.unsafe` is the verdict; `decision.mode` is `micro_layer`, `per_category`
  (override thresholds with `decision.thresholds`) or `max`; `decision.critical` lists categories that make the content
  unsafe whenever flagged (`[]` turns it off). Choose thresholds on the user's own labelled sample.
- A pack with `parameters` (the brand pack) takes them per input: `inputs[].parameters = {"brand": "Kestrel"}`.
- Rules: the content is wrapped in the pack's context, and the user's own questions read it wrapped too; leave
  `scoring.case_insensitive` out (`400 pack_conflict`); don't use the pack's id prefixes (`tax.`, `gen.`) for your own
  questions; text only for the moderation packs. `include_queries: true` returns the pack's raw question results, so a
  stored result can be re-decided later without scoring again (spinf-client: `Moderation.redecide()`).
- spinf-client: `client.moderate(texts, context="forum", thresholds=…)` wraps the moderation pack with typed results and
  helpers to pick thresholds from the user's own traffic (`moderation.rate_thresholds`, `fit_thresholds`).

## Cost and capacity

- **Hosted only**: input tokens are billed, the scores are free. Billed tokens per call = each content once + per
  question the filled template and each option's tokens after its first + the same for the floor (once per distinct
  question per call) + a flat amount per extraction slot, with a per-call minimum. Prices and the minimum are in
  `GET /v1/models`; estimate the cost of a large run (tokens per item × items) and tell the user before starting it.
- **Self-hosted**: no bill; the budget is GPU time. One 24 GB card scores about 10,000 processed tokens per second on
  news-length content with the 12B model. Send many inputs per call rather than many small calls; the `usage` and
  `timing` blocks of a response say what a call consumed.

## Errors and retries

Errors are `{"error": {"type", "code", "message", "param"}, "request_id"}`; `param` points at the bad field.

- `400`: fix the request (`invalid_template`, `option_boundary`, `option_too_long`, `content_too_long`,
  `unknown_placeholder`, `too_many_extractions`, `unknown_pack`, `pack_conflict`, …). Do not retry unchanged.
  `400 model_not_served`: a server of your own serves one model; use its name or leave `model` out.
- `429 concurrency_limit` and `503 warming_up` / `overloaded`: retry after the `Retry-After` header, with backoff. A
  server of your own answers 503 on `/readyz` and `/v1/score` while it loads and warms up (a few minutes after start).
- **Hosted only**: `401` missing or invalid key; `402 no_credit`; `403 feature_not_enabled` / `media_not_allowed` /
  `model_not_enabled` (account features); `429 rate_limit`; concurrency per account (read `X-Spinf-Concurrency-Limit`
  and send fewer, larger calls). The first call after an idle period can take a few seconds while capacity starts.
  Failed or refused requests are not charged. Quote `request_id` when contacting info@spinf.com.

## Models

| hosted name | public name | card | reads |
|---|---|---|---|
| `spinf-12b` (default) | `spinf/gemma-4-12b-fp8` | one 24 GB GPU | text, images, audio, video frames |
| `spinf-31b` | `spinf/gemma-4-31b-fp8` | one 48 GB GPU | text, images, video frames |
| `qwen3.8-27b-s` | `spinf/qwen3.8-27b-int4` | one 24 GB GPU | text, images, video frames |

Both names work on both servers. **Hosted only**: `spinf-31b` and media are enabled per account. Media parts go in the
content as `image_url`, `input_audio` and `video` (frames the caller sampled, with timestamps; spinf-client samples
them with ffmpeg and resizes to what the model reads); `GET /v1/models` says which kinds a server accepts.
