---
name: spinf
description: Classify and score content with the spinf scoring API (api.spinf.com) - many questions about a piece of content in one call, a probability for each allowed answer, no generated text. Use when writing, reviewing or debugging code that calls spinf (/v1/score, or /v1/chat/completions with extra_body.scoring), or when designing the questions, answer lists, few-shot examples, thresholds, calibration and aggregation (macro cells) of spinf scores.
---

# spinf scoring API

spinf is a specialized inference provider. It reads a piece of content once and answers many closed questions about
it: for each question you give a **template** ending at a `{?}` slot and the **options** (answers) allowed there;
spinf returns the probability of each option. Nothing is generated, so only input tokens are billed
($0.09 per million, minimum 1,000 tokens per call). Beta: field names may still change.

**Raw scores are not the result.** A single raw `p` carries the bias of its question's wording. Before scores are
used for decisions or analysis, calibrate them (against labelled items or a baseline set) and, for measurements,
combine several cells into a macro cell: see [Calibrate, then aggregate](#calibrate-then-aggregate-macro-cells).

Full docs, as Markdown: https://spinf.com/llms.txt (index), https://spinf.com/docs/scoring-api.md (reference),
https://spinf.com/docs/writing-questions.md, https://spinf.com/docs/calibration.md, https://spinf.com/docs/limits.md,
https://spinf.com/docs/prompt-packs.md (ready-made packs such as content moderation).
Fetch the reference before relying on a field or limit not covered here.

## Setup

- The API key (`ssk-…`) lives in the `SPINF_API_KEY` environment variable. Keys are created at
  https://spinf.com/console/api-keys and shown only once. Never write a key into code, config, tests, logs or commits,
  and never ask the user to paste it into the chat: read it from the environment.
- Call spinf from servers, scripts or jobs only: the API does not accept browser (CORS) calls.
- Check the key and the balance: `curl -s https://api.spinf.com/v1/account -H "Authorization: Bearer $SPINF_API_KEY"`.
  `GET https://api.spinf.com/v1/models` (no key) lists the models, prices and the per-call minimum.
- No SDK is needed: plain HTTP, or the OpenAI SDK pointed at `https://api.spinf.com/v1`.

## A call

```bash
curl https://api.spinf.com/v1/score \
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

With the OpenAI SDK (Python), the same body goes to `/v1/chat/completions`, the scoring in `extra_body`:

```python
import os
from openai import OpenAI

client = OpenAI(base_url="https://api.spinf.com/v1", api_key=os.environ["SPINF_API_KEY"])
resp = client.chat.completions.create(
    model="spinf-12b",
    messages=[{"role": "user", "content": "Customer support message:\n" + ticket_text}],
    extra_body={"scoring": {"queries": [...]}},
)
results = resp.model_extra["results"]  # choices is empty: nothing is generated
```

Response shape (trimmed; the values are illustrative):

```json
{ "model": "spinf-12b", "system_fingerprint": "fp_…",
  "results": [ { "input_id": "0", "queries": [ { "id": "topic", "combinations": [ {
      "values": {},
      "options": [ { "text": " billing", "logprob": -0.9, "p": 0.97, "floor_p": 0.31 }, … ],
      "residual": 0.42, "floor_residual": 0.55 } ] } ] } ],
  "warnings": [],
  "usage": { "prompt_tokens": 131, "billed_tokens": 1000,
             "prompt_tokens_details": { "content_tokens": 14, "query_tokens": 60, "floor_tokens": 57 } } }
```

## Rules for a question (the API refuses the rest with a 400)

- The template is read **right after the content**: start it with `"\n\n"`, and put exactly one `{?}`, at the very
  end. Up to 5 `{name}` placeholders; `{{` / `}}` for literal braces. A filled template is at most 256 tokens.
- Write each option as it would continue the template, **with a leading space** (`" yes"`), and no space before
  `{?}` (`"…Answer:{?}"`, not `"…Answer: {?}"`, which fails with `option_boundary`).
- 1 to 1,000 options per question, 1 to 5 tokens each, no duplicates (case-insensitive by default).
- One content (`messages`, or one entry of `inputs`) is up to 4,096 tokens; roles are ignored, the text is read as
  it is. Up to 10,000 questions (templates × combinations) per content.
- Placeholders are filled from `combinations`: a list of `{name: value}` objects, or `"all"` with
  `"values": {name: [...]}` for the full grid (up to 1,000 per question).
- Request body up to 2 MB, up to 2M processed tokens.

## Many contents in one call

Send `inputs` instead of `messages`: `[{ "id": "t1", "messages": [...] }, …]`. Every input is scored with all of
`scoring.queries`, or only the ids in its `query_ids`; an input can add its own `queries`. Results come back one per
input, in request order, with `input_id`. Batch whenever there is more than one item: the 1,000-token minimum is per
call, and each question's floor is computed once per call. Size batches to stay under 2 MB.

## Reading the scores

- `p`: the probability of each option among the options listed (they sum to 1). Decide on `p`, with a threshold.
- `floor_p`: the same question scored with **empty** content, the question's own bias. The lift `p - floor_p` is
  what the content says. Without few-shot examples, the lift (or the option with the largest lift) often classifies
  better than raw `p` on multi-choice questions. With examples in the content, use `p` as it is.
- `residual`: the probability mass on words other than the options. Mostly set by the template; close to 1 for
  multi-token options. Do not judge a question by its size; compare it only across contents under the same question.
  A `low_option_mass` warning means the options are rarely the model's next words: reword the template or options.
- The floor depends only on the question: after a first call, keep its `floor_p` and send
  `"scoring": {"empty_floor": false, …}` to skip it (not computed, not billed). Recompute it when
  `system_fingerprint` changes.
- Scores are comparable over time only under the same template, options, examples and `system_fingerprint`.

## Calibrate, then aggregate (macro cells)

A **cell** is one score: one question (a template with one combination of its placeholders) on one content. A single
cell is noisy and carries the bias of its wording, so do not report or threshold raw cells on their own.

1. **Calibrate each cell.** Score a baseline set of contents of the same kind (e.g. a month of articles from the same
   sources, a random sample of tickets). For each cell keep the distribution of its lift `p - floor_p` (or of `p` when
   the content has few-shot examples), and express every new score as a z-score or percentile of that distribution.
   Calibrated cells are comparable across questions, wordings, sources and time; raw ones are not.
2. **Build macro cells.** Ask the thing you care about through several cells: 3 to 5 wordings of the question, the
   same question over related placeholder values, and related questions that point the same way (flip the sign of
   those that point the opposite way). Average their calibrated scores into one macro cell. Each cell's wording bias
   and noise partly cancel, so the macro cell is better calibrated and more stable than any of its cells. Extra cells
   are cheap: the content is read once per call, and each cell costs only its own template and option tokens.
3. **Aggregate.** Average macro cells over contents (per day, entity, product or segment), then standardise each
   aggregate on its own trailing window. This is how the Thinking Text indexes turn about 500 questions over millions
   of articles into daily reads (https://spinf.com/docs/calibration.md).
4. **For per-item decisions**, calibrate on labelled items instead: pick the threshold on `p`, or on a macro cell's
   score, that gives the precision and recall you need, and send the uncertain band to review.

Keep a macro cell's cells, templates, options, examples and baseline fixed once in use, and recalibrate when
`system_fingerprint` changes. Design the macro cells with the user when a project starts: which concepts, which
wordings, which baseline.

## Designing a classifier

1. **Label the content.** Start it with what it is: `Customer review:`, `User post:`, `News article:`.
2. **Name the answers in the question** and make the options those words: short, distinct, one or two words each.
   Put definitions in the question (`"Is this urgent (business blocked, security, …)?"`). For a long label list,
   ask the broad category first, then the label within it.
3. **Add 4 to 8 labelled examples** in the content, before the item, in the same format as the question, separated
   by `---`; cover every answer and the hard cases. This is the biggest lever on accuracy. Examples are content, so
   they are billed with every input.
4. **Measure before scaling.** Score 100–300 items with known labels (never the ones used as examples), try two or
   three wordings and 0 / 4 / 8 examples, keep the best, and pick thresholds (e.g. act above 0.9, send 0.5–0.9 to
   review). Then freeze the template, options and examples.
5. **Ask everything in one read.** Each extra question costs only its own tokens; the content is read once per call.
   Use that for the checks you would otherwise skip, and for the extra wordings that make up macro cells.

## Prompt-packs (content moderation)

A pack is a ready-made, calibrated set of questions hosted by spinf. When the job is content moderation, use the
`spinf/moderation` pack instead of writing moderation questions, and add the user's own questions to the same call:
the content is read once and billed once for both.

```json
"scoring": {
  "packs": [ { "id": "spinf/moderation", "context": "forum" } ],
  "queries": [ { "id": "on_topic", "template": "\n\nIs this post about cooking?\nAnswer:{?}", "options": [" yes", " no"] } ]
}
```

- `GET https://api.spinf.com/v1/packs` (no key) lists the packs: versions, `contexts` (moderation: `ai` for messages
  to an assistant, `forum` for posts on a platform), the categories with their default `threshold` (`null` = reported,
  never decides) and the decision modes. Read thresholds from there; don't hardcode them. Pin `version` in production.
- Response: `results[].packs[]` = `{id, version, context, categories: {<id>: {score, percentile, flagged}},
  decision}`. `score` is calibrated 0–1 and comparable across categories; `percentile` places it on normal traffic;
  `flagged` = over the category's threshold (`null` when it has none). `decision.unsafe` is the verdict.
- `decision.mode`: `micro_layer` (default, a layer trained on labelled data), `per_category` (any category over its
  threshold; override them with `decision.thresholds`) or `max` (the highest category against one `threshold`).
  `decision.critical` (0.1.3 default `["selfharm", "child"]`; 0.1.1 none) lists categories that make the content unsafe
  whenever they are flagged, in every mode; the response then has `decision.critical` and, when that changed the
  outcome, `"by": "critical"`. Send `"critical": []` to turn it off.
  Choose thresholds on the user's own labelled sample, as for any question.
- Rules: text only (media parts → `400 pack_media_not_supported`); `spinf-12b` only; leave
  `scoring.case_insensitive` out (the pack sets it, `400 pack_conflict`); the whole content is wrapped in the pack's
  context, and the user's own questions read it wrapped too; don't use the `tax.` / `gen.` id prefixes.
- Cost: no surcharge; moderation adds 1,626 question tokens per content (about 1,710 billed tokens for a typical
  message, content included).

## Cost

Billed tokens per call = each content once + per question the filled template and each option's tokens after its
first + the same for the floor (once per distinct question per call), with a minimum of 1,000. Price:
$0.09 per million tokens; the scores are free. Estimate the cost of a large run (tokens per item ×
items) and tell the user before starting it. New accounts get $4.50 of free credit.

## Errors and retries

Errors are `{"error": {"type", "code", "message", "param"}, "request_id"}`; `param` points at the bad field.

- `400`: fix the request (`invalid_template`, `option_boundary`, `option_too_long`, `content_too_long`,
  `unknown_placeholder`, …). Do not retry unchanged.
- `401`: missing or invalid key. `402 no_credit`: the balance is under $0.10, add credit in the console.
- `429 concurrency_limit` / `rate_limit` and `503 warming_up` / `overloaded`: retry after the `Retry-After` header,
  with backoff. The first call after an idle period can take a few seconds while capacity starts.
- Concurrency is 1 request at a time per $100 of balance (at least 1) and 600 requests per minute; read
  `X-Spinf-Concurrency-Limit` and send fewer, larger batched calls rather than many parallel small ones.
- Failed or refused requests are not charged. Quote `request_id` when contacting info@spinf.com.

## Models

`spinf-12b` (default): Gemma 4 12B, optimized by spinf for scoring; also accepted as `google/gemma-4-12B`.
`spinf-31b` (Gemma 4 31B) is enabled on request. Images, video frames and audio are in preview, enabled per account on
request (see the reference).
