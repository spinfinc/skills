---
name: spinf-serve
description: Install, run, configure and deploy spinf-serve, the open-source inference engine and server for decision models (one input read once, hundreds of questions scored in parallel on one GPU), and choose its model and GPU. Use when setting up a self-hosted spinf server, writing its config or deployment, sizing hardware or throughput, loading prompt packs into it, or when a self-hosted server misbehaves (startup, readiness, limits, tokens, packs). Calling the API from code is the `spinf` skill.
---

# spinf-serve: the self-hosted server

spinf-serve serves one model on one GPU behind an OpenAI-compatible HTTP API that is also the API of the hosted spinf
service: a client switches between the two by changing the base URL and the key. The server reads a content once and
answers every question about it from that read; scores are **batch-invariant** with the calibrated fp8 scheme (the same
score whatever else is in the batch, bit for bit). Apache 2.0, https://github.com/spinfinc/spinf-serve.

Reference (Markdown, fetch before relying on a flag or default not covered here): `docs/quickstart.md`,
`docs/configuration.md` (models, hardware, every option and limit, trusted callers), `docs/scoring-api.md`,
`docs/prompt-packs.md`, `docs/README.md` for the index, all under https://github.com/spinfinc/spinf-serve/blob/main/.

## Install and run

Requirements: Linux, one NVIDIA GPU with 24 GB (fp8 tensor cores for the batch-invariant scheme: RTX 4090 / 5090, L4,
L40S, RTX 6000 Ada, H100, RTX PRO 6000; Ampere cards run with int8 weights, deterministic but not batch-invariant), a
driver supporting CUDA 13, Python 3.12, `gcc` (kernels compile on the host at first start, then cached), `ffmpeg` for
audio, about 15 GB of disk for the default model.

```sh
pip install "spinf-serve[server]"           # or: uv add "spinf-serve[server]"
spinf-serve serve spinf/gemma-4-12b-fp8     # downloads the checkpoint on first start; serves on port 8000
```

- `GET /readyz` answers 200 once the model is loaded **and warm**: by default the warm-up runs every compile a request
  could pay (about two to two and a half minutes on a 4090 for the 12B after the load). Poll it before sending work;
  `/v1/score` answers `503 warming_up` until then. `GET /healthz` is "process up", `/livez` ignores draining.
- Without `--token` the server is open to whoever can reach the port. Add `--token <secret>` and the client sends
  `Authorization: Bearer <secret>`.
- The exact `torch`, `triton` and `transformers` versions are pinned by the package: the scores come from those kernels.
  Do not override them.

## Models

| name | based on | card | reads |
|---|---|---|---|
| `spinf/gemma-4-12b-fp8` (default) | Gemma 4 12B, FP8 with static activation scales | one 24 GB GPU | text, images, audio, video frames |
| `spinf/gemma-4-31b-fp8` | Gemma 4 31B, FP8 static | one 48 GB GPU | text, images, video frames |
| `spinf/qwen3.8-27b-int4` | Qwen3.8-27B, GPTQ int4 | one 24 GB GPU (the whole card) | text, images, video frames |
| `spinf/pplx-decider-v1-27b-int4` | a decision model with a 255-code readout | one 24 GB GPU | text; its own packs |

The name is the Hugging Face repo (public, under `spinf/`), pinned by revision in the package's manifests, so the first
start downloads exactly the calibrated checkpoint. The hosted names (`spinf-12b`, `spinf-31b`, `qwen3.8-27b-s`) are
aliases, so a client written for the hosted API works unchanged. Another Hub checkpoint of a supported family (Gemma 4,
Qwen3.5) can be served by its path (`spinf-serve serve google/gemma-4-12B`): quantized at load, served under that
name, not batch-invariant.

One GPU per server process; several GPUs are several processes behind your own load balancer. A process never shares
its GPU with another engine.

## Configuration

Settings are layered: flags over `SPINF_*` environment variables over a TOML config (`--config server.toml` or
`SPINF_CONFIG`) over the manifest's defaults. The config has the same keys as the flags, with underscores:

```toml
[engine]
model = "spinf/gemma-4-12b-fp8"
[server]
port = 8010
token = "change-me"
[features]
media_kinds = ["image", "video"]            # or [] for text only (--media none)
packs = ["github:acme/prompt-packs@v1.2", "./my-packs"]
[limits]
max_content_tokens = 8192
```

The flags that matter in practice:

| flag (env) | default | use |
|---|---|---|
| `--device` (`SPINF_DEVICE`) | `cuda:0` | which GPU; with several processes give each its own |
| `--model-dir` (`SPINF_MODEL_DIR`), `SPINF_CACHE_DIRS` | the Hub | a local snapshot, or extra snapshot directories (a shared model volume) |
| `--port`, `--host` | 8000, `0.0.0.0` | |
| `--token` (`SPINF_TOKEN`) | none | require a bearer token on every call |
| `--trusted-token` (`SPINF_ENGINE_TOKEN`) | none | for your own front end: it may send per-call `limits` and `features` and sees engine internals |
| `--warmup full\|short\|none` | `full` | what runs before ready; `short` or `none` trade a faster start for slow first calls |
| `--media image,audio,video` / `--media none` | all the model reads | which content kinds requests may send; `--no-media` also skips loading the embedders (less memory) |
| `--no-extraction`, `--no-generation` | on | turn the `{*}` `{n}` `{$}` slots or free-text generation off |
| `--packs <source>` (repeatable), `--no-official-packs` | the official repo | prompt-pack sources ([below](#prompt-packs)) |
| `--token-budget` | 16,384 | packed tokens per forward pass; the default is the fastest for news-length content on a 4090 |
| `--queue` | 4 | requests accepted while one is on the GPU; beyond it `429 concurrency_limit` with `Retry-After` |
| `--debug` | off | engine internals in every response |

**Limits** (defaults are those of the hosted service; every one is also `SPINF_MAX_…` and a `[limits]` key):
`--max-content-tokens` 4,096 per input, `--max-query-tokens` 256 for a filled template, `--max-prompts-per-input`
10,000, `--max-extractions-per-input` 10, `--max-processed-tokens` 2M per request, `--max-body-bytes` 2 MB,
`--max-options` 1,000 and `--max-option-tokens` 5, `--max-frames` 1,024 per video, `--max-audio-seconds` 600,
`--max-image-pixels` 50 M. To take long content raise `--max-content-tokens` and `--max-tokens` together (tested to 64k;
long inputs run alone in their own pass and the models become less effective on them). `GET /v1/models` reports the
effective features and limits, so a client can size a request before sending.

## Prompt packs

The server preloads the official packs of the model it serves from spinf-prompts at the tag the release pins, and
loads more from `--packs` sources: `github:<owner>/<repo>[@<tag>][#<path>]` (a repo with a `prompt-packs.json`; a tag
or commit is reproducible, a branch is not; `GITHUB_TOKEN` for a private one) or a local folder.

```sh
spinf-serve serve spinf/gemma-4-12b-fp8 --packs github:acme/prompt-packs@v1.2 --packs ./my-packs
spinf-serve packs list --packs ./my-packs --model spinf/gemma-4-12b-fp8   # what would be enabled, what is skipped and why
spinf-serve packs validate ./my-packs                                     # a pack author's check (the spinf-packs skill)
```

Packs written for another model are skipped with a log line. The `spinf/` namespace is reserved for the official
repository (`--packs-allow-namespace spinf` for a locally edited copy). If the official repository cannot be reached the
server logs it and starts without the official packs; a `--packs` source that cannot be read refuses the start.
`GET /v1/packs` lists what is enabled and where each pack came from.

## Deployment notes

- **Readiness and rolling restarts.** Route traffic on `/readyz`. Trusted callers can `POST /v0/drain` to take a server
  out of a balancer (readyz turns 503 while calls in flight finish) and `DELETE /v0/drain` to put it back.
- **Throughput.** One 16k-token forward at a time, inputs packed together: about 10,000 processed tokens per second on
  a 4090 with the 12B on news-length content, roughly 500 two-option questions per second when ~50 questions share a
  content. Size a job as (content tokens + question tokens) × items ÷ 10,000 per second per card. Many inputs per
  call beat many small calls.
- **Scores and fingerprints.** `system_fingerprint` identifies the engine, weights, scheme and GPU architecture that
  produced a score; scores are comparable only under the same fingerprint. With the calibrated fp8 scheme the same
  model gives bit-identical scores on the hosted service and on an Ada card of your own (the fingerprint still changes
  with the engine version); Ampere differs by design. Anything that calibrates on scores (thresholds, packs, baselines)
  must be measured on the server that will run.
- **Memory.** The 12B fp8 fits a 24 GB card with media on; `--no-media` frees the embedders when only text is served.
  The Qwen int4 model needs the whole 24 GB card.
- **Container.** No official image is published yet; a Dockerfile is `FROM` an NVIDIA CUDA 13 runtime image with
  Python 3.12, `pip install "spinf-serve[server]"`, the model directory mounted at `--model-dir` or downloaded at
  first start (`HF_HOME` on a volume), `EXPOSE 8000`, and the health check on `/readyz`.

## Troubleshooting

- `503 warming_up` on every call: the server is loading or compiling; wait for `/readyz`. A first start compiles kernels
  for several minutes; later starts reuse the cache.
- `401 invalid_token`: the server was started with `--token` and the request has none or another.
- `400 model_not_served`: the request names a model the server does not serve; use the served name, an alias, or omit
  `model`.
- `429 concurrency_limit`: more requests than `--queue` allows while one is on the GPU; the client should wait
  `Retry-After` and batch more per call.
- Official packs missing from `GET /v1/packs`: `official_error` in the log says why (offline, a tag not reachable); set
  `SPINF_PACKS_OFFICIAL` to a checkout of spinf-prompts, or point `--packs` at one.
- Scores differ from the hosted API: compare `system_fingerprint`; a different GPU architecture or scheme gives
  different numbers by design, see `docs/configuration.md` and `docs/PRECISION.md`.
- Parity after a change of hardware or package version: `spinf-serve parity --fixture fixtures/wikinews_20261024.json.gz
  --reference fixtures/wikinews_20261024.spinf-gemma-4-12b-fp8.json.gz` from a checkout reproduces the published
  reference bit for bit on an fp8 card.
