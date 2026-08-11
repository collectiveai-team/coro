<p align="center">
  <img src="https://raw.githubusercontent.com/collectiveai-team/coro/main/assets/coro-logo.png" alt="Coro — OpenAI- and Deepgram-compatible ASR + speaker diarization" style="width:600px; max-width:100%; height:auto;" />
</p>

<p align="center">
  <em>Self-hosted speech-to-text that knows who said what — speaks both the OpenAI and Deepgram API contracts.</em>
</p>

<p align="center">
  <a href="https://github.com/collectiveai-team/coro/releases"><img alt="Release" src="https://img.shields.io/github/v/release/collectiveai-team/coro?logo=github" /></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" alt="Python 3.12"></a>
  <a href="https://platform.openai.com/docs/api-reference/audio"><img src="https://img.shields.io/badge/API-OpenAI--compatible-412991?logo=openai&logoColor=white" alt="OpenAI-compatible API"></a>
  <a href="https://developers.deepgram.com/reference/speech-to-text-api/listen"><img src="https://img.shields.io/badge/API-Deepgram--compatible-13EF93?logo=deepgram&logoColor=black" alt="Deepgram-compatible API"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

---

**Source Code**: [https://github.com/collectiveai-team/coro](https://github.com/collectiveai-team/coro)

---

Coro is an embedded ASR + speaker-diarization server that speaks **two industry
API contracts natively** — OpenAI's and Deepgram's. Point the official `openai`
SDK *or* the official `deepgram-sdk` at it and get back typed transcripts that
know *who* said *what*, no custom schema package needed.

Each provider gets its own endpoint implementing that provider's own contract.
Coro never bolts one vendor's data onto another vendor's format:

| you already use | point it at | and you get |
|---|---|---|
| `openai` SDK | `POST /v1/audio/transcriptions` | `Transcription` / `TranscriptionVerbose` / `TranscriptionDiarized`, plus OpenAI-exact SSE |
| `deepgram-sdk` | `POST /v1/listen` | `ListenV1Response` — **a speaker on every word** |
| `deepgram-sdk` | `WebSocket /v1/listen` | live `Results` / `Metadata` frames |

Responses are validated against both vendors' own published SDK types in CI, so
"compatible" is asserted rather than asserted-in-prose.

The name nods to *coro* (Spanish for "chorus") — many voices, transcribed and
attributed to who spoke them.

The key features are:
- **Two native API contracts** — OpenAI *and* Deepgram, each on its own endpoint with its own request shape, defaults and error format; neither is an approximation of the other
- **OpenAI-compatible API** — drop-in `/v1/audio/transcriptions`; clients reuse the official `openai` SDK types (`Transcription` / `TranscriptionVerbose` / `TranscriptionDiarized`) with no custom schema
- **Deepgram-compatible API** — drop-in `POST /v1/listen` and `WebSocket /v1/listen`; the only way to get **per-word speaker labels**, since no OpenAI type has a slot for one
- **Audio *and* video input** — uploads are decoded through ffmpeg, so any container it supports works: audio (`.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, …) and video (`.mp4`, `.mkv`, `.mov`, `.webm`, …); the audio track is extracted to 16 kHz mono PCM automatically — same endpoint, same response shapes
- **Pluggable diarization backends** — pick per deployment: NVIDIA NeMo Sortformer (streaming-capable, **≤ 4 speakers**) or pyannote community-1 (batch/whole-file, **handles > 4 speakers**); both attribute every segment to a speaker (`diarized_json`), so you get *who spoke, when, and what*
- **Pluggable ASR backends** — pick per deployment: onnx-asr Parakeet (the default — fastest on CPU *and* GPU, strongest on Spanish), Faster-Whisper (best English meeting accuracy, multilingual), or onnx-genai Nemotron (real-time streaming)
- **Two transcription pipelines** — `full-memory` (default) decodes and holds the whole recording in RAM for lowest latency on short/medium clips; `streaming` streams 1 s PCM chunks off disk and spills the growing transcript to a per-request on-disk store, trading a little latency for **flat host RAM on arbitrarily long audio**. Select with `CORO_PIPELINE` / `--pipeline` — see [the pipeline comparison](#two-transcription-pipelines-full-memory-vs-streaming)
- **Streaming both ways** — OpenAI-exact SSE (`transcript.text.delta` / `transcript.text.done` / `[DONE]`) with `stream=true`, *and* a Deepgram-compatible WebSocket at `/v1/listen` that pushes `Results` frames as audio arrives
- **Flat-memory long audio** — the streaming pipeline spills the transcript to disk so host RSS stays flat from 11 s to multi-hour recordings
- **CPU & GPU** — mutually-exclusive `cpu` / `cuda` extras carry the matching `onnxruntime` wheels; multilingual on either
- **Run it your way** — ephemeral `uvx`, a standalone `uv tool install` command, or a full `uv sync` dev checkout

## Quickstart

Run the server without installing it into a project, straight from the repo,
using `uvx` (the alias for `uv tool run`). Pick the hardware extra that matches
your machine:

```bash
# CPU-only
uvx --from "coro-asr[cpu]" coro --port 8000

# NVIDIA GPU
uvx --from "coro-asr[cuda]" coro --port 8000
```

`uvx` builds a throwaway isolated environment and launches the `coro` command —
no `uv sync`/`uv run` and nothing added to your current project. The server now
speaks both the OpenAI and Deepgram transcription contracts at
`http://127.0.0.1:8000/v1`.

Then write a tiny client with the official `openai` SDK, pointing `base_url` at
your Coro server (`api_key` is required by the SDK but ignored by Coro):

```bash
pip install "openai>=2.0.0"     # or: uv pip install "openai>=2.0.0"
```

```python
from openai import OpenAI

# Point the OpenAI client at your Coro server instead of api.openai.com.
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

with open("audio.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        file=f,
        model="whisper-1",  # accepted but ignored; server uses its backend
        response_format="diarized_json",  # json | verbose_json | diarized_json
    )

print(result.text)
for segment in result.segments:  # who spoke, when, and what
    print(f"[{segment.start:.2f}-{segment.end:.2f}] {segment.speaker}: {segment.text}")
```

Or hit the endpoint directly with `curl` (the same OpenAI multipart contract):

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=whisper-1 \
  -F response_format=diarized_json
```

That's the whole integration — because Coro returns standard OpenAI shapes, the
SDK parses the response into typed objects with no custom schema. See
[Client integration](#client-integration) for streaming (SSE) and the full
format ↔ type mapping.

## Standalone install

To run Coro as a server (not hack on it), install it as an isolated CLI tool
with `uv tool install`. This puts `coro` and `coro-bench` on your `PATH` —
no clone, no project environment. Pick the hardware extra that matches your
machine (`cpu` / `cuda` are mutually exclusive):

```bash
uv tool install "coro-asr[cpu]"    # CPU-only
uv tool install "coro-asr[cuda]"   # NVIDIA GPU
```

Then run the server directly (no `uv run`):

```bash
coro --port 8000
```

Upgrade with `uv tool upgrade coro`; uninstall with `uv tool uninstall coro`.
For a throwaway run without installing at all, use `uvx` (see
[Quickstart](#quickstart)). On a GPU host the `coro-asr[cuda]` build still needs the
`libcublas.so.12` loader-path fix — see [GPU on a bare host](#gpu-on-a-bare-host).

## Run with Docker

Prebuilt images are published to GHCR with `-cpu` / `-gpu` flavour suffixes
(`latest`, the release version, and `sha-…` tags). The image entrypoint is
`coro`, so append any `--flag` or `CORO_*` env var just like the CLI; the server
binds `0.0.0.0:8000` inside the container.

```bash
# CPU
docker run --rm -p 8000:8000 \
  ghcr.io/collectiveai-team/coro:latest-cpu \
  --backend-asr onnx-asr --model-asr nemo-parakeet-tdt-0.6b-v3 --asr-device cpu \
  --backend-diarization nemo

# NVIDIA GPU (needs the NVIDIA Container Toolkit)
docker run --rm --gpus all -p 8000:8000 \
  ghcr.io/collectiveai-team/coro:latest-gpu \
  --backend-asr onnx-asr --model-asr nemo-parakeet-tdt-0.6b-v3 \
  --backend-diarization nemo
```

The `--backend-diarization nemo` flag turns on Sortformer speaker labels; omit it
for an ASR-only server. The diarizer device defaults to `auto` (GPU when one is
available), so you only need `--diarization-device` to pin it explicitly.

Cache downloaded model weights across runs by mounting a Hugging Face cache
volume (avoids re-downloading on every container start):

```bash
docker run --rm -p 8000:8000 \
  -v coro-hf-cache:/root/.cache/huggingface \
  ghcr.io/collectiveai-team/coro:latest-cpu --port 8000
```

To build the image yourself instead of pulling, pass the matching
`CORE_IMAGE` / `EXTRA` build args (see the [Dockerfile](Dockerfile)):

```bash
# CPU
docker build -t coro:cpu \
  --build-arg CORE_IMAGE=ubuntu:noble --build-arg EXTRA=cpu .

# NVIDIA GPU
docker build -t coro:gpu \
  --build-arg CORE_IMAGE=nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04 \
  --build-arg EXTRA=cuda .
```

## Configuration

Coro can be configured two equivalent ways — use whichever fits your
deployment, or mix both:

- **Environment variables** — `CORO_`-prefixed (host, port, backends, devices,
  etc.).
- **CLI flags** — every setting is also a `--kebab-case` flag, auto-derived
  from `ServerSettings` via pydantic-settings. Run `coro --help` to list them.

Each `ServerSettings` field maps to both forms, e.g. `backend_asr` →
`CORO_BACKEND_ASR` (env) or `--backend-asr` (CLI). Precedence is **CLI flags >
environment variables > defaults**. See `coro/settings.py` for the full list.

```bash
# Env vars (add CORO_BACKEND_DIARIZATION to enable speaker labels; omit for ASR-only)
CORO_BACKEND_ASR=onnx-asr CORO_MODEL_ASR=nemo-parakeet-tdt-0.6b-v3 \
  CORO_ASR_DEVICE=cuda CORO_BACKEND_DIARIZATION=nemo \
  coro --port 8000

# Equivalent CLI flags
coro --backend-asr onnx-asr --model-asr nemo-parakeet-tdt-0.6b-v3 \
  --asr-device cuda --backend-diarization nemo --port 8000
```

The diarizer device defaults to `auto` (GPU when available); add
`--diarization-device` only to pin it. Drop `--backend-diarization` for an
ASR-only server, or swap `nemo` → `pyannote` (`--pipeline full-memory`, needs
`--extra diar-pyannote` and an HF token) for > 4 speakers — see
[Diarization backends](#diarization-backends).

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Readiness / capability status. |
| `POST` | `/v1/audio/transcriptions` | OpenAI-compatible transcription (multipart). |
| `POST` | `/v1/listen` | Deepgram-compatible transcription (raw body); per-word speakers. |
| `WS`   | `/v1/listen` | Deepgram-compatible live transcription; `Results` frames as windows complete. |

`response_format` accepts `json`, `verbose_json`, and `diarized_json`. With
`stream=true` the endpoint emits OpenAI-exact SSE
(`transcript.text.delta` / `transcript.text.done` / `[DONE]`).

For a **speaker on every word**, use the Deepgram-native `POST /v1/listen`
below — no OpenAI type has a slot for one.

### API docs and contracts

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/docs` | Scalar API reference — both contracts behind one document picker. |
| `GET`  | `/openapi.json` | OpenAPI 3.1 contract for the request/response surface. |
| `GET`  | `/asyncapi.json` | AsyncAPI 3.0 contract for the SSE stream and the `/v1/listen` socket. |

Both documents are generated from the code — OpenAPI by FastAPI from the routes,
AsyncAPI from the same types the SSE writer and the WebSocket handler serialise
— so neither is hand-maintained and neither is committed to the repo. The
socket is published only as AsyncAPI, because OpenAPI cannot describe one. CI exports both, lints
them with `redocly`, and fails a PR that breaks the REST contract. Swagger UI
and ReDoc are switched off: Scalar is the only renderer that can show the
event-driven contract alongside the REST one. See
[ADR 0013](docs/adr/0013-published-api-contracts.md).

## Two transcription pipelines (full-memory vs streaming)

Coro ships two interchangeable pipelines behind the same OpenAI endpoint and
response shapes; switch between them with `CORO_PIPELINE` / `--pipeline`
(default `full-memory`). They differ only in *how* the audio and transcript are
held in memory — the wire format you get back is identical.

- **`full-memory` (default)** — ffmpeg decodes the upload to PCM **once, in
  full**, and the pipeline holds the entire signal plus the accumulated
  tokens/segments/words in RAM. Simplest and lowest-latency for short to medium
  clips, but **host RAM grows ~linearly with recording length**, so it is not
  suited to unbounded audio. It is the only pipeline that works with the
  whole-file `pyannote` diarizer.
- **`streaming`** — ffmpeg streams 1 s PCM chunks off disk instead of buffering
  the whole recording, and the growing transcript spills to a per-request
  on-disk SQLite (WAL) store instead of Python lists. Consumed over **SSE
  (`stream=true`)** it keeps **flat peak host RSS, independent of recording
  length** (11 s ≈ 58 min ≈ multi-hour): only bounded working buffers stay
  resident and the final `transcript.text.done` frame is rendered straight from
  the store one segment/word at a time. This is the **only** pipeline that can
  diarize live as audio arrives, and it requires a streaming-capable backend
  (NeMo Sortformer for diarization).

| | `full-memory` | `streaming` |
|---|---|---|
| Audio decode | whole recording at once | 1 s PCM chunks off disk |
| Transcript storage | in-RAM Python lists | per-request on-disk SQLite (WAL) |
| Host RAM vs length | grows ~linearly | **flat** (over SSE) |
| Live/incremental output | ❌ (one final response) | ✅ over SSE |
| Diarization backends | `nemo` *or* `pyannote` | `nemo` only (Sortformer) |
| Best for | short/medium clips, > 4-speaker pyannote | long/unbounded audio |

The spill directory needs to be on real disk, and that is handled for you: at
startup the server picks the system temp dir when it is real disk, otherwise a
directory under your cache dir, because `/tmp` is tmpfs (RAM-backed) on most
Linux distributions and spilling there would defeat the spill. Override it with
`CORO_TRANSCRIPT_SPILL_DIR`; pointing it at a RAM-backed path fails startup
rather than silently costing you the flat-RAM property. See
[Benchmarks](#benchmarks) for the measured memory behaviour.

## ASR backends

The ASR backend is pluggable behind a single adapter contract. Select it with
`CORO_BACKEND_ASR` + `CORO_MODEL_ASR`; pick the device with
`CORO_ASR_DEVICE` (`auto` | `cpu` | `cuda`).

| Backend (`CORO_BACKEND_ASR`) | Runtime | Typical model (`CORO_MODEL_ASR`) | Notes |
|---|---|---|---|
| `onnx-asr` | onnxruntime | `nemo-parakeet-tdt-0.6b-v3` | **Default.** NeMo Parakeet/Canary; multilingual, and strongest of the three on Spanish. Offline (batched) → very high GPU throughput. Leave `CORO_ASR_QUANTIZATION` unset (fp32) — `int8` saves memory but does *not* go faster here. |
| `faster-whisper` | CTranslate2 | `openai/whisper-medium` | Best English meeting accuracy; multilingual. `CORO_ASR_COMPUTE_TYPE` = `int8` (CPU) / `float16` (GPU). |
| `onnx-genai` | onnxruntime-genai | `onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4` | NVIDIA Nemotron **cache-aware streaming**; 40 locales. Built for low-latency real-time, not batch throughput. Timestamps are 560 ms-resolution. GPU strongly recommended. |

### Recommended configuration

Each setting below is shown as an env var; the equivalent CLI flag is the
`--kebab-case` form (e.g. `--backend-asr onnx-asr`).

The defaults (`onnx-asr` + `nemo-parakeet-tdt-0.6b-v3`, fp32) are already the
recommended configuration on both CPU and GPU — you only need the settings below
if you want to move off them.

**GPU (`--extra cuda`):**
```bash
CORO_ASR_DEVICE=cuda           # fp32 (leave CORO_ASR_QUANTIZATION unset)
```
Or as a CLI flag:
```bash
coro --asr-device cuda --port 8000
```
Fastest by a wide margin with near-best accuracy. Use `faster-whisper` +
`float16` if you want the top English-meeting accuracy point; use `onnx-genai`
only for real-time low-latency streaming.

**CPU (`--extra cpu`):** nothing to set — the default selection is the CPU pick.
Measured against `faster-whisper` + `openai/whisper-medium` on the same host:
**~8.8× the throughput**, **23% lower Spanish WER**, ~1 GB less resident memory,
and English meeting WER within noise. `onnx-genai` is not recommended on CPU.

> **Do not reach for `int8` for speed.** For this transducer model int8 measured
> **no throughput gain** (+0.6% / −3.6% across two workload sets — inside noise)
> and cost 3–6% relative WER. Its real benefit is memory: it drops the resident
> server from ~2.7 GB to ~1.3–1.7 GB. Set `CORO_ASR_QUANTIZATION=int8` to fit a
> memory budget, never to go faster. Full numbers:
> [docs/benchmark.md](docs/benchmark.md#quantization-int8-is-a-memory-tool-not-a-speed-tool).

**Streaming on long audio:** set `CORO_PIPELINE=streaming` so the per-request
transcript spills to disk and host RSS stays flat regardless of recording
length; the spill directory defaults to real disk automatically. Consume the
result over SSE (`stream=true`).

## Diarization backends

Diarization is **optional** (default `none` — an ASR-only server is valid) and
pluggable behind a single `DiarizationAdapter` contract, dispatched by a
per-capability Backend Adapter Factory (see ADR 0007). Select it with
`CORO_BACKEND_DIARIZATION` + `CORO_MODEL_DIARIZATION`; pick the device with
`CORO_DIARIZATION_DEVICE` (`auto` | `cpu` | `cuda`).

> **Off by default is a deliberate choice, not an oversight.** Turning
> Sortformer on costs ~24% throughput and ~1 GB of resident memory, adds a
> ~500 MB download on first start, and caps the server at 4 speakers — so it is
> opt-in rather than a default that could silently mis-attribute 5-speaker
> audio. Enabling it is one setting: `CORO_BACKEND_DIARIZATION=nemo`. Rationale
> and numbers: [docs/benchmark.md](docs/benchmark.md#diarization-by-default-stays-off).

| Backend (`CORO_BACKEND_DIARIZATION`) | Default model | Model licence | Speakers | Streaming | Gated / token | Install |
|---|---|---|---|---|---|---|
| `nemo` | `nvidia/diar_streaming_sortformer_4spk-v2` | CC-BY-4.0 | **≤ 4** (4-speaker Sortformer) | ✅ works with `CORO_PIPELINE=streaming` | no | core install |
| `pyannote` | `pyannote/speaker-diarization-community-1` | CC-BY-4.0 | **unbounded** — handles **> 4** | ❌ batch/whole-file only | **yes — Hugging Face token required** | `--extra diar-pyannote` |

**Which to pick:**

- **NeMo Sortformer** — choose for ≤ 4 speakers and/or when you need the
  **streaming pipeline** (Sortformer is the only streaming-capable backend). The
  `diar_streaming_sortformer_4spk-v2` model is **designed for at most 4
  speakers**; on meetings with more than 4 distinct speakers it will collapse the
  extras and DER degrades.
- **pyannote community-1** — choose when a recording may contain **more than 4
  speakers**. It clusters speakers over the **whole file**, so it is **batch-only**
  and is rejected at startup if you select `CORO_PIPELINE=streaming` (use
  `full-memory`). The model is **gated**: you must accept its conditions on the
  Hugging Face model page and provide a token.

### Model licensing

`coro-asr` itself is MIT (see [`LICENSE`](LICENSE)), but **model weights carry
their own licences** and you are responsible for complying with them. Every
diarization model this project names:

| Model | Licence | Commercial use | Used by `coro-asr` |
|---|---|---|---|
| [`nvidia/diar_streaming_sortformer_4spk-v2`](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) (streaming Sortformer) | **CC-BY-4.0** | ✅ permitted, with attribution | ✅ default for `--backend-diarization nemo` |
| [`nvidia/diar_sortformer_4spk-v1`](https://huggingface.co/nvidia/diar_sortformer_4spk-v1) (batch Sortformer) | **CC-BY-NC-4.0 — non-commercial only** | ❌ **not permitted** | ❌ never a default; named here only as the earlier, offline-only Sortformer |
| [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) | **CC-BY-4.0** (gated — accept conditions + token) | ✅ permitted, with attribution | ✅ default for `--backend-diarization pyannote` |

The NeMo backend accepts any Sortformer checkpoint via `CORO_MODEL_DIARIZATION`,
so `nvidia/diar_sortformer_4spk-v1` *will* load if you ask for it explicitly —
but it is **CC-BY-NC-4.0**, so doing so makes your deployment non-commercial.
Leave `CORO_MODEL_DIARIZATION` unset to get the permissively licensed streaming
default. When adding a new diarization model to this project, add its licence to
the table above.

### NeMo Sortformer setup (default, no token)

Sortformer ships with the **core install** — no extra dependency, no Hugging
Face token. Just turn the backend on; the default model
(`nvidia/diar_streaming_sortformer_4spk-v2`) is selected automatically and
downloaded on first run.

```bash
# Batch (full-memory pipeline, the default) — env-var form
CORO_BACKEND_DIARIZATION=nemo coro --port 8000
# equivalent CLI form:
coro --backend-diarization nemo --port 8000
```

Combine with an ASR backend and pin the device as usual:

```bash
coro --port 8000 \
  --backend-asr onnx-asr --model-asr nemo-parakeet-tdt-0.6b-v3 \
  --backend-diarization nemo --diarization-device cuda
```

Sortformer is the **only streaming-capable** backend. To diarize live as audio
arrives, switch the pipeline to `streaming` (optionally tune the latency tier):

```bash
coro --port 8000 \
  --backend-diarization nemo \
  --pipeline streaming \
  --diarization-latency very-high   # very-high | high | low | ultra-low
```

Either way, request `response_format=diarized_json` to get per-segment speaker
labels back, or `POST /v1/listen?diarize=true` for per-*word* labels.
Sortformer handles **≤ 4 speakers**; for more, use pyannote below.

#### Sortformer post-processing (optional, see ADR 0010)

Sortformer's raw speaker-activity predictions go through a threshold-based
post-processing step (onset/offset/padding/min-duration). Left unset, coro
uses NeMo's own unconfigured baseline — no smoothing, no padding. Set
`CORO_DIARIZATION_POSTPROCESSING` to one of NeMo's own published presets, or
to a path to a custom YAML in the same schema, to override it:

```bash
coro --backend-diarization nemo --diarization-postprocessing dihard3-dev
coro --backend-diarization nemo --diarization-postprocessing none  # explicit baseline
```

| Preset | Optimized on | Target scoring collar | NVIDIA's domain description |
|---|---|---|---|
| `dihard3-dev` | DIHARD III dev split | 0 s | Diverse, challenging recordings across many conditions |
| `callhome-part1` | CALLHOME (NIST SRE 2000 Disc8) | 0.25 s | Telephone conversations |

**Neither preset is a coro recommendation.** They are NVIDIA's own published
values for two specific domains; whether either is a good fit for *your*
deployment's audio depends on how close your traffic is to one of those
domains — coro does not know that, and does not compute or tune numbers
against any benchmark on your behalf. If you have a representative sample of
your own traffic to validate against, supply your own YAML in the same
`parameters:` schema instead.

**The target collar is part of the parameter set, not a footnote.** Zero-collar
scoring rewards boundary precision and near-zero padding; collar-tolerant
scoring rewards generous padding and aggressive short-segment deletion. Scoring
a set against the collar it was not tuned for measures the mismatch, not the
model. `coro-bench-diar` therefore defaults to `--postprocessing auto`, which
picks the preset matching its `--collar`; pass an explicit preset name to
override, or `none` for NeMo's baseline.

**If you A/B the presets yourself, record which reference you scored against.**
Which preset wins depends on whether the model is missing speech or inventing
it, and that split is a property of the reference as much as of the model — two
defensible references for the same corpus can agree on total DER while
disagreeing about the direction of the error. A ranking that holds under only
one reference is not yet a reason to change a default.

##### Speaker-count gate

When post-processing is enabled it is applied only when the estimated speaker
count is at or below `CORO_DIARIZATION_POSTPROCESSING_MAX_SPEAKERS` (default
`4`); above it, that recording falls back to NeMo's baseline. NVIDIA's own v2
results show these thresholds improve DER for four or fewer speakers and
consistently *degrade* it at five or more, because short-segment deletion
removes the brief, fragmentary evidence the model has for the extra speakers —
so applying them unconditionally makes the worst case worse.

**This gate cannot fire on any currently shipped Sortformer revision.** They
are all 4-speaker models emitting a `T x 4` activity matrix, so the estimate
can never exceed 4. It is implemented now so the behaviour is already correct
if a >4-speaker Diarization Model Selection is configured later, and the
ceiling is a setting rather than a constant for the same reason.

### pyannote setup (gated model + token)

1. Install the optional dependency (kept out of the core install):

   ```bash
   uv sync --extra cpu --extra diar-pyannote   # or: --extra cuda --extra diar-pyannote
   ```

2. Accept the user conditions for
   [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
   on Hugging Face, then provide a token. Any of these is read (and the value is
   masked in logs); `.env` is loaded automatically:

   ```bash
   # .env (auto-loaded), or any of these env vars:
   HF_TOKEN=hf_xxx                 # standard HF name
   HUGGING_FACE_HUB_TOKEN=hf_xxx   # standard HF name
   CORO_HF_TOKEN=hf_xxx            # coro-namespaced
   ```

3. Run with the full-memory pipeline:

   ```bash
   CORO_BACKEND_DIARIZATION=pyannote CORO_PIPELINE=full-memory coro --port 8000
   # equivalent CLI: coro --backend-diarization pyannote --pipeline full-memory
   ```

> Without a valid token (or before accepting the model conditions) the pyannote
> pipeline fails to load at startup with an actionable error.

### Settings reference

Every setting below is available as both an environment variable and a CLI
flag (CLI flags take precedence). Source of truth: `coro/settings.py`.

| Env var | CLI flag | Default | Description |
|---|---|---|---|
| `CORO_HOST` | `--host` | `0.0.0.0` | Bind host. |
| `CORO_PORT` | `--port` | `8000` | Bind port. |
| `CORO_CORS_ORIGINS` | `--cors-origins` | `["*"]` | Allowed CORS origins. |
| `CORO_PIPELINE` | `--pipeline` | `full-memory` | Transcription pipeline selector (`full-memory` \| `streaming`). |
| `CORO_BACKEND_ASR` | `--backend-asr` | `onnx-asr` | ASR backend provider (`faster-whisper` \| `onnx-asr` \| `onnx-genai`). |
| `CORO_MODEL_ASR` | `--model-asr` | `nemo-parakeet-tdt-0.6b-v3` | ASR model selection. |
| `CORO_ASR_DEVICE` | `--asr-device` | `auto` | ASR device (`auto` \| `cuda` \| `cpu`). |
| `CORO_ASR_COMPUTE_TYPE` | `--asr-compute-type` | `default` | Faster-Whisper compute type (ignored by `onnx-asr`). |
| `CORO_ASR_QUANTIZATION` | `--asr-quantization` | _(unset)_ | onnx-asr quantization (e.g. `int8`); ignored by `faster-whisper`. Unset = fp32; int8 is a memory-fitting option, not a speed one. |
| `CORO_ASR_ONNX_VAD` | `--asr-onnx-vad` | `disabled` | Silero VAD segmentation for `onnx-asr` (`enabled` \| `disabled`). |
| `CORO_ASR_ONNX_VAD_THRESHOLD` | `--asr-onnx-vad-threshold` | _(unset)_ | Silero VAD speech-probability threshold; only when VAD enabled. |
| `CORO_ASR_MAX_CONCURRENCY` | `--asr-max-concurrency` | `0` _(auto)_ | Max ASR inference calls running at once; `0` auto-sizes from the host core count. Ignored by `onnx-genai`, which always serialises. |
| `CORO_ASR_MAX_QUEUE_DEPTH` | `--asr-max-queue-depth` | `32` | Max ASR calls allowed to queue for a slot; beyond this the request gets HTTP 429 + `Retry-After` instead of waiting indefinitely. |
| `CORO_BACKEND_DIARIZATION` | `--backend-diarization` | `none` | Diarization backend provider (`none` \| `nemo` \| `pyannote`). |
| `CORO_MODEL_DIARIZATION` | `--model-diarization` | _(unset)_ | Diarization model; defaults to `nvidia/diar_streaming_sortformer_4spk-v2` (`nemo`) or `pyannote/speaker-diarization-community-1` (`pyannote`). |
| `CORO_DIARIZATION_DEVICE` | `--diarization-device` | `auto` | Diarization device (`auto` \| `cuda` \| `cpu`). |
| `CORO_DIARIZATION_LATENCY` | `--diarization-latency` | `very-high` | Streaming Sortformer latency tier (`very-high` \| `high` \| `low` \| `ultra-low`); `nemo` streaming only. |
| `CORO_DIARIZATION_POSTPROCESSING` | `--diarization-postprocessing` | _(unset)_ | Sortformer post-processing preset (`dihard3-dev` \| `callhome-part1`), a path to a custom YAML, or `none` for NeMo's baseline; `nemo` only, see below. |
| `CORO_DIARIZATION_POSTPROCESSING_MAX_SPEAKERS` | `--diarization-postprocessing-max-speakers` | `4` | Speaker-count ceiling above which post-processing is bypassed; `nemo` only, see above. No effect on 4-speaker models. |
| `CORO_HF_TOKEN` | `--CORO-HF-TOKEN` | _(unset)_ | Hugging Face token for gated diarization models (e.g. pyannote community-1). Also read from `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` (and matching `--HF-TOKEN` flags) and `.env`; masked in logs. |
| `CORO_TRANSCRIPT_SPILL_DIR` | `--transcript-spill-dir` | _(first real-disk default)_ | Streaming transcript spill dir. Unset resolves to the system temp dir, or the cache dir when temp is tmpfs. A RAM-backed value is rejected at startup. |
| `CORO_WARMUP` | `--warmup` | `enabled` | Run warmup against the warmup audio asset at startup (`enabled` \| `disabled`). |
| `CORO_LOG_LEVEL` | `--log-level` | `info` | Log level (CLI use only). |
| `CORO_SSL_CERTFILE` | `--ssl-certfile` | _(unset)_ | TLS certificate file path. |
| `CORO_SSL_KEYFILE` | `--ssl-keyfile` | _(unset)_ | TLS private key file path. |

## Benchmarks

> **Picking a backend?** See the full **[leaderboard →
> docs/benchmark.md](docs/benchmark.md)** (WER, DER, RTFx, VRAM and RAM across
> backends, with reproduction commands). TL;DR: the **default** (onnx-asr
> `parakeet`, fp32) is the CPU pick and the Spanish pick; **faster-whisper
> `large-v3-turbo`** is the best English-meeting GPU option; **faster-whisper
> `small`** for max GPU throughput; **nemotron** for real-time streaming. Don't
> run Whisper through the onnx-asr backend (slower and less accurate than
> faster-whisper).

The table below is a separate, ASR-only view (diarization off).

Long-form English meetings from AMI (`Mix-Headset`, far-field, overlapping
speech), diarization off, on an RTX 3070 Laptop (8 GB) and a loaded laptop CPU.
**RTFx** = audio ÷ processing time (higher is faster). **Quality** =
normalized ORC-WER, lower is better. (Absolute WER is high because AMI
`Mix-Headset` is a hard far-field/overlap benchmark; treat the numbers as a
*relative* comparison.)

> **Why this table's RTFx differs from
> [docs/benchmark.md](docs/benchmark.md#reading-the-throughput-tables).** RTFx is
> a property of the measurement, not the model, and the two documents measure
> different things: this table is **long-form (10 min+) audio with diarization
> off**, the leaderboard's headline tables are **60 s clips with diarization
> on**. Short clips do not amortise per-request cost and diarization is included
> in the pipeline timing, which is most of the ~18× gap that used to sit
> unexplained between the `~120×` below and the leaderboard's `6.7×`.

| Backend / model | precision | RTFx (CPU) | RTFx (GPU) | ORC-WER (norm) |
|---|---|---:|---:|---:|
| **onnx-asr `parakeet-tdt-0.6b-v3`** (default) | fp32 | **5.0×** | **~120×** ‡ | 51–57% |
| faster-whisper `whisper-medium` | int8/fp16 | 0.6× | ~20× ‡ | 52–53% |
| onnx-genai `nemotron-…-int4` | int4 streaming | ~0.4× (impractical) | ~10× ‡ | 44–57% |

‡ **GPU figures are historical and were not reproduced** by the current
benchmark program — no artifacts for them exist in this repo. The CPU column and
the WER column were re-measured on 40 min of AMI clips plus 12 min of Spanish
FLEURS; `whisper-medium`'s CPU RTFx was previously listed as `1.3×` and measured
**0.58×**. See [Measured defaults
(CPU)](docs/benchmark.md#measured-defaults-cpu).

Memory footprint — **baseline** (peak, model + runtime, short clip):

| Backend / model | CPU RAM | GPU VRAM |
|---|---|---|
| onnx-asr `parakeet-tdt-0.6b-v3` (default) | ~2.7 GB (fp32) / ~1.3–1.7 GB (int8) | ~3.6 GB (fp32) / ~0.6 GB (int8) |
| faster-whisper `whisper-medium` | ~3.8 GB (default compute type) | ~2.3 GB (fp16) |
| onnx-genai `nemotron-…-int4` | ~1.0 GB | ~1.4 GB |

**Memory is not just the model on long audio.** The default **full-memory**
pipeline decodes and holds the entire PCM plus the accumulated
tokens/segments/words, so **host RAM grows ~linearly with recording length**.
The **streaming** pipeline (`CORO_PIPELINE=streaming`) streams 1 s PCM chunks
from disk instead of buffering the whole recording, and spills the growing
transcript to a per-request on-disk SQLite (WAL) store instead of Python lists.

When consumed over **SSE (`stream=true`)**, the streaming pipeline keeps
**flat peak host RSS, independent of recording length** (11 s ≈ 58 min ≈
multi-hour): finalized segments and raw words live on disk during the stream,
only bounded working buffers stay resident, and the final
`transcript.text.done` frame is rendered straight from the store one
segment/word at a time (never materialised). The wire format is unchanged.

| Consumption | host RSS vs. length |
|---|---|
| streaming + `stream=true` (SSE) | **flat** (bounded working set + on-disk store) |
| streaming, non-SSE `transcribe()` | flat steady-state, one O(length) peak when the single response dict is built |
| full-memory | grows ~linearly with length |

Notes:
- The on-disk store **must live on real disk** for the flat-RSS property, and
  the default now guarantees that: startup picks the system temp dir when it is
  real disk and a cache directory when it is tmpfs (RAM-backed, as `/tmp` is on
  most Linux distributions). An explicit RAM-backed `CORO_TRANSCRIPT_SPILL_DIR`
  fails startup instead of silently keeping the transcript in memory.
- The **non-streaming** `transcribe()` response inherently returns the whole
  transcript as one object, so its peak is O(length) at assembly time — use SSE
  consumption for unbounded audio.
- Diarizer prediction state grows ~0.7 MB/hour (frames × speakers × 4 bytes),
  negligible beside the model.
- **GPU VRAM is length-independent** in both pipelines (inference is
  windowed/streamed): parakeet ~3.6 GB, nemotron ~1.4 GB, faster-whisper
  ~2.3–2.9 GB.

Takeaways:
- **English meeting quality** is close across all three on this benchmark
  (parakeet 51.4% vs `whisper-medium` 51.6% normalized ORC-WER — a wash). On
  **Spanish** the gap is real: parakeet 5.8% vs `whisper-medium` 7.6% WER on
  FLEURS `es_419`, a 23% relative reduction.
- **Parakeet is the throughput winner on both devices** — ~8.8× `whisper-medium`
  on CPU, and its offline encoder batches frames on GPU.
- **Use fp32, not int8, on either device.** On GPU int8 inserts many CPU↔GPU
  copies; on CPU it is compute-bound, so int8 measured *no* speed gain and a
  3–6% relative WER cost. int8's only real payoff is memory.
- **Nemotron** is a *streaming* model: ~10× on GPU and impractical on CPU
  (~0.4×). Its value is low-latency real-time transcription, not batch speed.
- **Memory**: all backends fit comfortably on an 8 GB GPU; nemotron (int4) is
  the lightest, and parakeet int8 is the smallest CPU footprint.

### Benchmark datasets

Quality runs score against trustworthy, human-or-openly-labelled references
only. Each is materialized into a `--clips-dir` of `(<stem>.wav,
<stem>.ref.stm)` pairs; the bench scores WER and/or DER per the reference:

| Dataset | License | Metrics | Materialize with |
|---|---|---|---|
| **AMI** (English meetings) | CC-BY | WER + DER | `utils.make_ami_clip` / `--ami-preset` |
| **VoxConverse** (multi-speaker, in-the-wild) | CC-BY-4.0 | DER only (no transcript) | `utils.make_rttm_clip` |
| **VoxPopuli** (Spanish parliamentary speech) | CC0-1.0 | WER only (single speaker) | `--spanish-preset voxpopuli` |
| **FLEURS** (`es_419`, read speech) | CC-BY-4.0 | WER only (single speaker) | `--spanish-preset fleurs` |
| **Multilingual LibriSpeech** (Spanish) | CC-BY-4.0 | WER only (single speaker) | `--spanish-preset mls` |

Diarization-only references (e.g. VoxConverse) carry speaker turns but no
words; the report shows their DER and leaves WER blank rather than emitting a
meaningless score.

> **Spanish is WER-only.** Every public Spanish corpus above is single-speaker,
> so the Spanish workload set validates ASR quality and yields no meaningful
> DER. **Diarization quality is measured on AMI.**

> **Common Voice is no longer supported.** It moved off its previous free
> distribution channel in late 2025 and is no longer reproducibly fetchable, so
> the `make_common_voice_clips` utility was removed. Use `--spanish-preset` for
> Spanish WER.

> **Note — Albayzín-RTVE2020 (out of scope).** It is the strongest Spanish
> *diarization* target (human-revised transcripts **and** speaker labels), but it
> is gated behind an RTVE licence and cannot be redistributed, so it is not part
> of the reproducible workload set. Spanish diarization decisions stay
> AMI-driven. (Avoid the RTVE2018 subtitle-only partitions — those captions are
> not verbatim.)

#### Spanish workload set + published-WER calibration

```bash
# Print the one-time download footprint before committing to a fetch:
uv run --group bench coro-bench quality --spanish-preset calibration --spanish-fetch-plan

# Materialize + score (audio and Reference STM files land under --spanish-root):
uv run --group bench coro-bench quality \
  --spanish-preset calibration --server-url http://127.0.0.1:8123 --out-dir run
```

Corpora are fetched from the Hugging Face Parquet index one row group at a time,
so only the rows the preset asks for are downloaded, and the result is cached.
Each materialised directory ships a `LICENCES.md` and `corpora.json` recording
the licence, source and item provenance of every corpus.

The `fleurs` and `mls` presets are **calibration sets**: their scored normalized
ORC-WER is compared against the published figure for the configured ASR Model
Selection, and a deviation beyond `--calibration-margin` (default `0.10` WER
points, two-sided) **fails the run with exit code 3**. Matching an external
figure is the only end-to-end proof the harness is free of systematic error, so
a large deviation is a harness bug until proven otherwise. Results are written to
`<out-dir>/quality/calibration.json`.

### Running benchmarks

By default `coro-bench` **starts and stops the server it measures** (a
*bench-managed* server): it spawns `coro` on a free port with the `CORO_*` env
vars implied by the `--server-*` flags, waits for `/health` to report ready and
warmup-ready, runs the workload, and tears the server down afterwards. Install
the bench tooling first:

```bash
uv sync --group bench                       # meeteval, nvidia-ml-py, rich
```

To measure a server you started yourself (a *bench-attached* server), pass
`--server-url`; the `--server-*` flags are then rejected as mutually exclusive:

```bash
uv run --group bench coro --port 8123 &     # server under test (add --extra cuda for GPU)
uv run --group bench coro-bench all --server-url http://127.0.0.1:8123 ...
```

> Pass `--group bench` (and your hardware `--extra`) on **every** `uv run`
> below: a bare `uv run` re-syncs to the default environment and would
> uninstall the bench tooling again (the same re-sync gotcha as the `cuda`
> extra — see [GPU on a bare host](#gpu-on-a-bare-host)).

Three subcommands share the same flags:

| Subcommand | Measures |
|---|---|
| `quality` | transcription/diarization scores (cpWER, ORC-WER, DI-cpWER, DER, WDER) against a reference STM, via MeetEval |
| `performance` | resource + timing of the server process tree (PSS/USS, VRAM, CPU/GPU %, throughput) |
| `all` | both in a single run |

#### Smoke test on one small audio

A reference STM has one line per segment —
`<recording_id> <channel> <speaker> <start> <end> <text>` — where `recording_id`
is the audio filename stem. The package vendors an 11 s `jfk.wav`:

```bash
echo "jfk 1 JFK 0.000 11.000 and so my fellow americans ask not what your country can do for you ask what you can do for your country" > jfk.ref.stm

uv run --group bench coro-bench all \
  --audio coro/bench/data/jfk.wav \
  --reference-stm jfk.ref.stm \
  --out-dir ./bench-out
```

`quality` requires `--reference-stm` (and `all` needs it to score the quality
half); `performance` does not. The run prints a report and writes `REPORT.md`
plus `responses/ hyp/ ref/ quality/ performance/` under `--out-dir`.

#### Larger workloads

- `--clips-dir DIR` — a directory of `(<stem>.wav, <stem>.ref.stm)` pairs, e.g.
  produced by the dataset materializers (`utils.make_ami_clip`,
  `utils.make_rttm_clip`).
- `--ami-preset sample|eval|full` (or `--ami-groups` / `--ami-meetings`) — pull
  AMI meetings into `--ami-root` (default `./amicorpus/`); add `--no-download` to
  use only what is already present.
- `--spanish-preset voxpopuli|fleurs|mls|calibration|all` — materialize a Spanish
  workload set from public CC0/CC-BY corpora into `--spanish-root` (default
  `./spanish-corpora/`) and score it. Mutually exclusive with `--clips-dir`.

#### Useful flags

| Flag | Purpose |
|---|---|
| `--reps N` | repetitions per workload item (default 1) |
| `--stream` | drive the server over SSE; `performance`/`all` only (rejected for `quality`) |
| `--server-asr-backend` / `--server-asr-model` / `--server-diar-backend` / `--server-diar-model` / `--server-pipeline` / `--server-port` / `--no-diarization` | how the bench-managed server is launched |
| `--server-url URL` | attach to an already-running server instead (excludes all `--server-*` launch flags) |
| `--server-pid PID` / `--server-match STR` | bench-attached only: which process tree to sample (default match: `coro`). An ambiguous or empty match fails the run rather than sampling an unrelated process |
| `--reuse-reference-stms` | reuse `<ami-root>/stm/*.ref.stm` instead of regenerating them (they then reflect an older STM builder) |
| `--der-collar SECONDS` / `--der-regions all\|nooverlap\|single` | DER scoring options |

## Client integration

This server returns standard OpenAI shapes. A consuming project does **not** need
to redefine any schemas — install the `openai` SDK and reuse its types.

```bash
pip install "openai>=2.0.0"
```

### Option A — use the OpenAI client directly (returns typed objects)

```python
from openai import OpenAI

client = OpenAI(base_url="http://<host>:<port>/v1", api_key="not-needed")

with open("audio.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        file=f,
        model="whisper-1",  # accepted but ignored; server uses its configured backend
        response_format="diarized_json",  # -> openai.types.audio.TranscriptionDiarized
    )

print(result.text)
for segment in result.segments:
    print(segment.speaker, segment.start, segment.end, segment.text)
```

### Option B — import the response types for manual validation

```python
from openai.types.audio import (
    Transcription,  # response_format="json"
    TranscriptionVerbose,  # response_format="verbose_json"
    TranscriptionDiarized,  # response_format="diarized_json"
)

payload = httpx.post(url, files=..., data={"response_format": "verbose_json"}).json()
parsed = TranscriptionVerbose.model_validate(payload)
```

### Option C — call the HTTP endpoint directly with `curl`

No SDK required — `POST /v1/audio/transcriptions` accepts a standard multipart
form (`file`, `model`, `response_format`) and returns the OpenAI JSON shapes:

```bash
# Non-streaming (json | verbose_json | diarized_json)
curl http://<host>:<port>/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=whisper-1 \
  -F response_format=diarized_json

# Streaming token deltas over SSE
curl -N http://<host>:<port>/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=whisper-1 \
  -F response_format=json \
  -F stream=true
```

### Format ↔ SDK type mapping

| `response_format` | OpenAI SDK type |
|-------------------|-----------------|
| `json`            | `openai.types.audio.Transcription` |
| `verbose_json`    | `openai.types.audio.TranscriptionVerbose` (segments: `TranscriptionSegment`, words: `TranscriptionWord`) |
| `diarized_json`   | `openai.types.audio.TranscriptionDiarized` (segments: `TranscriptionDiarizedSegment`) |
| SSE stream events | `TranscriptionTextDeltaEvent` / `TranscriptionTextDoneEvent` |

Conformance is enforced by `tests/test_openai_sdk_conformance.py`, which validates
every server response against the SDK types.

> Note: standard OpenAI types carry **segment-level** speaker labels only —
> there is no OpenAI-compatible slot for a per-word speaker. Use the
> Deepgram-native endpoint below to get one.

## Per-word speakers: `POST /v1/listen` (Deepgram-compatible)

Coro assigns a speaker to every word and keeps each word's real ASR timing and
confidence. No OpenAI type can carry that, so rather than bending OpenAI's
format, Coro implements **Deepgram's own endpoint contract**: raw audio body
(not multipart), Deepgram's query parameters and defaults, and Deepgram's error
shape.

```bash
curl -s "http://localhost:8000/v1/listen?diarize=true&utterances=true" \
  -H "Authorization: Token any-value" \
  -H "Content-Type: audio/wav" \
  --data-binary @audio.wav
```

```json
{
  "results": {
    "channels": [ { "alternatives": [ { "transcript": "hola mundo si",
      "words": [ { "word": "hola", "start": 0.0, "end": 0.5,
                   "confidence": 0.91, "speaker": 1 } ] } ] } ],
    "utterances": [ { "speaker": 1, "transcript": "hola mundo",
                      "start": 0.0, "end": 1.0, "confidence": 0.87,
                      "words": [ ... ] } ]
  }
}
```

Responses parse with the official SDK, enforced by
`tests/test_deepgram_sdk_conformance.py`:

```python
from deepgram.types.listen_v1response import ListenV1Response

ListenV1Response.model_validate(response.json())
```

Behaviour worth knowing:

- **`diarize` and `utterances` default to `false`**, exactly as at Deepgram, so
  per-word speakers need `?diarize=true`. Coro does not override vendor
  defaults to be more helpful.
- Timestamps are float **seconds**; speaker numbering is Coro's (1-based),
  passed through rather than renumbered, so labels stay comparable with
  `diarized_json`.
- A word the diarizer does not cover has **no `speaker` key** — Deepgram never
  emits a null speaker, and inventing a label would be a guess.
- `speaker_confidence` is **omitted**: Coro's diarizers binarize their
  per-frame posteriors, so the value does not exist to report.
- `Authorization` is accepted and never validated. Coro has no auth.
- `?utterances=true` roughly doubles the body, because the shape carries every
  word twice (flat and nested per utterance) — as the real API does.

### What is and isn't supported

This is a **documented subset**, not a full clone. Of Deepgram's 37
pre-recorded parameters — 3 honoured, 16 refused, 18 ignored:

| | parameters |
|---|---|
| **honoured** (3) | `diarize`, `utterances`, `language` |
| **refused** with a `400` (3) | `redact`, `callback`, `callback_method` |
| **accepted and ignored** (31) | everything else — `summarize`, `sentiment`, `topics`, `intents`, `detect_entities`, `paragraphs`, `search`, `multichannel`, `punctuate`, `smart_format`, `model`, … |

Unhonoured parameters are **ignored**, each documented in the OpenAPI schema
with what its absence means, so a standard parameter bundle still works and a
future Deepgram flag will not break the endpoint. Features Coro does not
compute simply produce no response key.

Two are **refused**, because ignoring them fails silently instead of visibly:
`redact` would return unredacted text under a redaction request (a compliance
failure wearing a 200), and `callback` would leave a client waiting forever for
a webhook that never fires. A missing `summary` key you can see; those two you
cannot.

Also not implemented:

- **URL ingest.** `{"url": "..."}` bodies are refused with a clear message;
  submit audio as the raw request body.
- **`listen/v2`** — WebSocket-only, and its distinguishing feature is
  contextual turn detection, which Coro has no equivalent of.
- **`interim_results`, `vad_events`, `utterance_end_ms`** — Coro emits only
  tokens it has already accepted, so every frame is final.

## Live streaming: `WebSocket /v1/listen`

Deepgram's streaming contract is a WebSocket, so Coro implements one. This is
genuine live transcription — `Results` frames are pushed as windows complete,
not after the client stops sending.

```python
import json, websockets

async with websockets.connect(
    "ws://localhost:8000/v1/listen?encoding=linear16&sample_rate=16000&diarize=true"
) as ws:
    await ws.send(pcm_chunk)  # binary frames: raw samples
    await ws.send(json.dumps({"type": "KeepAlive"}))
    print(json.loads(await ws.recv()))  # {"type": "Results", ...}
    await ws.send(json.dumps({"type": "CloseStream"}))
```

- **Audio is declared, not sniffed.** A socket has no container, so
  `encoding=linear16` is required; other encodings are refused at connect time
  with an `Error` frame rather than after a minute of noise. Non-16 kHz rates
  are resampled.
- **Control frames:** `KeepAlive`, `Finalize`, `CloseStream`.
- **Interim frames carry no speaker.** The diarization timeline is incomplete
  while audio is arriving, so a mid-stream label would be a guess later frames
  contradict. With `diarize=true` a final attributed frame is sent once the
  timeline is complete — a deliberate deviation from Deepgram, which labels
  interim words.
- The stream always ends with a `Metadata` frame.

See `docs/adr/0015-vendor-native-endpoints.md` for the fidelity policy.
`/v1/audio/transcriptions` is byte-unchanged, asserted in
`tests/test_openai_formats_unchanged.py`.

> AssemblyAI is not yet available. Its contract is asynchronous
> (`POST /v2/upload` → `POST /v2/transcript` → poll `GET /v2/transcript/{id}`),
> which needs job state Coro does not have; a synchronous approximation would
> be a partial clone. Tracked separately.

## Development

To hack on Coro, clone the repo and install into a project environment with
`uv sync`. Pick the runtime that matches your hardware — the `cpu` / `cuda`
extras are mutually exclusive and carry the matching `onnxruntime` /
`onnxruntime-genai` wheels:

```bash
git clone https://github.com/collectiveai-team/coro && cd coro
uv sync --extra cpu     # CPU-only
uv sync --extra cuda    # NVIDIA GPU
```

Add `--extra diar-pyannote` (combinable with `cpu` or `cuda`) for the gated
pyannote diarization backend — see [Diarization backends](#diarization-backends):

```bash
uv sync --extra cpu --extra diar-pyannote
```

Run the server and the checks from the project environment with `uv run`:

```bash
uv run coro             # or: uv run uvicorn coro.app:app
uv run pytest           # tests
uv run ruff check .     # lint
```

### GPU on a bare host

Running the GPU build outside the devcontainer has two gotchas:

1. **`uv run` re-syncs to the *default* environment and uninstalls the `cuda`
   extra.** Run the server with the extra explicitly so the GPU wheels stay
   installed: `uv run --extra cuda coro` (or re-run `uv sync --extra cuda`
    after any plain `uv sync` / `uv run`). (`uv tool install "coro-asr[cuda]"` is
   not affected — its environment is not re-synced.)
2. **faster-whisper (CTranslate2) needs `libcublas.so.12` + cuDNN 9**, which
   ship in the `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` wheels (pulled by the
   `cuda` extra) but are **not** on the loader path by default. If you see
   `RuntimeError: Library libcublas.so.12 is not found`, prepend the wheel lib
   dirs to `LD_LIBRARY_PATH`:
   ```bash
   export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cublas/lib:\
   $VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
   ```
   The shipped Docker GPU image bakes the `nvidia-cublas-cu12` wheel dir onto
   `LD_LIBRARY_PATH` for you (see the `Dockerfile` runtime stage). In the
   devcontainer or a bare `uv` env — both now on the `nvidia/cuda:13.x` base,
   which ships `libcublas.so.13`, not `.so.12` — you still need the export
   above. The `onnx-asr` / `onnx-genai` backends use onnxruntime-gpu (CUDA 13,
   matching the base) and are unaffected by gotcha 2.

## License

[MIT](LICENSE) © collective.ai, jedzill4
