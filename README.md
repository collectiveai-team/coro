<p align="center">
  <img src="assets/coro-logo.png" alt="Coro — OpenAI-compatible ASR + speaker diarization" style="width:600px; max-width:100%; height:auto;" />
</p>

<p align="center">
  <em>Self-hosted, OpenAI-compatible speech-to-text that knows who said what.</em>
</p>

<p align="center">
  <a href="https://github.com/collectiveai-team/coro/releases"><img alt="Release" src="https://img.shields.io/github/v/release/collectiveai-team/coro?logo=github" /></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" alt="Python 3.12"></a>
  <a href="https://platform.openai.com/docs/api-reference/audio"><img src="https://img.shields.io/badge/API-OpenAI--compatible-412991?logo=openai&logoColor=white" alt="OpenAI-compatible API"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

---

**Source Code**: [https://github.com/collectiveai-team/coro](https://github.com/collectiveai-team/coro)

---

Coro is an embedded ASR + speaker-diarization server that speaks the OpenAI
transcription contract — point the official `openai` SDK at it and get back
typed transcripts that know *who* said *what*, no custom schema package needed.

The name nods to *coro* (Spanish for "chorus") — many voices, transcribed and
attributed to who spoke them.

The key features are:
- **OpenAI-compatible API** — drop-in `/v1/audio/transcriptions`; clients reuse the official `openai` SDK types (`Transcription` / `TranscriptionVerbose` / `TranscriptionDiarized`) with no custom schema
- **Audio *and* video input** — uploads are decoded through ffmpeg, so any container it supports works: audio (`.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, …) and video (`.mp4`, `.mkv`, `.mov`, `.webm`, …); the audio track is extracted to 16 kHz mono PCM automatically — same endpoint, same response shapes
- **Pluggable diarization backends** — pick per deployment: NVIDIA NeMo Sortformer (streaming-capable, **≤ 4 speakers**) or pyannote community-1 (batch/whole-file, **handles > 4 speakers**); both attribute every segment to a speaker (`diarized_json`), so you get *who spoke, when, and what*
- **Pluggable ASR backends** — pick per deployment: Faster-Whisper (best accuracy, multilingual), onnx-asr Parakeet (highest GPU throughput), or onnx-genai Nemotron (real-time streaming)
- **Two transcription pipelines** — `full-memory` (default) decodes and holds the whole recording in RAM for lowest latency on short/medium clips; `streaming` streams 1 s PCM chunks off disk and spills the growing transcript to a per-request on-disk store, trading a little latency for **flat host RAM on arbitrarily long audio**. Select with `CORO_PIPELINE` / `--pipeline` — see [the pipeline comparison](#two-transcription-pipelines-full-memory-vs-streaming)
- **Streaming over SSE** — OpenAI-exact `transcript.text.delta` / `transcript.text.done` / `[DONE]` events with `stream=true`
- **Flat-memory long audio** — the streaming pipeline spills the transcript to disk so host RSS stays flat from 11 s to multi-hour recordings
- **CPU & GPU** — mutually-exclusive `cpu` / `cuda` extras carry the matching `onnxruntime` wheels; multilingual on either
- **Run it your way** — ephemeral `uvx`, a standalone `uv tool install` command, or a full `uv sync` dev checkout

## Quickstart

Run the server without installing it into a project, straight from the repo,
using `uvx` (the alias for `uv tool run`). Pick the hardware extra that matches
your machine:

```bash
# CPU-only
uvx --from "coro-asr[cpu]  @ git+https://github.com/collectiveai-team/coro" coro --port 8000

# NVIDIA GPU
uvx --from "coro-asr[cuda] @ git+https://github.com/collectiveai-team/coro" coro --port 8000
```

`uvx` builds a throwaway isolated environment and launches the `coro` command —
no `uv sync`/`uv run` and nothing added to your current project. The server now
speaks the OpenAI transcription contract at `http://127.0.0.1:8000/v1`.

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
        model="whisper-1",                # accepted but ignored; server uses its backend
        response_format="diarized_json",  # json | verbose_json | diarized_json
    )

print(result.text)
for segment in result.segments:          # who spoke, when, and what
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
uv tool install "coro-asr[cpu]"  @ git+https://github.com/collectiveai-team/coro   # CPU-only
uv tool install "coro-asr[cuda]" @ git+https://github.com/collectiveai-team/coro   # NVIDIA GPU
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
  --build-arg CORE_IMAGE=nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04 \
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

`response_format` accepts `json`, `verbose_json`, and `diarized_json`. With
`stream=true` the endpoint emits OpenAI-exact SSE
(`transcript.text.delta` / `transcript.text.done` / `[DONE]`).

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

For flat RAM on long audio, point `CORO_TRANSCRIPT_SPILL_DIR` at a persistent
(non-tmpfs) directory — the default temp dir is RAM-backed on many systems,
which would defeat the spill. See [Benchmarks](#benchmarks) for the measured
memory behaviour.

## ASR backends

The ASR backend is pluggable behind a single adapter contract. Select it with
`CORO_BACKEND_ASR` + `CORO_MODEL_ASR`; pick the device with
`CORO_ASR_DEVICE` (`auto` | `cpu` | `cuda`).

| Backend (`CORO_BACKEND_ASR`) | Runtime | Typical model (`CORO_MODEL_ASR`) | Notes |
|---|---|---|---|
| `faster-whisper` | CTranslate2 | `openai/whisper-medium` | Default. Best accuracy; multilingual. `CORO_ASR_COMPUTE_TYPE` = `int8` (CPU) / `float16` (GPU). |
| `onnx-asr` | onnxruntime | `nemo-parakeet-tdt-0.6b-v3` | NeMo Parakeet/Canary; multilingual. Offline (batched) → very high GPU throughput. `CORO_ASR_QUANTIZATION` = `int8` (CPU) or unset = fp32 (GPU). |
| `onnx-genai` | onnxruntime-genai | `onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4` | NVIDIA Nemotron **cache-aware streaming**; 40 locales. Built for low-latency real-time, not batch throughput. Timestamps are 560 ms-resolution. GPU strongly recommended. |

### Recommended configuration

Each setting below is shown as an env var; the equivalent CLI flag is the
`--kebab-case` form (e.g. `--backend-asr onnx-asr`).

**GPU (`--extra cuda`):**
```bash
CORO_BACKEND_ASR=onnx-asr
CORO_MODEL_ASR=nemo-parakeet-tdt-0.6b-v3
CORO_ASR_DEVICE=cuda           # fp32 (leave CORO_ASR_QUANTIZATION unset)
```
Or as CLI flags:
```bash
coro --backend-asr onnx-asr --model-asr nemo-parakeet-tdt-0.6b-v3 \
  --asr-device cuda --port 8000
```
Fastest by a wide margin with near-best accuracy. Use `faster-whisper` +
`float16` if you want the top accuracy point; use `onnx-genai` only for
real-time low-latency streaming.

**CPU (`--extra cpu`):**
```bash
CORO_BACKEND_ASR=onnx-asr
CORO_MODEL_ASR=nemo-parakeet-tdt-0.6b-v3
CORO_ASR_DEVICE=cpu
CORO_ASR_QUANTIZATION=int8     # ~4× faster than whisper-medium
```
For maximum accuracy on CPU (at ~1.3× realtime) use `faster-whisper` with
`CORO_ASR_COMPUTE_TYPE=int8`. `onnx-genai` is not recommended on CPU.

**Streaming on long audio:** set `CORO_PIPELINE=streaming` and point
`CORO_TRANSCRIPT_SPILL_DIR` at a persistent (non-tmpfs) directory so the
per-request transcript spills to disk and host RSS stays flat regardless of
recording length. Consume the result over SSE (`stream=true`).

## Diarization backends

Diarization is **optional** (default `none` — an ASR-only server is valid) and
pluggable behind a single `DiarizationAdapter` contract, dispatched by a
per-capability Backend Adapter Factory (see ADR 0007). Select it with
`CORO_BACKEND_DIARIZATION` + `CORO_MODEL_DIARIZATION`; pick the device with
`CORO_DIARIZATION_DEVICE` (`auto` | `cpu` | `cuda`).

| Backend (`CORO_BACKEND_DIARIZATION`) | Default model | Speakers | Streaming | Gated / token | Install |
|---|---|---|---|---|---|
| `nemo` | `nvidia/diar_streaming_sortformer_4spk-v2` | **≤ 4** (4-speaker Sortformer) | ✅ works with `CORO_PIPELINE=streaming` | no | core install |
| `pyannote` | `pyannote/speaker-diarization-community-1` | **unbounded** — handles **> 4** | ❌ batch/whole-file only | **yes — Hugging Face token required** | `--extra diar-pyannote` |

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
labels back. Sortformer handles **≤ 4 speakers**; for more, use pyannote below.

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
| `CORO_BACKEND_ASR` | `--backend-asr` | `faster-whisper` | ASR backend provider (`faster-whisper` \| `onnx-asr` \| `onnx-genai`). |
| `CORO_MODEL_ASR` | `--model-asr` | `openai/whisper-medium` | ASR model selection. |
| `CORO_ASR_DEVICE` | `--asr-device` | `auto` | ASR device (`auto` \| `cuda` \| `cpu`). |
| `CORO_ASR_COMPUTE_TYPE` | `--asr-compute-type` | `default` | Faster-Whisper compute type (ignored by `onnx-asr`). |
| `CORO_ASR_QUANTIZATION` | `--asr-quantization` | _(unset)_ | onnx-asr quantization (e.g. `int8`); ignored by `faster-whisper`. |
| `CORO_ASR_ONNX_VAD` | `--asr-onnx-vad` | `disabled` | Silero VAD segmentation for `onnx-asr` (`enabled` \| `disabled`). |
| `CORO_ASR_ONNX_VAD_THRESHOLD` | `--asr-onnx-vad-threshold` | _(unset)_ | Silero VAD speech-probability threshold; only when VAD enabled. |
| `CORO_BACKEND_DIARIZATION` | `--backend-diarization` | `none` | Diarization backend provider (`none` \| `nemo` \| `pyannote`). |
| `CORO_MODEL_DIARIZATION` | `--model-diarization` | _(unset)_ | Diarization model; defaults to `nvidia/diar_streaming_sortformer_4spk-v2` (`nemo`) or `pyannote/speaker-diarization-community-1` (`pyannote`). |
| `CORO_DIARIZATION_DEVICE` | `--diarization-device` | `auto` | Diarization device (`auto` \| `cuda` \| `cpu`). |
| `CORO_DIARIZATION_LATENCY` | `--diarization-latency` | `very-high` | Streaming Sortformer latency tier (`very-high` \| `high` \| `low` \| `ultra-low`); `nemo` streaming only. |
| `CORO_HF_TOKEN` | `--CORO-HF-TOKEN` | _(unset)_ | Hugging Face token for gated diarization models (e.g. pyannote community-1). Also read from `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` (and matching `--HF-TOKEN` flags) and `.env`; masked in logs. |
| `CORO_TRANSCRIPT_SPILL_DIR` | `--transcript-spill-dir` | _(system temp)_ | Streaming transcript spill dir; must be real disk (non-tmpfs) for flat RAM. |
| `CORO_WARMUP` | `--warmup` | `enabled` | Run warmup against the warmup audio asset at startup (`enabled` \| `disabled`). |
| `CORO_LOG_LEVEL` | `--log-level` | `info` | Log level (CLI use only). |
| `CORO_SSL_CERTFILE` | `--ssl-certfile` | _(unset)_ | TLS certificate file path. |
| `CORO_SSL_KEYFILE` | `--ssl-keyfile` | _(unset)_ | TLS private key file path. |

## Benchmarks

> **Picking a backend?** See the full **[leaderboard →
> docs/benchmark.md](docs/benchmark.md)** (WER, DER, RTFx, VRAM and RAM across
> backends, with reproduction commands). TL;DR: **faster-whisper
> `large-v3-turbo`** is the best GPU default — best WER *and* DER, multilingual,
> ~3 GB VRAM; **faster-whisper `small`** for max GPU throughput; **onnx-asr
> `parakeet`** for CPU; **nemotron** for real-time streaming. Don't run Whisper
> through the onnx-asr backend (slower and less accurate than faster-whisper).

The table below is a separate, ASR-only view (diarization off).

Long-form English meetings from AMI (`Mix-Headset`, far-field, overlapping
speech), diarization off, on an RTX 3070 Laptop (8 GB) and a loaded laptop CPU.
**RTFx** = audio ÷ processing time (higher is faster). **Quality** =
normalized ORC-WER, lower is better. (Absolute WER is high because AMI
`Mix-Headset` is a hard far-field/overlap benchmark; treat the numbers as a
*relative* comparison.)

| Backend / model | precision | RTFx (CPU) | RTFx (GPU) | ORC-WER (norm) |
|---|---|---:|---:|---:|
| faster-whisper `whisper-medium` | int8/fp16 | 1.3× | ~20× | **42–53%** |
| onnx-asr `parakeet-tdt-0.6b-v3` | int8 (CPU) / fp32 (GPU) | 5.0× | **~120×** | 44–57% |
| onnx-genai `nemotron-…-int4` | int4 streaming | ~0.4× (impractical) | ~10× | 44–57% |

Memory footprint — **baseline** (peak, model + runtime, short clip):

| Backend / model | CPU RAM | GPU VRAM |
|---|---|---|
| faster-whisper `whisper-medium` | ~2.0 GB (int8) | ~2.3 GB (fp16) |
| onnx-asr `parakeet-tdt-0.6b-v3` | ~1.2 GB (int8) / ~2.7 GB (fp32) | ~3.6 GB (fp32) / ~0.6 GB (int8) |
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
- The on-disk store **must live on real disk** for the flat-RSS property. Set
  `CORO_TRANSCRIPT_SPILL_DIR` to a persistent path; the default temp dir is
  tmpfs (RAM-backed) on many systems, which would keep the transcript in memory.
- The **non-streaming** `transcribe()` response inherently returns the whole
  transcript as one object, so its peak is O(length) at assembly time — use SSE
  consumption for unbounded audio.
- Diarizer prediction state grows ~0.7 MB/hour (frames × speakers × 4 bytes),
  negligible beside the model.
- **GPU VRAM is length-independent** in both pipelines (inference is
  windowed/streamed): parakeet ~3.6 GB, nemotron ~1.4 GB, faster-whisper
  ~2.3–2.9 GB.

Takeaways:
- **Quality** is close across all three on this benchmark; faster-whisper
  `medium` is marginally the most accurate.
- **Parakeet on GPU is the throughput winner** (~120× — its offline encoder
  batches frames). On GPU use **fp32**: int8 is *slower* there (onnxruntime
  inserts many CPU↔GPU copies), lowers accuracy, and only saves VRAM
  (~0.6 GB vs ~3.6 GB) — rarely worth it.
- **Nemotron** is a *streaming* model: ~10× on GPU and impractical on CPU
  (~0.4×). Its value is low-latency real-time transcription, not batch speed.
- **Memory**: all backends fit comfortably on an 8 GB GPU; nemotron (int4) is
  the lightest, and parakeet int8 is the smallest CPU footprint (~1.2 GB).

### Benchmark datasets

Quality runs score against trustworthy, human-or-openly-labelled references
only. Each is materialized into a `--clips-dir` of `(<stem>.wav,
<stem>.ref.stm)` pairs; the bench scores WER and/or DER per the reference:

| Dataset | License | Metrics | Materialize with |
|---|---|---|---|
| **AMI** (English meetings) | CC-BY | WER + DER | `utils.make_ami_clip` |
| **VoxConverse** (multi-speaker, in-the-wild) | CC-BY-4.0 | DER only (no transcript) | `utils.make_rttm_clip` |
| **Common Voice** (single-speaker read speech, any language incl. `es`) | CC0 | WER only (single speaker) | `utils.make_common_voice_clips` |

Diarization-only references (e.g. VoxConverse) carry speaker turns but no
words; the report shows their DER and leaves WER blank rather than emitting a
meaningless score.

> **TODO — apply for Albayzín-RTVE2020.** It is the strongest Spanish target
> (real peninsular broadcast, *fully human-revised* transcripts **and** speaker
> labels → trustworthy WER **and** DER), but it is gated: an accredited
> researcher/company must request access via the RTVE archive
> (<http://catedrartve.unizar.es/rtvedatabase.html>) and it cannot be
> redistributed/vendored. Once obtained locally, its RTTM diarization refs feed
> straight into `utils.make_rttm_clip`. (Avoid the RTVE2018 subtitle-only
> partitions — those captions are not verbatim.)

### Running benchmarks

`coro-bench` scores a **running** server — it attaches over HTTP and does *not*
start one for you. Install the bench tooling (MeetEval + samplers) and start the
server you want to measure first:

```bash
uv sync --group bench                       # meeteval, nvidia-ml-py, rich
uv run --group bench coro --port 8123 &     # server under test (add --extra cuda for GPU)
```

> Pass `--group bench` (and your hardware `--extra`) on **every** `uv run`
> below: a bare `uv run` re-syncs to the default environment and would
> uninstall the bench tooling again (the same re-sync gotcha as the `cuda`
> extra — see [GPU on a bare host](#gpu-on-a-bare-host)).

Three subcommands share the same flags:

| Subcommand | Measures |
|---|---|
| `quality` | transcription/diarization scores (cpWER, ORC-WER, DI-cpWER, DER) against a reference STM, via MeetEval |
| `performance` | resource + timing of the server process tree (PSS/USS, VRAM, CPU/GPU %, throughput) |
| `all` | both in a single run |

#### Smoke test on one small audio

A reference STM has one line per segment —
`<recording_id> <channel> <speaker> <start> <end> <text>` — where `recording_id`
is the audio filename stem. The package vendors an 11 s `jfk.wav`:

```bash
echo "jfk 1 JFK 0.000 11.000 and so my fellow americans ask not what your country can do for you ask what you can do for your country" > jfk.ref.stm

uv run --group bench coro-bench all \
  --server-url http://127.0.0.1:8123 \
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
  `utils.make_common_voice_clips`, `utils.make_rttm_clip`).
- `--ami-preset sample|eval|full` (or `--ami-groups` / `--ami-meetings`) — pull
  AMI meetings into `--ami-root` (default `./amicorpus/`); add `--no-download` to
  use only what is already present.

#### Useful flags

| Flag | Purpose |
|---|---|
| `--reps N` | repetitions per workload item (default 1) |
| `--stream` | drive the server over SSE; `performance`/`all` only (rejected for `quality`) |
| `--server-pid PID` / `--server-match STR` | which process tree to sample for `performance` (default match: `coro`) |
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
        model="whisper-1",              # accepted but ignored; server uses its configured backend
        response_format="diarized_json",  # -> openai.types.audio.TranscriptionDiarized
    )

print(result.text)
for segment in result.segments:
    print(segment.speaker, segment.start, segment.end, segment.text)
```

### Option B — import the response types for manual validation

```python
from openai.types.audio import (
    Transcription,          # response_format="json"
    TranscriptionVerbose,   # response_format="verbose_json"
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

> Note: standard OpenAI types carry **segment-level** speaker labels only.
> Word-level speaker/confidence is an internal detail and is not exposed at the
> HTTP boundary.

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
   The devcontainer avoids both: its `nvidia/cuda:12.x` base image provides
   `libcublas.so.12` system-wide via `LD_LIBRARY_PATH=/usr/local/cuda/lib64`.
   The `onnx-asr` / `onnx-genai` backends use onnxruntime-gpu and are
   unaffected by gotcha 2.

## License

[MIT](LICENSE) © collective.ai, jedzill4
