# Coro

Coro is an OpenAI-compatible ASR + diarization HTTP server (package
`coro`), with three pluggable ASR backends (Faster-Whisper, onnx-asr
Parakeet, onnx-genai Nemotron) and NVIDIA NeMo Sortformer diarization.

The name nods to *coro* (Spanish for "chorus") — many voices, transcribed and
attributed to who spoke them.

It exposes the OpenAI transcription contract, so **clients integrate using the
official `openai` SDK types — no custom schema package is needed.**

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Readiness / capability status. |
| `POST` | `/v1/audio/transcriptions` | OpenAI-compatible transcription (multipart). |

`response_format` accepts `json`, `verbose_json`, and `diarized_json`. With
`stream=true` the endpoint emits OpenAI-exact SSE
(`transcript.text.delta` / `transcript.text.done` / `[DONE]`).

## Running the server

```bash
uv sync
uv run coro                       # or: uv run uvicorn coro.app:app
```

Configuration is via `CORO_`-prefixed environment variables (host, port,
backends, devices, etc.); see `coro/settings.py`.

Install the runtime that matches your hardware (the `cpu` / `cuda` extras are
mutually exclusive and carry the matching `onnxruntime` / `onnxruntime-genai`
wheels):

```bash
uv sync --extra cpu     # CPU-only deployment
uv sync --extra cuda    # NVIDIA GPU deployment
```

## ASR backends

The ASR backend is pluggable behind a single adapter contract. Select it with
`CORO_BACKEND_ASR` + `CORO_MODEL_ASR`; pick the device with
`CORO_ASR_DEVICE` (`auto` | `cpu` | `cuda`).

| Backend (`CORO_BACKEND_ASR`) | Runtime | Typical model (`CORO_MODEL_ASR`) | Notes |
|---|---|---|---|
| `faster-whisper` | CTranslate2 | `openai/whisper-medium` | Default. Best accuracy; multilingual. `CORO_ASR_COMPUTE_TYPE` = `int8` (CPU) / `float16` (GPU). |
| `onnx-asr` | onnxruntime | `nemo-parakeet-tdt-0.6b-v3` | NeMo Parakeet/Canary; multilingual. Offline (batched) → very high GPU throughput. `CORO_ASR_QUANTIZATION` = `int8` (CPU) or unset = fp32 (GPU). |
| `onnx-genai` | onnxruntime-genai | `onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4` | NVIDIA Nemotron **cache-aware streaming**; 40 locales. Built for low-latency real-time, not batch throughput. Timestamps are 560 ms-resolution. GPU strongly recommended. |

### Benchmarks

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

### Recommended configuration

**GPU (`--extra cuda`):**
```bash
CORO_BACKEND_ASR=onnx-asr
CORO_MODEL_ASR=nemo-parakeet-tdt-0.6b-v3
CORO_ASR_DEVICE=cuda           # fp32 (leave CORO_ASR_QUANTIZATION unset)
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

## Client integration (the important part)

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

```bash
uv run pytest          # tests
uv run ruff check .    # lint
```
