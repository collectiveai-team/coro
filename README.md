# aymurai-asr

OpenAI-compatible ASR + diarization HTTP server (`asr_diar_server`), with three
pluggable ASR backends (Faster-Whisper, onnx-asr Parakeet, onnx-genai Nemotron)
and NVIDIA NeMo Sortformer diarization.

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
uv run asr-diar-server            # or: uv run uvicorn asr_diar_server.app:app
```

Configuration is via `ASR_DIAR_`-prefixed environment variables (host, port,
backends, devices, etc.); see `asr_diar_server/settings.py`.

Install the runtime that matches your hardware (the `cpu` / `cuda` extras are
mutually exclusive and carry the matching `onnxruntime` / `onnxruntime-genai`
wheels):

```bash
uv sync --extra cpu     # CPU-only deployment
uv sync --extra cuda    # NVIDIA GPU deployment
```

## ASR backends

The ASR backend is pluggable behind a single adapter contract. Select it with
`ASR_DIAR_BACKEND_ASR` + `ASR_DIAR_MODEL_ASR`; pick the device with
`ASR_DIAR_ASR_DEVICE` (`auto` | `cpu` | `cuda`).

| Backend (`ASR_DIAR_BACKEND_ASR`) | Runtime | Typical model (`ASR_DIAR_MODEL_ASR`) | Notes |
|---|---|---|---|
| `faster-whisper` | CTranslate2 | `openai/whisper-medium` | Default. Best accuracy; multilingual. `ASR_DIAR_ASR_COMPUTE_TYPE` = `int8` (CPU) / `float16` (GPU). |
| `onnx-asr` | onnxruntime | `nemo-parakeet-tdt-0.6b-v3` | NeMo Parakeet/Canary; multilingual. Offline (batched) → very high GPU throughput. `ASR_DIAR_ASR_QUANTIZATION` = `int8` (CPU) or unset = fp32 (GPU). |
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
tokens/segments/words, so **host RAM grows ~linearly with recording length**
(peak RSS, 11 s → 58 min):

| Backend | 11 s | 58 min | Δ |
|---|---:|---:|---:|
| onnx-asr parakeet (fp32, GPU) | 1.06 GB | 1.52 GB | +0.46 GB |
| faster-whisper (fp16, GPU) | 1.67 GB | 2.43 GB | +0.76 GB |
| onnx-genai nemotron (int4, GPU) | 0.97 GB | 1.49 GB | +0.52 GB |

**GPU VRAM stays roughly flat** with length (inference is windowed/streamed, so
the working set is bounded): parakeet ~3.6 GB and nemotron ~1.4 GB are
length-independent; faster-whisper grows mildly (~2.3 → ~2.9 GB). For long or
continuous recordings, use the **streaming pipeline** (`ASR_DIAR_PIPELINE=streaming`)
to bound host memory instead of buffering the whole recording.

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
ASR_DIAR_BACKEND_ASR=onnx-asr
ASR_DIAR_MODEL_ASR=nemo-parakeet-tdt-0.6b-v3
ASR_DIAR_ASR_DEVICE=cuda           # fp32 (leave ASR_DIAR_ASR_QUANTIZATION unset)
```
Fastest by a wide margin with near-best accuracy. Use `faster-whisper` +
`float16` if you want the top accuracy point; use `onnx-genai` only for
real-time low-latency streaming.

**CPU (`--extra cpu`):**
```bash
ASR_DIAR_BACKEND_ASR=onnx-asr
ASR_DIAR_MODEL_ASR=nemo-parakeet-tdt-0.6b-v3
ASR_DIAR_ASR_DEVICE=cpu
ASR_DIAR_ASR_QUANTIZATION=int8     # ~4× faster than whisper-medium
```
For maximum accuracy on CPU (at ~1.3× realtime) use `faster-whisper` with
`ASR_DIAR_ASR_COMPUTE_TYPE=int8`. `onnx-genai` is not recommended on CPU.

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
