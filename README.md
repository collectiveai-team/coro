# aymurai-asr

OpenAI-compatible ASR + diarization HTTP server (`asr_diar_server`), backed by
Faster-Whisper (ASR) and NVIDIA NeMo Sortformer (diarization).

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
