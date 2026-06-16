# Package and API Surface

The packaged server uses a root-level `coro` package, exposes a lightweight module-level `coro.app:app`, and exposes one public transcription route: `/v1/audio/transcriptions`. Full-memory and chunked-file behavior are not public API versions; they are startup-selected transcription pipeline implementations chosen by `CORO_PIPELINE`.

The supported endpoint set is `/health` and `/v1/audio/transcriptions`. `/v2/audio/transcriptions` is not a supported endpoint or compatibility alias. Prototype-only UI, WebSocket, Deepgram-compatible, model-list, behavior-specific transcription routes, and the old `custom_server.py` / `custom_server_chunked.py` script names do not shape the packaged API.

Pipeline implementations are named by behavior: `FullMemoryPipeline` and `ChunkedFilePipeline`. The configured pipeline is provided through a FastAPI pipeline dependency backed by singleton runtime state. Versioned pipeline names such as `V1Pipeline` and `V2Pipeline` are not retained.

Core interfaces live in the core boundary. `ASRAdapter`, `DiarizationAdapter`, and `TranscriptionPipeline` protocols are defined under `coro.core`, while provider implementations live under `coro.backends`. Backend integrations are provider-first rather than capability-first so one provider module can own ASR and diarization integration without leaking backend-native objects into pipelines or response schemas.

Boundary response schemas live at the API boundary and are strict Pydantic models. Multipart request parsing remains in route code; Pydantic is not used to model multipart request bodies.

Considered alternatives included a `src/` layout, compatibility shims for the old server filenames, factory-only ASGI startup, preserving `/asr` and `/v1/listen`, exposing separate `/v1` and `/v2` transcription routes, keeping behavior-specific pipeline names, and running behavior-specific APIs as separate apps. The chosen shape prioritizes simple imports and launch commands, shared model lifecycle, one stable OpenAI-compatible route, core-owned protocols, and a smaller supported surface for the refactor.
