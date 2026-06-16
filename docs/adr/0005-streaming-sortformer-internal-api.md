# Streaming Sortformer Internal API Dependency

## Status

Accepted

## Context

The **Streaming Pipeline** promises bounded memory per request: peak PCM held in the process should be capped at roughly one ASR window plus streaming diarization state, regardless of audio duration. To achieve this for diarization, the pipeline must feed PCM incrementally to the Sortformer model rather than accumulating the full audio array in memory.

NVIDIA's documented streaming-Sortformer API surface — the `nvidia/diar_streaming_sortformer_4spk-v2` model card and the `e2e_diarize_speech.py` example — centres on `model.diarize(audio=full_audio)` with parameter sweeps over `chunk_len`, `chunk_right_context`, `fifo_len`, `spkcache_update_period`, and `spkcache_len`. That entry point requires the complete audio array in memory before any speaker-timeline work begins. The existing **NeMo Diarization Adapter** (`NemoDiarizationAdapter.diarize_pcm`) already uses this entry point and writes full PCM to a temporary WAV before calling it. Both paths are incompatible with bounded-memory processing.

The Sortformer model class (`SortformerEncLabelModel`) exposes a lower-level `forward_streaming_step` method that processes one mel-spectrogram chunk at a time, carrying streaming state and cumulative predictions between calls. This method is public on the model class and is the primitive that NVIDIA's own `model.forward_streaming` uses internally, but it is not part of NVIDIA's documented streaming API surface. Using it directly means the **Streaming Diarizer** depends on NeMo internals.

This ADR records that dependency so that NeMo version upgrades flag the integration risk explicitly.

## Decision

The **Streaming Diarizer** will call `SortformerEncLabelModel.forward_streaming_step` directly instead of `model.diarize(audio=...)`. The specific NeMo APIs the **Streaming Diarizer** depends on are:

- `SortformerEncLabelModel.forward_streaming_step(processed_signal, processed_signal_length, streaming_state, total_preds, left_offset, right_offset)` — per-chunk inference returning updated state and predictions.
- `SortformerModules.init_streaming_state` — initialises the per-request streaming state (chunk counter, speaker cache, FIFO buffers).
- `SortformerModules._check_streaming_parameters` — validates that the configured chunk and cache parameters are compatible with the model architecture.
- `AudioToMelSpectrogramPreprocessor` (with `window_size=0.025, normalize="NA", n_fft=512, features=128, pad_to=0`) — mel-feature extraction that the **Streaming Diarizer** owns per request, with left-context carry between chunks.
- `ts_vad_post_processing` — post-processing over the cumulative `total_preds` tensor to produce speaker-active timelines, the same function the batch path uses indirectly through `model._diarize_output_processing`.

The NeMo version is pinned to `nemo-toolkit>=2.7.0,<2.8.0` in `pyproject.toml`. Minor NeMo version bumps may rename, re-signature, or remove any of these APIs; the ADR makes that risk visible so upgrade PRs can check the integration proactively.

The existing **NeMo Diarization Adapter** (`NemoDiarizationAdapter`, batch path) is **not changed**. It continues to call `model.diarize(audio=...)` with full PCM. Only the new **Streaming Diarizer** uses the internal API.

## Consequences

- The **Streaming Diarizer** owns its integration surface: mel preprocessing, streaming state lifecycle, left-context carry, cumulative prediction accumulation, and post-processing. Future NeMo changes to any of these areas require a manual update to the **Streaming Diarizer**.
- NeMo minor version bumps must be treated as potentially breaking for the **Streaming Diarizer**. CI should run the env-gated real-model smoke test (`CORO_RUN_REAL_MODEL_TESTS=1`) against any new NeMo version before merging.
- The batch **NeMo Diarization Adapter** is insulated from this risk because it uses the documented `diarize()` entry point, which has a stable public contract.
- The `nvidia/diar_streaming_sortformer_4spk-v2` model card and the `e2e_diarize_speech.py` example remain the documented baseline. If NVIDIA adds a documented chunk-by-chunk inference API in a future release, the **Streaming Diarizer** should migrate to it and this ADR should be superseded.

## Alternatives Considered

- **Use `model.diarize(audio=full_audio)` and accept full-memory diarization**: rejected — the **Streaming Pipeline** would then have bounded-memory ASR but unbounded-memory diarization, which defeats the architectural purpose of the pipeline. The user has confirmed that a half-measure is worse than not shipping the pipeline at all.
- **Write PCM chunks to a temporary WAV and batch-diagnose at the end**: rejected — peak memory at finalize time would be proportional to audio duration, making the pipeline architecturally indistinguishable from the **Full-Memory Pipeline**.
- **Fork or vendor the Sortformer streaming inference code**: rejected — the surface area is manageable (five APIs behind one class) and vendoring would diverge from upstream fixes.
