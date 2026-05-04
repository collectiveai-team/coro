Status: needs-triage

# Streaming Pipeline PRD

## Problem Statement

The packaged transcription service currently offers a **Full-Memory Pipeline** and a **Chunked-File Pipeline**, but neither provides the desired bounded-memory/no-disk behavior. The **Full-Memory Pipeline** materializes the uploaded audio and decoded PCM in memory, while the **Chunked-File Pipeline** writes request audio to a temp file and still rejoins decoded PCM into a full buffer before ASR and diarization. For long audio files, this makes memory usage scale with input duration and makes `chunked-file` misleading as a startup-selected pipeline.

## Solution

Replace the legacy **Chunked-File Pipeline** with a **Streaming Pipeline**. The **Streaming Pipeline** should implement a **No-Disk Audio Flow** by piping upload bytes directly into ffmpeg stdin, consuming decoded PCM as sequential chunks, feeding **Incremental ASR Windowing**, and using a **Streaming Diarization Feed** for SortFormer. The public **Transcription API Contract** remains unchanged: `/v1/audio/transcriptions` continues to return the same JSON response and SSE stream shape, with speaker assignment performed in final response construction.

## User Stories

1. As an API client, I want long audio requests to avoid loading the whole decoded PCM into RAM, so that transcription can handle longer files more reliably.
2. As an API client, I want the transcription endpoint path and request fields to remain unchanged, so that existing integrations do not need to change.
3. As an API client using SSE, I want transcript deltas to continue arriving as ASR windows complete, so that streaming response behavior remains familiar.
4. As an API client using JSON responses, I want the final response shape to remain unchanged, so that downstream parsers continue to work.
5. As an operator, I want request audio to avoid temp-file staging, so that disk IO and cleanup failure modes are reduced.
6. As an operator, I want memory use for decoded audio to be bounded by the active ASR window, overlap, diarization state, and result objects, so that resource usage does not grow linearly with audio duration.
7. As an operator, I want `ASR_DIAR_PIPELINE=streaming` to clearly describe the bounded-memory pipeline, so that startup configuration matches runtime behavior.
8. As an operator, I want `ASR_DIAR_PIPELINE=chunked-file` removed rather than silently aliased, so that obsolete behavior is not preserved under a misleading name.
9. As a developer, I want **Incremental ASR Windowing** to preserve the existing 30-second window and 2-second overlap semantics, so that transcription behavior is not intentionally retuned during the memory rewrite.
10. As a developer, I want ASR windowing to be testable independently from ffmpeg and model adapters, so that offset and overlap behavior can be verified with fake ASR adapters.
11. As a developer, I want ffmpeg upload streaming to be testable independently from pipeline orchestration, so that pipe handling and chunk alignment can be verified without loading models.
12. As a developer, I want the **Streaming Diarization Feed** to hide SortFormer-specific streaming state behind a small interface, so that pipelines do not manage backend-native details.
13. As a developer, I want final speaker assignment to stay in response construction, so that diarization streaming input does not force a new live speaker-event contract.
14. As a maintainer, I want the **Full-Memory Pipeline** to remain available as a simple baseline, so that benchmarks and debugging can compare bounded-memory behavior against the current straightforward path.
15. As a maintainer, I want tests to prove no full decoded PCM buffer is assembled in the streaming path, so that future refactors do not regress into `b"".join(chunks)` behavior.
16. As a maintainer, I want cleanup behavior to be simple for the streaming path, so that there are no request-scoped temp files to unlink after success or failure.
17. As a benchmark author, I want the **Streaming Pipeline** to preserve the **Transcription API Contract**, so that existing performance and quality benchmarks can compare pipeline implementations without endpoint-specific clients.
18. As a benchmark author, I want the old **Chunked-File Pipeline** terminology marked legacy in the glossary, so that benchmark results and architecture notes do not overstate its streaming value.

## Implementation Decisions

- Add a **Streaming Pipeline** startup selector and remove the obsolete `chunked-file` selector instead of aliasing it.
- Keep the **Full-Memory Pipeline** as the simple baseline implementation.
- Implement **No-Disk Audio Flow** by streaming upload bytes into ffmpeg stdin and reading decoded PCM chunks from ffmpeg stdout.
- Drain ffmpeg stderr concurrently while streaming to avoid subprocess deadlocks.
- Keep PCM chunk sizes aligned to 16-bit sample boundaries.
- Introduce **Incremental ASR Windowing** as a deep module that consumes sequential PCM chunks and emits the same accepted transcript events as full-buffer **ASR Windowing**.
- Preserve the current 30-second ASR window and 2-second overlap semantics.
- Track byte offsets incrementally so token timestamps match the existing full-buffer windowing behavior.
- Introduce a small streaming diarization interface or feeder that receives PCM chunks and returns the final speaker timeline at finish.
- Feed ASR windowing and diarization from the same decoded PCM stream without teeing into full-memory buffers.
- Keep final response construction responsible for assigning speakers to ASR tokens.
- Preserve the existing `/v1/audio/transcriptions` JSON and SSE response contracts.
- Avoid adding live speaker-attributed SSE deltas in this change.
- Avoid request-scoped temporary audio files in the streaming path.
- Update settings validation and app startup construction to know `full-memory` and `streaming` as the supported configured transcription pipelines.
- Update documentation and tests to describe **Chunked-File Pipeline** as legacy behavior rather than the desired bounded-memory design.

## Testing Decisions

- Good tests should assert externally observable behavior: response shape, emitted transcript deltas, timestamp offsets, selector validation, bounded buffering, and cleanup behavior.
- Tests should avoid asserting incidental implementation details except where needed to prevent the specific full-buffer regression this feature is designed to remove.
- Test ffmpeg streaming with fake subprocess or monkeypatched chunk iterators so pipe orchestration can be validated without real model execution.
- Test **Incremental ASR Windowing** with fake ASR adapters, using short configured windows and overlaps to verify window offsets, prompt carry, final partial windows, and transcript delta emission.
- Test **Streaming Pipeline** orchestration with fake ASR and diarization adapters to verify that chunks are consumed incrementally and final responses match the existing schema.
- Test that the streaming path does not call temp-file APIs or require **Audio Input Cleanup** for request-scoped audio files.
- Test settings validation so `streaming` is accepted and `chunked-file` is rejected once removed.
- Reuse prior pipeline tests, ASR windowing tests, audio chunking tests, SSE streaming tests, response schema tests, and app factory tests as patterns.

## Out of Scope

- Changing the public **Transcription API Contract**.
- Adding a new transcription endpoint for streaming behavior.
- Adding live speaker-attributed SSE deltas.
- Retuning ASR window size, ASR overlap, or prompt-carry behavior.
- Removing the **Full-Memory Pipeline**.
- Changing ASR or diarization model providers.
- Changing final response grouping or speaker-assignment semantics except as required to consume a streamed diarization timeline.
- Preserving `chunked-file` as a compatibility alias.

## Further Notes

This PRD follows ADR-0002: **Replace Chunked-File with Streaming Pipeline**. The key success criterion is not merely chunked decoding; it is that the streaming path avoids request-audio temp files and never materializes the full upload or decoded PCM while preserving the existing endpoint contract.
