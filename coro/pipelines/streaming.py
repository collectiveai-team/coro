"""Streaming Pipeline implementation.

Sources PCM through streamed file I/O rather than full-memory decode,
then tees each chunk to ASR Windowing and optional StreamingDiarizer.
Duration is computed from bytes consumed as chunks flow, never from a
full PCM buffer.

The Streaming Diarization Feed runs its mel preprocessor and model forward
step off the event loop, like every other adapter: doing that work inline
would stall every other in-flight request and block transcript delta events
from flushing while the diarizer runs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from coro.audio import BYTES_PER_SAMPLE, SAMPLE_RATE, AudioInput, stream_pcm_from_file
from coro.core.protocols import ASRAdapter
from coro.core.models import (
    SpeakerSegment,
    StreamEvent,
    TokenBatchEvent,
    TranscriptionResult,
)
from coro.pipelines.done_frame import StreamingDoneFrame
from coro.pipelines.finalizer import (
    StreamingTranscriptFinalizer,
    build_streaming_response,
)
from coro.pipelines.transcript_store import TranscriptSpillStore
from coro.pipelines.windowing import ASRWindowing

logger = logging.getLogger(__name__)

_CHUNK_SECONDS = 1.0
_CHUNK_LOG_INTERVAL = 10


@dataclass
class _RunOutcome:
    """Results a streaming run produces only after its event stream is drained."""

    timeline: list[SpeakerSegment] = field(default_factory=list)
    duration: float = 0.0
    chunk_count: int = 0
    diarizer_chunks: int = 0


class StreamingPipeline:
    """Stream PCM from a spooled upload file through ASR Windowing.

    When a ``streaming_diarizer_factory`` is provided, each PCM chunk is
    also fed to a fresh :class:`StreamingDiarizer` instance so that
    diarization runs in bounded memory in parallel with ASR windowing.
    """

    def __init__(
        self,
        *,
        asr: ASRAdapter,
        windowing: ASRWindowing | None = None,
        streaming_diarizer_factory=None,
        spill_dir: str | None = None,
    ) -> None:
        self._asr = asr
        self._windowing = windowing or ASRWindowing()
        self._streaming_diarizer_factory = streaming_diarizer_factory
        # Directory for the per-request transcript spill store, already resolved
        # to real disk by coro.pipelines.spill (a tmpfs path would defeat the
        # spill).  None falls back to the system temp dir, for tests only.
        self._spill_dir = spill_dir

    # Batch Transcription ---------------------------------------------------
    async def transcribe(
        self,
        audio: AudioInput,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe an upload and assemble the response in one call."""
        store = TranscriptSpillStore(directory=self._spill_dir)
        outcome = _RunOutcome()
        try:
            async for _ in self._run(audio, store, outcome, language=language, prompt=prompt):
                pass
            return build_streaming_response(store, outcome.timeline)
        finally:
            store.close()
            await audio.cleanup()

    # Streaming Transcription ----------------------------------------------
    async def stream(
        self,
        audio: AudioInput,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ):
        """Yield transcript delta events, then a store-backed done frame."""
        store = TranscriptSpillStore(directory=self._spill_dir)
        outcome = _RunOutcome()
        store_released = False
        try:
            async for event in self._run(audio, store, outcome, language=language, prompt=prompt):
                yield event
            # Ownership of the store passes to the frame, which closes it once
            # rendered; the final transcript is never materialised in memory.
            store_released = True
            yield StreamingDoneFrame(store=store, timeline=outcome.timeline)
        finally:
            if not store_released:
                store.close()
            await audio.cleanup()

    # Shared Run ------------------------------------------------------------
    async def _run(
        self,
        audio: AudioInput,
        store: TranscriptSpillStore,
        outcome: _RunOutcome,
        *,
        language: str | None,
        prompt: str | None,
    ) -> AsyncIterator[StreamEvent]:
        """Drive one request, yielding public events and filling ``outcome``.

        Shared by batch and SSE transcription so the two paths cannot drift:
        they differ only in how they consume the events and the finished store.
        """
        started = time.perf_counter()
        try:
            path = await audio.temp_path()
            diarizer = (
                self._streaming_diarizer_factory()
                if self._streaming_diarizer_factory is not None
                else None
            )
            logger.info("streaming_pipeline start path=%s diarizer=%s", path, diarizer is not None)

            finalizer = StreamingTranscriptFinalizer(store)
            async for event in self._windowing.stream_chunks(
                self._chunks(path, diarizer, outcome),
                asr=self._asr,
                language=language,
                prompt=prompt,
            ):
                if isinstance(event, TokenBatchEvent):
                    finalizer.add_tokens(event.tokens)
                    continue
                yield event
            finalizer.finish()

            outcome.timeline = await self._finalize_diarizer(diarizer, outcome)
            logger.info(
                "streaming_pipeline complete elapsed=%.3fs chunks=%d "
                "total_audio_s=%.2f segments=%d timeline=%d",
                time.perf_counter() - started,
                outcome.chunk_count,
                outcome.duration,
                store.segment_count,
                len(outcome.timeline),
            )
        except Exception:
            logger.exception(
                "streaming_pipeline failed elapsed=%.3fs chunks=%d diarizer_chunks=%d",
                time.perf_counter() - started,
                outcome.chunk_count,
                outcome.diarizer_chunks,
            )
            raise

    async def _chunks(self, path: str, diarizer, outcome: _RunOutcome) -> AsyncIterator[bytes]:
        """Yield PCM chunks, teeing each into the Streaming Diarization Feed."""
        total_bytes = 0
        async for chunk in stream_pcm_from_file(path, chunk_seconds=_CHUNK_SECONDS):
            outcome.chunk_count += 1
            if diarizer is not None:
                # Off-loop: ingest runs a mel preprocessor and a model forward
                # step, which would otherwise block every other request and
                # stall delta events for the duration of the chunk.
                await asyncio.to_thread(diarizer.ingest_pcm_chunk, chunk)
                outcome.diarizer_chunks = getattr(
                    diarizer, "processed_chunks", outcome.diarizer_chunks
                )
            total_bytes += len(chunk)
            outcome.duration = total_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE)
            if outcome.chunk_count == 1 or outcome.chunk_count % _CHUNK_LOG_INTERVAL == 0:
                logger.info(
                    "streaming_pipeline chunk=%d bytes=%d total_audio_s=%.2f diarizer_chunks=%d",
                    outcome.chunk_count,
                    len(chunk),
                    outcome.duration,
                    outcome.diarizer_chunks,
                )
            yield chunk

    async def _finalize_diarizer(self, diarizer, outcome: _RunOutcome) -> list[SpeakerSegment]:
        """Close out the Streaming Diarization Feed and return its timeline."""
        if diarizer is None:
            return []
        logger.info(
            "streaming_pipeline diarizer finalize start chunks=%d total_audio_s=%.2f",
            getattr(diarizer, "processed_chunks", outcome.diarizer_chunks),
            outcome.duration,
        )
        # Off-loop for the same reason as ingest: finalize runs per-speaker VAD
        # post-processing over the whole prediction tensor.
        timeline = await asyncio.to_thread(diarizer.finalize)
        logger.info(
            "streaming_pipeline diarizer finalize complete timeline=%d",
            len(timeline),
        )
        return timeline
