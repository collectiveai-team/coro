"""Streaming Pipeline implementation.

Sources PCM through streamed file I/O rather than full-memory decode,
then tees each chunk to ASR Windowing and optional StreamingDiarizer.
Duration is computed from bytes consumed as chunks flow, never from a
full PCM buffer.
"""

from __future__ import annotations

import logging
import time

from coro.audio import BYTES_PER_SAMPLE, SAMPLE_RATE, AudioInput, stream_pcm_from_file
from coro.core.protocols import ASRAdapter
from coro.core.types import SpeakerSegment, TokenBatchEvent
from coro.pipelines.done_frame import StreamingDoneFrame
from coro.pipelines.finalizer import (
    StreamingTranscriptFinalizer,
    build_streaming_response,
)
from coro.pipelines.transcript_store import TranscriptSpillStore
from coro.pipelines.windowing import ASRWindowing

logger = logging.getLogger(__name__)


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
        # Directory for the per-request transcript spill store. MUST be real
        # disk for flat RSS (on this platform /tmp is tmpfs/RAM-backed).
        self._spill_dir = spill_dir

    async def transcribe(
        self,
        audio: AudioInput,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> dict:
        started = time.perf_counter()
        chunk_count = 0
        diarizer_chunks = 0
        store = TranscriptSpillStore(directory=self._spill_dir)
        try:
            path = await audio.temp_path()
            total_bytes = 0
            diarizer = (
                self._streaming_diarizer_factory()
                if self._streaming_diarizer_factory is not None
                else None
            )
            logger.info(
                "streaming_pipeline transcribe start path=%s diarizer=%s",
                path,
                diarizer is not None,
            )

            async def _chunks():
                nonlocal total_bytes, chunk_count, diarizer_chunks
                async for chunk in stream_pcm_from_file(path, chunk_seconds=1.0):
                    chunk_count += 1
                    if diarizer is not None:
                        diarizer.ingest_pcm_chunk(chunk)
                        diarizer_chunks = getattr(diarizer, "processed_chunks", diarizer_chunks)
                    total_bytes += len(chunk)
                    if chunk_count == 1 or chunk_count % 10 == 0:
                        logger.info(
                            "streaming_pipeline transcribe chunk=%d bytes=%d "
                            "total_audio_s=%.2f diarizer_chunks=%d",
                            chunk_count,
                            len(chunk),
                            total_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE),
                            diarizer_chunks,
                        )
                    yield chunk

            finalizer = StreamingTranscriptFinalizer(store)
            async for event in self._windowing.stream_chunks(
                _chunks(),
                asr=self._asr,
                language=language,
                prompt=prompt,
            ):
                if isinstance(event, TokenBatchEvent):
                    finalizer.add_tokens(event.tokens)
            finalizer.finish()
            duration = total_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE)

            timeline: list[SpeakerSegment] = []
            if diarizer is not None:
                logger.info(
                    "streaming_pipeline transcribe diarizer finalize start "
                    "chunks=%d total_audio_s=%.2f",
                    getattr(diarizer, "processed_chunks", diarizer_chunks),
                    duration,
                )
                timeline = diarizer.finalize()
                logger.info(
                    "streaming_pipeline transcribe diarizer finalize complete timeline=%d",
                    len(timeline),
                )

            logger.info(
                "streaming_pipeline transcribe complete elapsed=%.3fs chunks=%d "
                "total_audio_s=%.2f segments=%d timeline=%d",
                time.perf_counter() - started,
                chunk_count,
                duration,
                store.segment_count,
                len(timeline),
            )
            return build_streaming_response(store, timeline)
        except Exception:
            logger.exception(
                "streaming_pipeline transcribe failed elapsed=%.3fs chunks=%d diarizer_chunks=%d",
                time.perf_counter() - started,
                chunk_count,
                diarizer_chunks,
            )
            raise
        finally:
            store.close()
            await audio.cleanup()

    async def stream(
        self,
        audio: AudioInput,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ):
        started = time.perf_counter()
        chunk_count = 0
        diarizer_chunks = 0
        store = TranscriptSpillStore(directory=self._spill_dir)
        store_released = False
        try:
            path = await audio.temp_path()
            total_bytes = 0
            diarizer = (
                self._streaming_diarizer_factory()
                if self._streaming_diarizer_factory is not None
                else None
            )
            logger.info(
                "streaming_pipeline sse start path=%s diarizer=%s", path, diarizer is not None
            )

            async def _chunks():
                nonlocal total_bytes, chunk_count, diarizer_chunks
                async for chunk in stream_pcm_from_file(path, chunk_seconds=1.0):
                    chunk_count += 1
                    if diarizer is not None:
                        diarizer.ingest_pcm_chunk(chunk)
                        diarizer_chunks = getattr(diarizer, "processed_chunks", diarizer_chunks)
                    total_bytes += len(chunk)
                    if chunk_count == 1 or chunk_count % 10 == 0:
                        logger.info(
                            "streaming_pipeline sse chunk=%d bytes=%d "
                            "total_audio_s=%.2f diarizer_chunks=%d",
                            chunk_count,
                            len(chunk),
                            total_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE),
                            diarizer_chunks,
                        )
                    yield chunk

            finalizer = StreamingTranscriptFinalizer(store)
            async for event in self._windowing.stream_chunks(
                _chunks(),
                asr=self._asr,
                language=language,
                prompt=prompt,
            ):
                if isinstance(event, TokenBatchEvent):
                    finalizer.add_tokens(event.tokens)
                    continue
                yield event
            finalizer.finish()

            duration = total_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE)

            timeline: list[SpeakerSegment] = []
            if diarizer is not None:
                logger.info(
                    "streaming_pipeline sse diarizer finalize start chunks=%d total_audio_s=%.2f",
                    getattr(diarizer, "processed_chunks", diarizer_chunks),
                    duration,
                )
                timeline = diarizer.finalize()
                logger.info(
                    "streaming_pipeline sse diarizer finalize complete timeline=%d",
                    len(timeline),
                )

            logger.info(
                "streaming_pipeline sse complete elapsed=%.3fs chunks=%d "
                "total_audio_s=%.2f segments=%d timeline=%d",
                time.perf_counter() - started,
                chunk_count,
                duration,
                store.segment_count,
                len(timeline),
            )
            # Ownership of the store passes to the frame, which closes it once
            # rendered; the final transcript is never materialised in memory.
            store_released = True
            yield StreamingDoneFrame(store=store, timeline=timeline)
        except Exception:
            logger.exception(
                "streaming_pipeline sse failed elapsed=%.3fs chunks=%d diarizer_chunks=%d",
                time.perf_counter() - started,
                chunk_count,
                diarizer_chunks,
            )
            raise
        finally:
            if not store_released:
                store.close()
            await audio.cleanup()
