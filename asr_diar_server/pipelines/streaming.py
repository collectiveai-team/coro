"""Streaming Pipeline implementation.

Sources PCM through streamed file I/O rather than full-memory decode,
then tees each chunk to ASR Windowing and optional StreamingDiarizer.
Duration is computed from bytes consumed as chunks flow, never from a
full PCM buffer.
"""

from __future__ import annotations

import json
import logging
import time

from asr_diar_server.audio import BYTES_PER_SAMPLE, SAMPLE_RATE, AudioInput, stream_pcm_from_file
from asr_diar_server.core.response import build_transcription_response
from asr_diar_server.core.protocols import ASRAdapter
from asr_diar_server.core.types import (
    SpeakerSegment,
    TokenBatchEvent,
    TranscriptDoneEvent,
    TranscriptToken,
)
from asr_diar_server.pipelines.windowing import ASRWindowing

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
    ) -> None:
        self._asr = asr
        self._windowing = windowing or ASRWindowing()
        self._streaming_diarizer_factory = streaming_diarizer_factory

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
                            "streaming_pipeline transcribe chunk=%d bytes=%d total_audio_s=%.2f diarizer_chunks=%d",
                            chunk_count,
                            len(chunk),
                            total_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE),
                            diarizer_chunks,
                        )
                    yield chunk

            result = await self._windowing.transcribe_chunks(
                _chunks(),
                asr=self._asr,
                language=language,
                prompt=prompt,
            )
            duration = total_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE)

            timeline: list[SpeakerSegment] = []
            if diarizer is not None:
                logger.info(
                    "streaming_pipeline transcribe diarizer finalize start chunks=%d total_audio_s=%.2f",
                    getattr(diarizer, "processed_chunks", diarizer_chunks),
                    duration,
                )
                timeline = diarizer.finalize()
                logger.info(
                    "streaming_pipeline transcribe diarizer finalize complete timeline=%d",
                    len(timeline),
                )

            logger.info(
                "streaming_pipeline transcribe complete elapsed=%.3fs chunks=%d total_audio_s=%.2f tokens=%d timeline=%d",
                time.perf_counter() - started,
                chunk_count,
                duration,
                len(result.tokens),
                len(timeline),
            )
            return build_transcription_response(result.tokens, timeline, duration)
        except Exception:
            logger.exception(
                "streaming_pipeline transcribe failed elapsed=%.3fs chunks=%d diarizer_chunks=%d",
                time.perf_counter() - started,
                chunk_count,
                diarizer_chunks,
            )
            raise
        finally:
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
        try:
            path = await audio.temp_path()
            total_bytes = 0
            diarizer = (
                self._streaming_diarizer_factory()
                if self._streaming_diarizer_factory is not None
                else None
            )
            logger.info("streaming_pipeline sse start path=%s diarizer=%s", path, diarizer is not None)

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
                            "streaming_pipeline sse chunk=%d bytes=%d total_audio_s=%.2f diarizer_chunks=%d",
                            chunk_count,
                            len(chunk),
                            total_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE),
                            diarizer_chunks,
                        )
                    yield chunk

            tokens: list[TranscriptToken] = []
            async for event in self._windowing.stream_chunks(
                _chunks(),
                asr=self._asr,
                language=language,
                prompt=prompt,
            ):
                if isinstance(event, TokenBatchEvent):
                    tokens.extend(event.tokens)
                    logger.info(
                        "streaming_pipeline sse token_batch tokens=%d total_tokens=%d",
                        len(event.tokens),
                        len(tokens),
                    )
                    continue
                yield event

            duration = total_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE)

            timeline: list[SpeakerSegment] = []
            if diarizer is not None:
                logger.info(
                    "streaming_pipeline sse diarizer finalize start chunks=%d total_audio_s=%.2f",
                    getattr(diarizer, "processed_chunks", diarizer_chunks),
                    duration,
                )
                timeline = diarizer.finalize()
                logger.info("streaming_pipeline sse diarizer finalize complete timeline=%d", len(timeline))

            logger.info(
                "streaming_pipeline sse complete elapsed=%.3fs chunks=%d total_audio_s=%.2f tokens=%d timeline=%d",
                time.perf_counter() - started,
                chunk_count,
                duration,
                len(tokens),
                len(timeline),
            )
            yield TranscriptDoneEvent(
                text=json.dumps(build_transcription_response(tokens, timeline, duration)),
            )
        except Exception:
            logger.exception(
                "streaming_pipeline sse failed elapsed=%.3fs chunks=%d diarizer_chunks=%d",
                time.perf_counter() - started,
                chunk_count,
                diarizer_chunks,
            )
            raise
        finally:
            await audio.cleanup()
