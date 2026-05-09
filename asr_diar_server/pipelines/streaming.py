"""Streaming Pipeline implementation.

Sources PCM through streamed file I/O rather than full-memory decode,
then tees each chunk to ASR Windowing and optional StreamingDiarizer.
Duration is computed from bytes consumed as chunks flow, never from a
full PCM buffer.
"""

from __future__ import annotations

import json

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
        try:
            path = await audio.temp_path()
            total_bytes = 0
            diarizer = (
                self._streaming_diarizer_factory()
                if self._streaming_diarizer_factory is not None
                else None
            )

            async def _chunks():
                nonlocal total_bytes
                async for chunk in stream_pcm_from_file(path, chunk_seconds=1.0):
                    if diarizer is not None:
                        diarizer.ingest_pcm_chunk(chunk)
                    total_bytes += len(chunk)
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
                timeline = diarizer.finalize()

            return build_transcription_response(result.tokens, timeline, duration)
        finally:
            await audio.cleanup()

    async def stream(
        self,
        audio: AudioInput,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ):
        try:
            path = await audio.temp_path()
            total_bytes = 0
            diarizer = (
                self._streaming_diarizer_factory()
                if self._streaming_diarizer_factory is not None
                else None
            )

            async def _chunks():
                nonlocal total_bytes
                async for chunk in stream_pcm_from_file(path, chunk_seconds=1.0):
                    if diarizer is not None:
                        diarizer.ingest_pcm_chunk(chunk)
                    total_bytes += len(chunk)
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
                    continue
                yield event

            duration = total_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE)

            timeline: list[SpeakerSegment] = []
            if diarizer is not None:
                timeline = diarizer.finalize()

            yield TranscriptDoneEvent(
                text=json.dumps(build_transcription_response(tokens, timeline, duration)),
            )
        finally:
            await audio.cleanup()
