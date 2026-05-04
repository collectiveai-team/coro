"""Chunked-File Pipeline implementation."""

from __future__ import annotations

import json

from asr_diar_server.audio import BYTES_PER_SAMPLE, SAMPLE_RATE, AudioInput, stream_pcm_from_file
from asr_diar_server.core.response import build_transcription_response
from asr_diar_server.core.protocols import ASRAdapter, DiarizationAdapter
from asr_diar_server.core.types import (
    SpeakerSegment,
    TokenBatchEvent,
    TranscriptDoneEvent,
    TranscriptToken,
)
from asr_diar_server.pipelines.windowing import ASRWindowing


# MARK: Chunked-File Pipeline
class ChunkedFilePipeline:
    """Spool upload to a temp file and stream PCM chunks from that file."""

    def __init__(
        self,
        *,
        asr: ASRAdapter,
        diarization: DiarizationAdapter | None = None,
        windowing: ASRWindowing | None = None,
    ) -> None:
        self._asr = asr
        self._diarization = diarization
        self._windowing = windowing or ASRWindowing()

    # PCM Streaming ---------------------------------------------------------
    async def _read_pcm(self, audio: AudioInput) -> bytes:
        chunks: list[bytes] = []
        async for chunk in stream_pcm_from_file(await audio.temp_path(), chunk_seconds=1.0):
            chunks.append(chunk)
        return b"".join(chunks)

    # Batch Transcription ---------------------------------------------------
    async def transcribe(
        self,
        audio: AudioInput,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> dict:
        try:
            pcm = await self._read_pcm(audio)
            duration = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
            result = await self._windowing.transcribe_pcm(
                pcm,
                asr=self._asr,
                language=language,
                prompt=prompt,
            )
            timeline: list[SpeakerSegment] = []
            if self._diarization is not None:
                timeline = await self._diarization.diarize_pcm(pcm)
            return build_transcription_response(result.tokens, timeline, duration)
        finally:
            await audio.cleanup()

    # Streaming Transcription ----------------------------------------------
    async def stream(
        self,
        audio: AudioInput,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ):
        try:
            pcm = await self._read_pcm(audio)
            duration = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
            tokens: list[TranscriptToken] = []
            async for event in self._windowing.stream_pcm(
                pcm,
                asr=self._asr,
                language=language,
                prompt=prompt,
            ):
                if isinstance(event, TokenBatchEvent):
                    tokens.extend(event.tokens)
                    continue
                yield event
            timeline: list[SpeakerSegment] = []
            if self._diarization is not None:
                timeline = await self._diarization.diarize_pcm(pcm)
            yield TranscriptDoneEvent(
                text=json.dumps(build_transcription_response(tokens, timeline, duration)),
            )
        finally:
            await audio.cleanup()
