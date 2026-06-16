"""Full-Memory Pipeline implementation."""

from __future__ import annotations

import json

from coro.audio import BYTES_PER_SAMPLE, SAMPLE_RATE, AudioInput, convert_to_pcm_bytes
from coro.core.response import build_transcription_response
from coro.core.protocols import ASRAdapter, DiarizationAdapter
from coro.core.types import TokenBatchEvent, TranscriptDoneEvent
from coro.pipelines.windowing import ASRWindowing


# MARK: Full-Memory Pipeline
class FullMemoryPipeline:
    """Decode the full upload into memory before shared ASR Windowing."""

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

    # PCM Decoding ----------------------------------------------------------
    async def _pcm(self, audio: AudioInput) -> bytes:
        return await convert_to_pcm_bytes(await audio.read_bytes())

    # Batch Transcription ---------------------------------------------------
    async def transcribe(
        self,
        audio: AudioInput,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> dict:
        pcm = await self._pcm(audio)
        duration = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        result = await self._windowing.transcribe_pcm(
            pcm,
            asr=self._asr,
            language=language,
            prompt=prompt,
        )
        timeline = []
        if self._diarization is not None:
            timeline = await self._diarization.diarize_pcm(pcm)
        return build_transcription_response(result.tokens, timeline, duration)

    # Streaming Transcription ----------------------------------------------
    async def stream(
        self,
        audio: AudioInput,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ):
        pcm = await self._pcm(audio)
        duration = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        tokens = []
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
        timeline = []
        if self._diarization is not None:
            timeline = await self._diarization.diarize_pcm(pcm)
        yield TranscriptDoneEvent(
            text=json.dumps(build_transcription_response(tokens, timeline, duration)),
        )
