"""Full-Memory Pipeline implementation."""

from __future__ import annotations

import json

from asr_diar_server.audio import BYTES_PER_SAMPLE, SAMPLE_RATE, AudioInput, convert_to_pcm_bytes
from asr_diar_server.core.response import build_whisperx_response
from asr_diar_server.core.protocols import ASRAdapter, DiarizationAdapter
from asr_diar_server.pipelines.windowing import ASRWindowing


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

    async def _pcm(self, audio: AudioInput) -> bytes:
        return await convert_to_pcm_bytes(await audio.read_bytes())

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
        return build_whisperx_response(result.tokens, timeline, duration)

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
            if event["type"] == "_tokens":
                tokens.extend(event["tokens"])
                continue
            yield event
        timeline = []
        if self._diarization is not None:
            timeline = await self._diarization.diarize_pcm(pcm)
        yield {
            "type": "transcript.text.done",
            "text": json.dumps(build_whisperx_response(tokens, timeline, duration)),
        }
