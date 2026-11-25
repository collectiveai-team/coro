"""Full-Memory Pipeline implementation."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import asdict

from coro.audio import BYTES_PER_SAMPLE, SAMPLE_RATE, AudioInput, convert_path_to_pcm_bytes
from coro.core.response import build_transcription_response
from coro.core.protocols import ASRAdapter, DiarizationAdapter
from coro.core.models import (
    PipelineStreamEvent,
    SpeakerSegment,
    TokenBatchEvent,
    TranscriptDoneEvent,
    TranscriptionResult,
    TranscriptToken,
)
from coro.pipelines.windowing import ASRWindowing, LanguageState


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

    # Window Geometry -------------------------------------------------------
    @property
    def windowing(self) -> ASRWindowing:
        """The ASR Windowing this pipeline transcribes with."""
        return self._windowing

    def with_windowing(self, windowing: ASRWindowing) -> FullMemoryPipeline:
        """Return a pipeline over the same adapters that transcribes with ``windowing``."""
        return FullMemoryPipeline(asr=self._asr, diarization=self._diarization, windowing=windowing)

    def without_diarization(self) -> FullMemoryPipeline:
        """Return a pipeline over the same ASR Adapter that skips diarization."""
        return FullMemoryPipeline(asr=self._asr, diarization=None, windowing=self._windowing)

    # PCM Decoding ----------------------------------------------------------
    async def _pcm(self, audio: AudioInput) -> bytes:
        """Decode the upload to PCM, letting ffmpeg read it from disk.

        Decoding from the path rather than from ``read_bytes()`` keeps the
        encoded upload out of resident memory; the decoded PCM is still fully
        materialised, which is what makes this the Full-Memory Pipeline.
        """
        return await convert_path_to_pcm_bytes(await audio.temp_path())

    # Batch Transcription ---------------------------------------------------
    async def transcribe(
        self,
        audio: AudioInput,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> TranscriptionResult:
        # The pipeline consuming the audio owns its temp file, mirroring the
        # Streaming Pipeline: uploads are now spooled eagerly, so skipping this
        # would leak a file the size of the upload on every request.
        try:
            pcm = await self._pcm(audio)
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
            return build_transcription_response(
                result.tokens, timeline, duration, detected_language=result.detected_language
            )
        finally:
            await audio.cleanup()

    # Streaming Transcription ----------------------------------------------
    async def stream(
        self,
        audio: AudioInput,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> AsyncIterator[PipelineStreamEvent]:
        try:
            pcm = await self._pcm(audio)
            duration = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
            tokens: list[TranscriptToken] = []
            state = LanguageState()
            async for event in self._windowing.stream_pcm(
                pcm,
                asr=self._asr,
                language=language,
                prompt=prompt,
                language_state=state,
            ):
                if isinstance(event, TokenBatchEvent):
                    tokens.extend(event.tokens)
                    continue
                yield event
            timeline: list[SpeakerSegment] = []
            if self._diarization is not None:
                timeline = await self._diarization.diarize_pcm(pcm)
            yield TranscriptDoneEvent(
                text=json.dumps(
                    asdict(
                        build_transcription_response(
                            tokens, timeline, duration, detected_language=state.resolved
                        )
                    )
                ),
            )
        finally:
            await audio.cleanup()
