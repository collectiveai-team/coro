"""Core Boundary protocols for adapters and transcription pipelines."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from coro.audio import AudioInput
from coro.core.models import (
    PipelineStreamEvent,
    SpeakerSegment,
    TranscriptionResult,
    TranscriptToken,
)


# MARK: Backend Adapter Protocols
class ASRAdapter(Protocol):
    """Protocol for ASR adapters used by transcription pipelines."""

    async def transcribe_pcm(
        self,
        pcm: bytes,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> list[TranscriptToken]: ...


class DiarizationAdapter(Protocol):
    """Protocol for diarization adapters used by transcription pipelines."""

    async def diarize_pcm(self, pcm: bytes) -> list[SpeakerSegment]: ...


class StreamingDiarizerFactory(Protocol):
    """Protocol for factories that produce per-request StreamingDiarizer instances."""

    def __call__(self):
        """Return a fresh StreamingDiarizer bound to the shared model."""


# MARK: Pipeline Protocols
class TranscriptionPipeline(Protocol):
    """Protocol for the configured transcription pipeline."""

    async def transcribe(
        self,
        audio: AudioInput,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> TranscriptionResult: ...

    def stream(
        self,
        audio: AudioInput,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> AsyncIterator[PipelineStreamEvent]: ...
