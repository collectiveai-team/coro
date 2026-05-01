"""Core Boundary protocols for adapters and transcription pipelines."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from asr_diar_server.audio import AudioInput
from asr_diar_server.core.types import SpeakerSegment, TranscriptToken


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


class TranscriptionPipeline(Protocol):
    """Protocol for the configured transcription pipeline."""

    async def transcribe(
        self,
        audio: AudioInput,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> dict: ...

    def stream(
        self,
        audio: AudioInput,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> AsyncIterator[dict]: ...
