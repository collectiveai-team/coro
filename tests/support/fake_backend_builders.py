"""In-memory adapter builders for the ASR and diarization Backend Adapter Factories.

``load_configured_asr_builder`` and ``load_configured_diarization_builder`` take
the place of ``load_asr_adapter_builder`` and ``load_diarization_adapter_builder``.
The builders they return load no model: each builds an adapter that keeps the
provider, model and options it was built with, so a test asserts on the adapter
that the factory returns.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from coro.core.models import SpeakerSegment, TranscriptToken


@dataclass
class ConfiguredASR:
    """An ASR Adapter that keeps its configuration and hears nothing."""

    provider: str
    model: str
    options: dict[str, Any] = field(default_factory=dict)
    honours_prompt: bool = False

    async def transcribe_pcm(
        self, pcm: bytes, *, language: str | None = None, prompt: str | None = None
    ) -> list[TranscriptToken]:
        """Return no tokens."""
        return []


@dataclass
class ConfiguredDiarization:
    """A Diarization Adapter that keeps its configuration and finds no speaker."""

    provider: str
    model: str
    options: dict[str, Any] = field(default_factory=dict)

    async def diarize_pcm(self, pcm: bytes) -> list[SpeakerSegment]:
        """Return an empty timeline."""
        return []


def load_configured_asr_builder(provider: str) -> Callable[..., ConfiguredASR]:
    """Return a builder of ``ConfiguredASR`` adapters for ``provider``."""

    def build(model: str, **options: Any) -> ConfiguredASR:
        return ConfiguredASR(provider=provider, model=model, options=options)

    return build


def load_configured_diarization_builder(provider: str) -> Callable[..., ConfiguredDiarization]:
    """Return a builder of ``ConfiguredDiarization`` adapters for ``provider``."""

    def build(model: str, **options: Any) -> ConfiguredDiarization:
        return ConfiguredDiarization(provider=provider, model=model, options=options)

    return build
