"""Server response shapes the benchmark reads back.

The benchmark parses the JSON bodies of ``POST /v1/audio/transcriptions``
(``diarized_json``) and ``POST /v1/listen`` (Deepgram) without importing the
serving layer, so these TypedDicts describe only the keys it reads. Every key is
optional because each wire format carries a different subset.
"""

from __future__ import annotations

from typing import TypedDict


class HypSegment(TypedDict, total=False):
    """One ``segments`` entry of a ``diarized_json`` response."""

    start: float | None
    end: float | None
    text: str | None
    speaker: str | int


class HypWord(TypedDict, total=False):
    """One per-word entry: top-level ``word_segments``/``words`` or Deepgram ``words``."""

    start: float | None
    end: float | None
    word: str | None
    speaker: str | int | None


class DeepgramAlternative(TypedDict, total=False):
    words: list[HypWord]


class DeepgramChannel(TypedDict, total=False):
    alternatives: list[DeepgramAlternative]


class DeepgramResults(TypedDict, total=False):
    channels: list[DeepgramChannel]


class TranscriptResponse(TypedDict, total=False):
    """A transcription response body, in either wire format."""

    segments: list[HypSegment]
    word_segments: list[HypWord]
    words: list[HypWord]
    results: DeepgramResults


class HealthResponse(TypedDict, total=False):
    """The ``GET /health`` body, as far as the benchmark reads it."""

    ready: bool
    warmup_ready: bool
    startup_selection: dict[str, str]
