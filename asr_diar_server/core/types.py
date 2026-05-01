"""Project-Owned Transcript Model types.

These lightweight dataclasses are used at package boundaries so that
backend-native types (e.g. whisperlivekit ASRToken) do not leak through.
Backend adapters convert native objects into these types at adapter edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TranscriptToken:
    """A single transcript word/token with timing and confidence.

    Corresponds to a whisperlivekit ASRToken but is owned by this package.
    """

    start: float
    end: float
    text: str
    probability: float | None = None

    def duration(self) -> float:
        """Return token duration in seconds."""
        return max(0.0, self.end - self.start)


@dataclass
class TranscriptSegment:
    """A speaker-attributed transcript segment built from one or more tokens."""

    start: float
    end: float
    text: str
    speaker: int = -1
    words: list[dict] = field(default_factory=list)


@dataclass
class SpeakerSegment:
    """A speaker timeline entry produced by the Diarization Adapter."""

    start: float
    end: float
    speaker: int
