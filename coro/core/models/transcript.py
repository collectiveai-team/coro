"""Project-Owned Transcript Model types.

These lightweight dataclasses are used at package boundaries so that
backend-native types do not leak through.
Backend adapters convert native objects into these types at adapter edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# MARK: Token and Segment Types
@dataclass
class TranscriptToken:
    """A single transcript word/token with timing and confidence.

    Corresponds to a backend-native word/token but is owned by this package.
    """

    start: float
    end: float
    text: str
    probability: float | None = None

    def duration(self) -> float:
        """Return token duration in seconds."""
        return max(0.0, self.end - self.start)


@dataclass
class TranscriptWord:
    """A single interpolated word with timing and speaker attribution.

    Produced by ``_build_words_for_segment``; serialised to dict at the API boundary.
    """

    word: str
    start: float
    end: float
    score: float
    speaker: str


@dataclass
class TranscriptSegment:
    """A speaker-attributed transcript segment built from one or more tokens."""

    start: float
    end: float
    text: str
    speaker: int = -1
    words: list[TranscriptWord] = field(default_factory=list)


@dataclass
class SpeakerSegment:
    """A speaker timeline entry produced by the Diarization Adapter."""

    start: float
    end: float
    speaker: int
