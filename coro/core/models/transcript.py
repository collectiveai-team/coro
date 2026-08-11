"""Project-Owned Transcript Model types.

These lightweight dataclasses are used at package boundaries so that
backend-native types do not leak through.
Backend adapters convert native objects into these types at adapter edges.
"""

from __future__ import annotations

from dataclasses import dataclass


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
    """A response word carrying its backend's real timing.

    ``start``/``end`` come from the ASR Adapter's own per-word output, ``speaker``
    from word-level attribution, and ``overlap`` marks a word whose span contains
    concurrently active speakers. Serialised to dict at the API boundary.

    ``score`` is the backend's own probability **when it reports one**; backends
    that express no probability fall back to ``1.0``, which is a placeholder and
    not a measured certainty. Do not read it as confidence without checking the
    backend first.
    """

    word: str
    start: float
    end: float
    score: float
    speaker: str
    overlap: bool = False


@dataclass
class SpeakerSegment:
    """A speaker timeline entry produced by the Diarization Adapter."""

    start: float
    end: float
    speaker: int
