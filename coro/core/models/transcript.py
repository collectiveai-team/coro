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
    """A single word with a **Measured Word Start** and speaker attribution.

    Produced by ``_build_words_for_segment`` from the ASR's own tokens;
    serialised to dict at the API boundary. ``start`` is the backend's token
    emission time. ``end`` is still derived from the following word's start —
    see ADR 0008; nothing reads word ends and obtaining true ones belongs
    upstream in ``onnx-asr``. ``score`` is None when the backend expresses no
    per-word probability, rather than claiming a fabricated 1.0.
    """

    word: str
    start: float
    end: float
    score: float | None
    speaker: str


@dataclass
class TranscriptSegment:
    """A speaker-attributed transcript segment built from one or more tokens.

    ``tokens`` retains the tokens the segment was grouped from, so word timings
    can be read from the ASR rather than interpolated across the span. The
    Streaming Pipeline persists segments before the speaker timeline exists, so
    this is what a later Speaker Boundary Split has to cut on.
    """

    start: float
    end: float
    text: str
    speaker: int = -1
    words: list[TranscriptWord] = field(default_factory=list)
    tokens: list[TranscriptToken] = field(default_factory=list)


@dataclass
class SpeakerSegment:
    """A speaker timeline entry produced by the Diarization Adapter."""

    start: float
    end: float
    speaker: int
