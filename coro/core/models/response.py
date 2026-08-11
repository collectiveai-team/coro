"""Transcription Response Model.

Project-owned, API-agnostic response model. The pipeline boundary returns
``TranscriptionResult``; the API boundary serialises it (``dataclasses.asdict``)
into the strict pydantic Boundary Response Schema. Field order mirrors that
schema so JSON output is byte-identical between batch and streaming paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from coro.core.models.transcript import TranscriptWord


@dataclass
class RawWord:
    """A raw ASR word as the backend emitted it, in response shape.

    ``score`` is ``None`` when the backend expresses no probability; it is never
    stubbed (ADR 0015 rule 3).
    """

    word: str
    start: float
    end: float
    score: float | None


@dataclass
class ResponseSegment:
    """A speaker-attributed, serialisable transcript segment with word timings.

    Boundaries are sentence-shaped, so a segment may span a speaker turn and its
    ``speaker`` is the duration-weighted *majority* of ``words`` — a summary, not
    a homogeneity guarantee. ``words`` carries the per-word truth. ``overlap`` is
    set when any of its words falls inside concurrently active speaker timeline
    entries. See ADR 0014.
    """

    start: float
    end: float
    text: str
    speaker: str
    words: list[TranscriptWord] = field(default_factory=list)
    overlap: bool = False


@dataclass
class TranscriptItem:
    """A transcript convenience entry (segment text with timing)."""

    start: float
    end: float
    text: str


@dataclass
class DiarizationItem:
    """A diarization convenience entry (segment speaker with timing)."""

    start: float
    end: float
    speaker: str


@dataclass
class TranscriptionResult:
    """The enriched transcription response produced at the pipeline boundary."""

    segments: list[ResponseSegment] = field(default_factory=list)
    word_segments: list[TranscriptWord] = field(default_factory=list)
    transcript: list[TranscriptItem] = field(default_factory=list)
    diarization: list[DiarizationItem] = field(default_factory=list)
    raw_words: list[RawWord] = field(default_factory=list)
