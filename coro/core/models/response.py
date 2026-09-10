"""Transcription Response Model.

Project-owned, API-agnostic response model and the *single* internal
representation of a transcription: the pipeline boundary returns
``TranscriptionResult`` and every vendor projection reads it directly. It was
formerly copied into a field-for-field pydantic mirror at the API boundary,
which cost more heap than everything else on the JSON path combined and grew
with audio length; the mirror is gone (ADR 0018).

Nothing is lost by dropping that validation step, because a dataclass is closed
to unknown fields by construction — the structural form of ``extra="forbid"``.
Field order is the wire key order for the streamed done frame, which derives it
from these declarations, so reordering a field here reorders published JSON.
Both properties are pinned by ``tests/test_boundary_schemas.py``.
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
    detected_language: str | None = None
    """Language auto-LID resolved for this request, or None when no request
    language was given and no window's detection ever succeeded (the
    fallback language decoded throughout instead -- see
    ``coro/pipelines/windowing.py``'s sticky auto-LID state). Only ever set
    by a backend that exposes ``detect_language``; ``None`` for every other
    backend and for an explicit request language (detection never runs)."""
