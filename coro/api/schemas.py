"""Strict Transcription Response Schema — the provider-agnostic boundary.

The pipeline's own response shape, validated at the API boundary before any
provider projection runs. Every vendor endpoint projects *from* this; none of
them may leak into it, so nothing here is named after a provider.

Provider-shaped models live beside their endpoint: ``coro/api/openai/schemas.py``
and ``coro/api/deepgram/schemas.py``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


# Item Models ---------------------------------------------------------------
class TranscriptWord(BaseModel):
    """Word-level timestamp item in a segment.

    ``start``/``end`` are the ASR backend's real per-word values, and ``speaker``
    is decided for this word alone — the normative per-word speaker truth.
    ``overlap`` flags a word whose span contains concurrently active speakers, so
    the single-label collapse is visible rather than silent (see ADR 0014).

    ``score`` is the backend's own probability, and ``None`` when the backend
    expresses none. Unmeasured is never stubbed (ADR 0015 rule 3).
    """

    model_config = ConfigDict(extra="forbid")

    word: str
    start: float
    end: float
    score: float | None
    speaker: str
    overlap: bool = False


class ResponseSegment(BaseModel):
    """Segment item in the Strict Transcription Response Schema.

    Segments are sentence-shaped, so one may span a speaker turn; ``speaker`` is
    the duration-weighted majority of ``words`` and is ``"-1"`` only when every
    word is. ``overlap`` is true when any word in the segment falls in overlapped
    speech (see ADR 0014).
    """

    model_config = ConfigDict(extra="forbid")

    start: float
    end: float
    text: str
    speaker: str
    words: list[TranscriptWord]
    overlap: bool = False


class TranscriptItem(BaseModel):
    """Transcript convenience item."""

    model_config = ConfigDict(extra="forbid")

    start: float
    end: float
    text: str


class DiarizationItem(BaseModel):
    """Diarization convenience item."""

    model_config = ConfigDict(extra="forbid")

    start: float
    end: float
    speaker: str


class RawWord(BaseModel):
    """Raw ASR word item as the backend emitted it.

    ``score`` is ``None`` when the backend expresses no probability.
    """

    model_config = ConfigDict(extra="forbid")

    word: str
    start: float
    end: float
    score: float | None


# Response Model ------------------------------------------------------------
class TranscriptionResponse(BaseModel):
    """Strict internal transcription response schema exposed by the pipeline."""

    model_config = ConfigDict(extra="forbid")

    segments: list[ResponseSegment]
    word_segments: list[TranscriptWord]
    transcript: list[TranscriptItem]
    diarization: list[DiarizationItem]
    raw_words: list[RawWord]
