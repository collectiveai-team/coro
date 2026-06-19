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


# MARK: Transcription Response Model
# Project-owned, API-agnostic response model. The pipeline boundary returns
# ``TranscriptionResult``; the API boundary serialises it (``dataclasses.asdict``)
# into the strict pydantic Boundary Response Schema. Field order mirrors that
# schema so JSON output is byte-identical between batch and streaming paths.
@dataclass
class RawWord:
    """A raw ASR word before segment interpolation, in response shape."""

    word: str
    start: float
    end: float
    score: float


@dataclass
class ResponseSegment:
    """A speaker-attributed, serialisable transcript segment with word timings."""

    start: float
    end: float
    text: str
    speaker: str
    words: list[TranscriptWord] = field(default_factory=list)


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


# MARK: Pipeline Stream Event Types
@dataclass
class TokenBatchEvent:
    """Internal event: a batch of accepted tokens from one ASR window.

    Never emitted to SSE clients; consumed internally by pipelines.
    """

    tokens: list[TranscriptToken]
    type: str = field(default="_tokens", init=False)


@dataclass
class TranscriptDeltaEvent:
    """Public SSE event: incremental transcript text delta."""

    delta: str
    type: str = field(default="transcript.text.delta", init=False)


@dataclass
class TranscriptDoneEvent:
    """Public SSE event: final transcript JSON string."""

    text: str
    type: str = field(default="transcript.text.done", init=False)


# MARK: Stream Event Union
StreamEvent = TokenBatchEvent | TranscriptDeltaEvent | TranscriptDoneEvent
PipelineStreamEvent = TranscriptDeltaEvent | TranscriptDoneEvent
