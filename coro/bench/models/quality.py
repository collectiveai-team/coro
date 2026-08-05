"""Quality Benchmark scoring models.

Models for the MeetEval Metric Set, per-item score results, and the
workload-level quality summary. Scoring logic that builds them lives in
``bench.quality``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WerStats:
    """Word-error-rate breakdown for one metric."""

    wer: float
    errors: int
    length: int
    insertions: int
    deletions: int
    substitutions: int


@dataclass
class DerStats:
    """Diarization-error-rate breakdown."""

    der: float
    false_alarm: float
    missed_detection: float
    speaker_error: float
    total_speech: float


@dataclass
class DiarizationSanity:
    """Degenerate-diarization check for one item."""

    ref_speakers: int
    hyp_speakers: int
    degenerate: bool


@dataclass
class SegmentShapeCounters:
    """Unscored transcript-shape counts reported beside the MeetEval Metric Set.

    No WER penalises fragmentation — cpWER concatenates all words per speaker,
    so it improves monotonically as segments are shredded. These counters make a
    cpWER win bought with a segment explosion visible in the same report.
    ``median_words_per_segment`` is None when there are no segments.
    """

    segment_count: int = 0
    median_words_per_segment: float | None = None
    single_word_segment_count: int = 0


@dataclass
class SchemaMetrics:
    """WER metrics computed under one text schema.

    Used for every schema in ``bench.text.TEXT_SCHEMAS``, so the block shape is
    identical whichever normalization produced it.
    """

    cpwer: WerStats | None = None
    orcwer: WerStats | None = None
    dicpwer: WerStats | None = None


@dataclass
class ScoreMetrics:
    """Per-item metric block produced by :func:`score_item`.

    ``unpunctuated`` strips punctuation only; ``whisper_english`` applies the
    Leaderboard Text Schema, under which published ASR numbers are reported.
    """

    cpwer: WerStats | None = None
    orcwer: WerStats | None = None
    dicpwer: WerStats | None = None
    unpunctuated: SchemaMetrics | None = None
    whisper_english: SchemaMetrics | None = None
    der: DerStats | None = None


@dataclass
class ScoreError:
    """Captured exception info when scoring an item fails."""

    type: str
    message: str


@dataclass
class ScoreResult:
    """Result of scoring one hypothesis against its reference."""

    session_id: str = ""
    audio_seconds: float = 0.0
    metrics: ScoreMetrics | None = None
    diarization_only: bool = False
    diarization: DiarizationSanity | None = None
    error: ScoreError | None = None
    # Raw meeteval result objects, keyed by metric, retained for cross-item
    # combination. Not JSON-serialisable and never written to artifacts.
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
    # Words per segment in the Hypothesis STM, retained so the workload-level
    # Segment Shape Counters can pool every item's segments before reducing.
    # combine_error_rates takes only meeteval objects, so this cannot ride on
    # ``raw``. Never written to artifacts; the reduced counters are.
    segment_word_counts: list[int] = field(default_factory=list, repr=False)


@dataclass
class QualityItemArtifact:
    """The per-item ``quality/<item_id>.json`` payload.

    Segment Shape Counters sit beside the MeetEval Metric Set rather than
    inside it: they are unscored counts, and they are present even when WER
    scoring was skipped for a diarization-only reference.
    """

    session_id: str
    audio_seconds: float
    metrics: ScoreMetrics | None = None
    segment_shape: SegmentShapeCounters | None = None
    diarization: DiarizationSanity | None = None
    error: ScoreError | None = None


@dataclass
class CombinedMetrics:
    """Workload-level combined metrics across all succeeded items."""

    cpwer: WerStats | None = None
    orcwer: WerStats | None = None
    dicpwer: WerStats | None = None
    unpunctuated: SchemaMetrics | None = None
    whisper_english: SchemaMetrics | None = None
    der: DerStats | None = None


@dataclass
class PerItemEntry:
    """Flattened per-item summary row (WER values, not full breakdowns)."""

    session_id: str = ""
    audio_seconds: float | None = None
    diarization_only: bool | None = None
    diarization: DiarizationSanity | None = None
    segment_shape: SegmentShapeCounters | None = None
    cpwer: float | None = None
    orcwer: float | None = None
    dicpwer: float | None = None
    der: float | None = None
    unpunctuated_cpwer: float | None = None
    unpunctuated_orcwer: float | None = None
    unpunctuated_dicpwer: float | None = None
    whisper_english_cpwer: float | None = None
    whisper_english_orcwer: float | None = None
    whisper_english_dicpwer: float | None = None


@dataclass
class QualitySummary:
    """Workload-level quality summary written to ``quality/summary.json``."""

    workload_set: list[str] = field(default_factory=list)
    n_succeeded: int = 0
    n_failed: int = 0
    n_degenerate_diarization: int = 0
    combined: CombinedMetrics | None = None
    per_item: list[PerItemEntry] = field(default_factory=list)
    n_skipped: int = 0
    # Pooled over every segment in the workload set, matching how meeteval
    # combines raw error counts — not a median of per-item medians.
    segment_shape: SegmentShapeCounters | None = None
