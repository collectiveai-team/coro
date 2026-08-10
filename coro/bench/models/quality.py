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
class WderStats:
    """Word Diarization Error Rate breakdown (Shafey, Soltau & Shafran 2019).

    ``WDER = (S_IS + C_IS) / (S + C)`` — speaker errors over the ASR words that
    actually exist in both transcripts. Insertions and deletions are excluded
    from the denominator, so the metric is not diluted by the ASR error floor
    and is blind to segmentation.

    Counts are additive across sessions, so a workload-level value is obtained
    by summing them rather than by averaging rates.
    """

    wder: float | None
    wder_claimed: float | None
    abstention_rate: float | None
    scored: int
    speaker_errors: int
    claimed: int
    claimed_speaker_errors: int
    abstentions: int
    correct: int
    substitutions: int


@dataclass
class DiarizationSanity:
    """Degenerate-diarization check for one item."""

    ref_speakers: int
    hyp_speakers: int
    degenerate: bool


@dataclass
class NormalizedMetrics:
    """WER metrics after punctuation/whitespace normalization."""

    cpwer: WerStats | None = None
    orcwer: WerStats | None = None
    dicpwer: WerStats | None = None
    wder: WderStats | None = None


@dataclass
class ScoreMetrics:
    """Per-item metric block produced by :func:`score_item`."""

    cpwer: WerStats | None = None
    orcwer: WerStats | None = None
    dicpwer: WerStats | None = None
    wder: WderStats | None = None
    normalized: NormalizedMetrics | None = None
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


@dataclass
class CombinedMetrics:
    """Workload-level combined metrics across all succeeded items."""

    cpwer: WerStats | None = None
    orcwer: WerStats | None = None
    dicpwer: WerStats | None = None
    wder: WderStats | None = None
    normalized: NormalizedMetrics | None = None
    der: DerStats | None = None


@dataclass
class PerItemEntry:
    """Flattened per-item summary row (WER values, not full breakdowns)."""

    session_id: str = ""
    audio_seconds: float | None = None
    diarization_only: bool | None = None
    diarization: DiarizationSanity | None = None
    cpwer: float | None = None
    orcwer: float | None = None
    dicpwer: float | None = None
    der: float | None = None
    wder: float | None = None
    wder_claimed: float | None = None
    abstention_rate: float | None = None
    normalized_cpwer: float | None = None
    normalized_orcwer: float | None = None
    normalized_dicpwer: float | None = None
    normalized_wder: float | None = None


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
