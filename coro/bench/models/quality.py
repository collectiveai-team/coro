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
class NormalizedMetrics:
    """WER metrics after punctuation/whitespace normalization."""

    cpwer: WerStats | None = None
    orcwer: WerStats | None = None
    dicpwer: WerStats | None = None


@dataclass
class ScoreMetrics:
    """Per-item metric block produced by :func:`score_item`."""

    cpwer: WerStats | None = None
    orcwer: WerStats | None = None
    dicpwer: WerStats | None = None
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
    normalized_cpwer: float | None = None
    normalized_orcwer: float | None = None
    normalized_dicpwer: float | None = None


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
