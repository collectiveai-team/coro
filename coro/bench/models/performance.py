"""Performance Benchmark aggregation models.

Models for parsing Resource CSVs and aggregating per-rep / per-item / run-level
performance metrics. The pure functions that build them live in
``bench.performance``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedResourceRows:
    """Per-column value series and run-scalar fields parsed from a Resource CSV."""

    series: dict[str, list[float]] = field(default_factory=dict)
    scalars: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerRepSummary:
    """Peak/scalar metrics for one repetition, derived from a Resource CSV."""

    peak_pss_kb: float | None = None
    peak_pss_delta_kb: float | None = None
    baseline_pss_kb: float | str | None = None
    peak_vram_mib: float | None = None
    peak_vram_delta_mib: float | None = None
    baseline_vram_mib: float | str | None = None
    peak_cpu_pct: float | None = None
    peak_gpu_util_pct: float | None = None
    wall_seconds: float | None = None
    audio_seconds: float | None = None
    transcription_throughput: float | None = None
    time_to_first_delta_s: float | None = None
    observed_hardware_profile: str = "cpu-only"


@dataclass
class MetricStats:
    """Median/min/max/mean/stddev across repetitions for one metric."""

    median: float
    min: float
    max: float
    mean: float
    stddev: float


@dataclass
class PerItemAggregation:
    """Across-rep statistics per metric for one workload item."""

    transcription_throughput: MetricStats | None = None
    peak_pss_kb: MetricStats | None = None
    peak_pss_delta_kb: MetricStats | None = None
    peak_cpu_pct: MetricStats | None = None
    peak_vram_mib: MetricStats | None = None
    peak_vram_delta_mib: MetricStats | None = None
    peak_gpu_util_pct: MetricStats | None = None
    time_to_first_delta_s: MetricStats | None = None


@dataclass
class RunTotals:
    """Workload-level totals for a performance run."""

    total_wall_seconds: float
    total_audio_seconds: float
    workload_set_size: int


@dataclass
class PerformanceSummary:
    """Full performance summary written to ``performance/summary.json``."""

    per_rep: list[dict[str, Any]] = field(default_factory=list)
    per_item_aggregation: dict[str, PerItemAggregation] = field(default_factory=dict)
    run_totals: RunTotals | None = None
