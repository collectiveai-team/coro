"""Benchmark report models.

In-memory report model and its row types, consumed by both the stdout and GFM
markdown renderers. The builder and renderers live in ``bench.report``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QualityRow:
    """One row in the quality results table."""

    session_id: str
    duration: float
    cpwer: float | None
    orcwer: float | None
    dicpwer: float | None
    der: float | None
    error: str | None = None


@dataclass
class PerformanceRow:
    """One row in the performance results table."""

    session_id: str
    rep: int
    duration: float
    wall_seconds: float
    throughput: float
    peak_pss_kb: float | None
    peak_pss_delta_kb: float | None
    peak_vram_mib: float | None
    peak_vram_delta_mib: float | None
    peak_gpu_util_pct: float | None
    peak_cpu_pct: float | None
    observed_profile: str
    ttft: float | None = None


@dataclass
class BenchReport:
    """In-memory report model consumed by both renderers."""

    subcommand: str
    timestamp: str
    out_dir: str
    git_sha: str
    total_wall_seconds: float
    stream: bool
    server_config: dict
    workload_set: list[str]
    quality_rows: list[QualityRow] = field(default_factory=list)
    quality_combined: QualityRow | None = None
    normalized_quality_rows: list[QualityRow] = field(default_factory=list)
    normalized_quality_combined: QualityRow | None = None
    quality_footnotes: list[str] = field(default_factory=list)
    performance_rows: list[PerformanceRow] = field(default_factory=list)
    versions: dict = field(default_factory=dict)
    cli_args: list[str] = field(default_factory=list)
