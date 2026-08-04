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
class QualitySchemaTable:
    """One text schema's quality table, rendered beside the raw-text one.

    A WER only means something next to the text conventions behind it, so each
    schema gets its own titled table carrying the same columns.
    """

    key: str
    title: str
    note: str
    rows: list[QualityRow] = field(default_factory=list)
    combined: QualityRow | None = None


@dataclass
class SegmentShapeRow:
    """One row in the Segment Shape Counters table.

    Unscored transcript-shape counts. No WER penalises fragmentation, so these
    are the only place a segment explosion is visible in the report.
    """

    session_id: str
    segment_count: int
    median_words_per_segment: float | None
    single_word_segment_count: int


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
    # One entry per text schema the metrics were also scored under, in report
    # order. A new schema adds an entry, never another pair of fields.
    schema_quality_tables: list[QualitySchemaTable] = field(default_factory=list)
    quality_footnotes: list[str] = field(default_factory=list)
    segment_shape_rows: list[SegmentShapeRow] = field(default_factory=list)
    segment_shape_combined: SegmentShapeRow | None = None
    performance_rows: list[PerformanceRow] = field(default_factory=list)
    versions: dict = field(default_factory=dict)
    cli_args: list[str] = field(default_factory=list)
