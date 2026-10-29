"""Terminal renderer for the benchmark report, drawn with Rich."""

from __future__ import annotations

from coro.bench.models.report import BenchReport
from coro.bench.report_markdown import (
    NORMALIZED_LANE_CAPTION,
    NORMALIZED_LANE_LABEL,
    RAW_LANE_CAPTION,
    RAW_LANE_LABEL,
    _fmt,
    _fmt_kb_as_mb,
    _fmt_mib,
    _fmt_pct,
    _wder_rows,
)


def render_stdout(report: BenchReport) -> None:
    """Render the report to stdout with Rich panels and tables."""
    _render_stdout_rich(report)


def _render_stdout_rich(report: BenchReport) -> None:
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print(Panel(_rich_header(report), title="Benchmark Report"))
    _rich_quality_table(console, report)
    _rich_wder_table(console, report)
    _rich_normalized_quality_table(console, report)
    _rich_performance_table(console, report)


def _rich_header(report: BenchReport) -> str:
    sc = report.server_config
    lines = [
        f"Timestamp : {report.timestamp}",
        f"Output    : {report.out_dir}",
        f"Git SHA   : {report.git_sha}",
        f"Wall time : {report.total_wall_seconds:.1f}s",
        f"ASR       : {sc.get('asr_backend', '')} / {sc.get('asr_model', '')}",
        f"Diarization: {sc.get('diar_backend', '')} / {sc.get('diar_model', '')}",
        f"Pipeline  : {sc.get('pipeline', '')}",
        f"Warmup    : {sc.get('warmup', False)}",
    ]
    if report.subcommand == "all":
        lines.append("Note: Quality scored from rep1; performance averaged across all reps.")
    return "\n".join(lines)


def _rich_quality_table(console: object, report: BenchReport) -> None:
    if not (report.quality_rows or report.quality_combined):
        return
    from rich.table import Table

    qt = Table(
        title=f"Quality Results — {RAW_LANE_LABEL}",
        caption=RAW_LANE_CAPTION,
        show_lines=True,
    )
    for col in ("session", "duration", "cpWER", "ORC-WER", "DI-cpWER", "DER"):
        qt.add_column(col)
    for row in report.quality_rows:
        if row.error is not None:
            qt.add_row(
                row.session_id,
                f"{row.duration:.1f}",
                "ERROR",
                "ERROR",
                "ERROR",
                "ERROR",
            )
        else:
            qt.add_row(
                row.session_id,
                f"{row.duration:.1f}",
                _fmt(row.cpwer),
                _fmt(row.orcwer),
                _fmt(row.dicpwer),
                _fmt(row.der),
            )
    if report.quality_combined is not None:
        c = report.quality_combined
        qt.add_row(
            f"[bold]{c.session_id}[/bold]",
            f"{c.duration:.1f}",
            _fmt(c.cpwer),
            _fmt(c.orcwer),
            _fmt(c.dicpwer),
            _fmt(c.der),
        )
    console.print(qt)  # type: ignore[attr-defined]
    for note in report.quality_footnotes:
        console.print(f"  * {note}")  # type: ignore[attr-defined]


def _rich_wder_table(console: object, report: BenchReport) -> None:
    rows = _wder_rows(report)
    if not rows:
        return
    from rich.table import Table

    wt = Table(title=f"Speaker Attribution (WDER) — {RAW_LANE_LABEL}", show_lines=True)
    for col in ("session", "WDER", "WDER-claimed", "abstention"):
        wt.add_column(col)
    for row in rows:
        label = row.session_id
        if row is report.quality_combined:
            label = f"[bold]{label}[/bold]"
        wt.add_row(
            label,
            _fmt(row.wder),
            _fmt(row.wder_claimed),
            _fmt(row.abstention_rate),
        )
    console.print(wt)  # type: ignore[attr-defined]


def _rich_normalized_quality_table(console: object, report: BenchReport) -> None:
    if not (report.normalized_quality_rows or report.normalized_quality_combined):
        return
    from rich.table import Table

    qt = Table(
        title=f"Quality Results — {NORMALIZED_LANE_LABEL}",
        caption=NORMALIZED_LANE_CAPTION,
        show_lines=True,
    )
    for col in ("session", "duration", "cpWER", "ORC-WER", "DI-cpWER", "WDER"):
        qt.add_column(col)
    for row in report.normalized_quality_rows:
        qt.add_row(
            row.session_id,
            f"{row.duration:.1f}",
            _fmt(row.cpwer),
            _fmt(row.orcwer),
            _fmt(row.dicpwer),
            _fmt(row.wder),
        )
    if report.normalized_quality_combined is not None:
        c = report.normalized_quality_combined
        qt.add_row(
            f"[bold]{c.session_id}[/bold]",
            f"{c.duration:.1f}",
            _fmt(c.cpwer),
            _fmt(c.orcwer),
            _fmt(c.dicpwer),
            _fmt(c.wder),
        )
    console.print(qt)  # type: ignore[attr-defined]


def _rich_performance_table(console: object, report: BenchReport) -> None:
    if not report.performance_rows:
        return
    from rich.table import Table

    has_ttft = report.stream
    pt = Table(title="Performance Results", show_lines=True)
    cols = [
        "session",
        "rep",
        "duration",
        "wall (s)",
        "throughput",
        "peak PSS",
        "pred PSS",
        "peak VRAM",
        "pred VRAM",
        "peak GPU",
        "peak CPU",
    ]
    if has_ttft:
        cols.append("TTFT (s)")
    cols.append("observed profile")
    for col in cols:
        pt.add_column(col)
    for row in report.performance_rows:
        cells = [
            row.session_id,
            str(row.rep),
            f"{row.duration:.1f}",
            f"{row.wall_seconds:.2f}",
            f"{row.throughput:.2f}x",
            _fmt_kb_as_mb(row.peak_pss_kb),
            _fmt_kb_as_mb(row.peak_pss_delta_kb),
            _fmt_mib(row.peak_vram_mib),
            _fmt_mib(row.peak_vram_delta_mib),
            _fmt_pct(row.peak_gpu_util_pct),
            _fmt_pct(row.peak_cpu_pct),
        ]
        if has_ttft:
            cells.append(_fmt(row.ttft))
        cells.append(row.observed_profile)
        pt.add_row(*cells)
    console.print(pt)  # type: ignore[attr-defined]
