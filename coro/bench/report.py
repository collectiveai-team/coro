"""Benchmark report model, builder, and renderers (stdout + GFM markdown)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from coro.bench.models.report import (
    BenchReport,
    PerformanceRow,
    QualityRow,
    QualitySchemaTable,
    SegmentShapeRow,
)


def build_report(out_dir: Path) -> BenchReport:
    """Read artifacts from out_dir and construct a BenchReport model."""
    manifest = _read_json(out_dir / "manifest.json")

    subcommand = manifest.get("subcommand", "all")
    timestamp = manifest.get("timestamp", "")
    git_sha = manifest.get("git_sha", "unknown")
    cli_args = manifest.get("cli_args", [])
    versions = manifest.get("versions", {})

    server_health = manifest.get("server_health", {})
    startup = server_health.get("startup_selection", {})
    # /health reports provider/model under *_provider keys; accept the older
    # *_backend names too so the report records the run config either way.
    server_config = {
        "asr_backend": startup.get("asr_backend") or startup.get("asr_provider", ""),
        "asr_model": startup.get("asr_model", ""),
        "diar_backend": startup.get("diar_backend") or startup.get("diarization_provider", ""),
        "diar_model": startup.get("diar_model") or startup.get("diarization_model", ""),
        "pipeline": startup.get("pipeline", ""),
        "warmup": manifest.get("warmup", False),
    }

    workload_items = manifest.get("workload_set", [])
    workload_set = [it["item_id"] for it in workload_items]

    stream = "--stream" in cli_args

    (
        quality_rows,
        quality_combined,
        schema_quality_tables,
        quality_footnotes,
    ) = _load_quality(out_dir)
    segment_shape_rows, segment_shape_combined = _load_segment_shape(out_dir)
    performance_rows = _load_performance(out_dir, stream=stream)

    total_wall = _compute_total_wall(performance_rows, quality_rows)

    return BenchReport(
        subcommand=subcommand,
        timestamp=timestamp,
        out_dir=str(out_dir),
        git_sha=git_sha,
        total_wall_seconds=total_wall,
        stream=stream,
        server_config=server_config,
        workload_set=workload_set,
        quality_rows=quality_rows,
        quality_combined=quality_combined,
        schema_quality_tables=schema_quality_tables,
        quality_footnotes=quality_footnotes,
        segment_shape_rows=segment_shape_rows,
        segment_shape_combined=segment_shape_combined,
        performance_rows=performance_rows,
        versions=versions,
        cli_args=cli_args,
    )


# Presentation metadata for the text schemas bench.quality scores under. Keys
# match the summary.json blocks; order is report order.
SCHEMA_TABLE_META: tuple[tuple[str, str, str], ...] = (
    (
        "normalized",
        "Normalized Quality Results",
        "WER metrics after removing punctuation and collapsing whitespace.",
    ),
    (
        "leaderboard",
        "Leaderboard Text Schema Quality Results",
        "WER metrics under the Whisper EnglishTextNormalizer conventions used by the "
        "Open ASR Leaderboard: lowercased, punctuation removed, contractions expanded, "
        "fillers (um/uh/hmm/mm) deleted. Comparable with published ASR numbers.",
    ),
)


def _load_quality(
    out_dir: Path,
) -> tuple[list[QualityRow], QualityRow | None, list[QualitySchemaTable], list[str]]:
    summary_path = out_dir / "quality" / "summary.json"
    if not summary_path.exists():
        return [], None, [], []

    summary = _read_json(summary_path)
    per_item = summary.get("per_item", [])
    combined_data = summary.get("combined", {})

    rows: list[QualityRow] = []
    schema_rows: dict[str, list[QualityRow]] = {key: [] for key, _, _ in SCHEMA_TABLE_META}
    footnotes: list[str] = []

    for item in per_item:
        session_id = item.get("session_id", "")
        duration = float(item.get("audio_seconds", 0.0))
        error_str = item.get("error")

        diar = item.get("diarization") or {}
        if diar.get("degenerate"):
            footnotes.append(
                f"WARNING for {session_id}: degenerate diarization "
                f"({diar.get('hyp_speakers')} hyp speaker(s) vs "
                f"{diar.get('ref_speakers')} ref) — speaker-blind WER (ORC-WER) "
                f"will look good regardless; check DER/cpWER."
            )

        # Diarization-only items (e.g. VoxConverse) carry a valid DER but no WER;
        # a missing cpwer is expected for them and must not be flagged as an error.
        diarization_only = bool(item.get("diarization_only"))
        if not diarization_only and (error_str or item.get("cpwer") is None):
            err_msg = str(error_str) if error_str else "unknown error"
            footnotes.append(f"ERROR for {session_id}: {err_msg}")
            rows.append(
                QualityRow(
                    session_id=session_id,
                    duration=duration,
                    cpwer=None,
                    orcwer=None,
                    dicpwer=None,
                    der=None,
                    error=err_msg,
                )
            )
        else:
            rows.append(
                QualityRow(
                    session_id=session_id,
                    duration=duration,
                    cpwer=_wer_val(item.get("cpwer")),
                    orcwer=_wer_val(item.get("orcwer")),
                    dicpwer=_wer_val(item.get("dicpwer")),
                    der=_wer_val(item.get("der")),
                )
            )
            for key, _, _ in SCHEMA_TABLE_META:
                if item.get(f"{key}_cpwer") is None:
                    continue
                schema_rows[key].append(
                    QualityRow(
                        session_id=session_id,
                        duration=duration,
                        cpwer=_wer_val(item.get(f"{key}_cpwer")),
                        orcwer=_wer_val(item.get(f"{key}_orcwer")),
                        dicpwer=_wer_val(item.get(f"{key}_dicpwer")),
                        der=None,
                    )
                )

    combined: QualityRow | None = None
    if combined_data:
        combined = QualityRow(
            session_id="COMBINED",
            duration=sum(r.duration for r in rows),
            cpwer=_nested_wer(combined_data, "cpwer"),
            orcwer=_nested_wer(combined_data, "orcwer"),
            dicpwer=_nested_wer(combined_data, "dicpwer"),
            der=_nested_der(combined_data),
        )

    return rows, combined, _schema_tables(schema_rows, combined_data), footnotes


def _schema_tables(
    schema_rows: dict[str, list[QualityRow]], combined_data: dict
) -> list[QualitySchemaTable]:
    """Build one table per text schema present in the summary, in report order."""
    tables: list[QualitySchemaTable] = []
    for key, title, note in SCHEMA_TABLE_META:
        rows = schema_rows[key]
        schema_combined_data = combined_data.get(key) or {}
        combined: QualityRow | None = None
        if schema_combined_data:
            combined = QualityRow(
                session_id="COMBINED",
                duration=sum(r.duration for r in rows),
                cpwer=_nested_wer(schema_combined_data, "cpwer"),
                orcwer=_nested_wer(schema_combined_data, "orcwer"),
                dicpwer=_nested_wer(schema_combined_data, "dicpwer"),
                der=None,
            )
        if rows or combined:
            tables.append(
                QualitySchemaTable(key=key, title=title, note=note, rows=rows, combined=combined)
            )
    return tables


def _segment_shape_row(session_id: str, data: Any) -> SegmentShapeRow | None:
    """Build one Segment Shape Counters row, or None when the run has none."""
    if not isinstance(data, dict):
        return None
    return SegmentShapeRow(
        session_id=session_id,
        segment_count=int(data.get("segment_count", 0)),
        median_words_per_segment=_wer_val(data.get("median_words_per_segment")),
        single_word_segment_count=int(data.get("single_word_segment_count", 0)),
    )


def _load_segment_shape(out_dir: Path) -> tuple[list[SegmentShapeRow], SegmentShapeRow | None]:
    """Read Segment Shape Counters from the quality summary.

    Runs recorded before the counters existed simply yield no rows, so the
    table is omitted rather than rendered empty.
    """
    summary_path = out_dir / "quality" / "summary.json"
    if not summary_path.exists():
        return [], None

    summary = _read_json(summary_path)

    rows: list[SegmentShapeRow] = []
    for item in summary.get("per_item", []):
        row = _segment_shape_row(item.get("session_id", ""), item.get("segment_shape"))
        if row is not None:
            rows.append(row)

    return rows, _segment_shape_row("COMBINED", summary.get("segment_shape"))


def _wer_val(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _nested_wer(data: dict, key: str) -> float | None:
    val = data.get(key)
    if val is None:
        return None
    if isinstance(val, dict):
        return _wer_val(val.get("wer"))
    return _wer_val(val)


def _nested_der(data: dict) -> float | None:
    val = data.get("der")
    if val is None:
        return None
    if isinstance(val, dict):
        return _wer_val(val.get("der"))
    return _wer_val(val)


def _load_performance(out_dir: Path, *, stream: bool) -> list[PerformanceRow]:
    summary_path = out_dir / "performance" / "summary.json"
    if not summary_path.exists():
        return []

    summary = _read_json(summary_path)
    per_rep = summary.get("per_rep", [])
    rows: list[PerformanceRow] = []

    for rep_data in per_rep:
        item_id = rep_data.get("item_id", "")
        rep = int(rep_data.get("rep", 1))
        duration = float(rep_data.get("audio_seconds", 0.0))
        wall_seconds = float(rep_data.get("wall_seconds", 0.0))
        throughput = float(rep_data.get("transcription_throughput", 0.0))
        peak_pss_kb = _wer_val(rep_data.get("peak_pss_kb"))
        peak_pss_delta_kb = _wer_val(rep_data.get("peak_pss_delta_kb"))
        peak_vram_mib = _wer_val(rep_data.get("peak_vram_mib"))
        peak_vram_delta_mib = _wer_val(rep_data.get("peak_vram_delta_mib"))
        peak_gpu_util_pct = _wer_val(rep_data.get("peak_gpu_util_pct"))
        peak_cpu_pct = _wer_val(rep_data.get("peak_cpu_pct"))
        observed_profile = rep_data.get("observed_hardware_profile", "cpu-only")
        ttft = _wer_val(rep_data.get("time_to_first_delta_s")) if stream else None

        rows.append(
            PerformanceRow(
                session_id=item_id,
                rep=rep,
                duration=duration,
                wall_seconds=wall_seconds,
                throughput=throughput,
                peak_pss_kb=peak_pss_kb,
                peak_pss_delta_kb=peak_pss_delta_kb,
                peak_vram_mib=peak_vram_mib,
                peak_vram_delta_mib=peak_vram_delta_mib,
                peak_gpu_util_pct=peak_gpu_util_pct,
                peak_cpu_pct=peak_cpu_pct,
                observed_profile=observed_profile,
                ttft=ttft,
            )
        )

    return rows


def _compute_total_wall(
    performance_rows: list[PerformanceRow],
    quality_rows: list[QualityRow],
) -> float:
    if performance_rows:
        return sum(r.wall_seconds for r in performance_rows)
    if quality_rows:
        return sum(r.duration for r in quality_rows)
    return 0.0


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        empty: dict[str, Any] = {}
        return empty


def render_markdown(report: BenchReport) -> str:
    """Return a GFM markdown string for the benchmark report."""
    lines: list[str] = []

    lines.append("# Benchmark Report")
    lines.append("")
    lines.append(f"**Timestamp**: {report.timestamp}")
    lines.append(f"**Output**: {report.out_dir}")
    lines.append(f"**Git SHA**: {report.git_sha}")
    lines.append(f"**Total wall time**: {report.total_wall_seconds:.1f}s")

    sc = report.server_config
    lines.append(f"**ASR**: {sc.get('asr_backend', '')} / {sc.get('asr_model', '')}")
    lines.append(f"**Diarization**: {sc.get('diar_backend', '')} / {sc.get('diar_model', '')}")
    lines.append(f"**Pipeline**: {sc.get('pipeline', '')}")
    lines.append(f"**Warmup**: {sc.get('warmup', False)}")

    if report.subcommand == "all":
        lines.append("")
        lines.append("> Quality scored from rep1; performance averaged across all reps.")

    if report.quality_rows or report.quality_combined:
        lines.append("")
        lines += _quality_table_md(report)

    for table in report.schema_quality_tables:
        lines.append("")
        lines += _schema_quality_table_md(table)

    if report.segment_shape_rows or report.segment_shape_combined:
        lines.append("")
        lines += _segment_shape_table_md(report)

    if report.performance_rows:
        lines.append("")
        lines += _performance_table_md(report)

    lines.append("")
    lines.append("## Run Configuration")
    lines.append("")
    lines.append("**CLI args**: `" + " ".join(report.cli_args) + "`")
    lines.append("")
    lines.append("**Versions**:")
    for pkg, ver in report.versions.items():
        lines.append(f"- {pkg}: {ver}")

    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    out_path = Path(report.out_dir)
    for family in ("responses", "hyp", "ref", "performance", "quality"):
        family_dir = out_path / family
        if family_dir.exists():
            lines.append(f"- `{family}/`")

    return "\n".join(lines) + "\n"


def _quality_table_md(report: BenchReport) -> list[str]:
    lines: list[str] = []
    lines.append("## Quality Results")
    lines.append("")
    lines.append("| session | duration | cpWER | ORC-WER | DI-cpWER | DER |")
    lines.append("|---------|----------|-------|---------|----------|-----|")

    for row in report.quality_rows:
        if row.error is not None:
            lines.append(
                f"| {row.session_id} | {row.duration:.1f} | ERROR | ERROR | ERROR | ERROR |"
            )
        else:
            lines.append(
                f"| {row.session_id} | {row.duration:.1f} "
                f"| {_fmt(row.cpwer)} "
                f"| {_fmt(row.orcwer)} | {_fmt(row.dicpwer)} "
                f"| {_fmt(row.der)} |"
            )

    if report.quality_combined is not None:
        c = report.quality_combined
        lines.append(
            f"| **{c.session_id}** | {c.duration:.1f} "
            f"| {_fmt(c.cpwer)} "
            f"| {_fmt(c.orcwer)} | {_fmt(c.dicpwer)} "
            f"| {_fmt(c.der)} |"
        )

    if report.quality_footnotes:
        lines.append("")
        for note in report.quality_footnotes:
            lines.append(f"[^]: {note}")

    return lines


def _schema_quality_table_md(table: QualitySchemaTable) -> list[str]:
    lines: list[str] = []
    lines.append(f"## {table.title}")
    lines.append("")
    lines.append(table.note)
    lines.append("")
    lines.append("| session | duration | cpWER | ORC-WER | DI-cpWER |")
    lines.append("|---------|----------|-------|---------|----------|")

    for row in table.rows:
        lines.append(
            f"| {row.session_id} | {row.duration:.1f} "
            f"| {_fmt(row.cpwer)} | {_fmt(row.orcwer)} | {_fmt(row.dicpwer)} |"
        )

    if table.combined is not None:
        c = table.combined
        lines.append(
            f"| **{c.session_id}** | {c.duration:.1f} "
            f"| {_fmt(c.cpwer)} | {_fmt(c.orcwer)} | {_fmt(c.dicpwer)} |"
        )

    return lines


_SEGMENT_SHAPE_NOTE = (
    "Unscored counts. cpWER concatenates all words per speaker, so it improves "
    "monotonically as segments are shredded; read these beside it."
)


def _fmt_median(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def _segment_shape_table_md(report: BenchReport) -> list[str]:
    lines: list[str] = []
    lines.append("## Segment Shape")
    lines.append("")
    lines.append(_SEGMENT_SHAPE_NOTE)
    lines.append("")
    lines.append("| session | segments | median words/seg | single-word segs |")
    lines.append("|---------|----------|------------------|------------------|")

    for row in report.segment_shape_rows:
        lines.append(
            f"| {row.session_id} | {row.segment_count} "
            f"| {_fmt_median(row.median_words_per_segment)} "
            f"| {row.single_word_segment_count} |"
        )

    c = report.segment_shape_combined
    if c is not None:
        lines.append(
            f"| **{c.session_id}** | {c.segment_count} "
            f"| {_fmt_median(c.median_words_per_segment)} "
            f"| {c.single_word_segment_count} |"
        )

    return lines


def _performance_table_md(report: BenchReport) -> list[str]:
    lines: list[str] = []
    lines.append("## Performance Results")
    lines.append("")

    has_ttft = report.stream
    if has_ttft:
        hdr = (
            "| session | rep | duration | wall (s) | throughput"
            " | peak PSS | pred PSS | peak VRAM | pred VRAM | peak GPU | peak CPU | TTFT (s) | observed profile |"
        )
        sep = (
            "|---------|-----|----------|----------"
            "|------------|----------|----------|-----------|-----------|----------|----------|----------|-----------------|"
        )
    else:
        hdr = (
            "| session | rep | duration | wall (s) | throughput"
            " | peak PSS | pred PSS | peak VRAM | pred VRAM | peak GPU | peak CPU | observed profile |"
        )
        sep = (
            "|---------|-----|----------|----------"
            "|------------|----------|----------|-----------|-----------|----------|----------|-----------------|"
        )
    lines.append(hdr)
    lines.append(sep)

    for row in report.performance_rows:
        if has_ttft:
            lines.append(
                f"| {row.session_id} | {row.rep} | {row.duration:.1f} "
                f"| {row.wall_seconds:.2f} | {row.throughput:.2f}x "
                f"| {_fmt_kb_as_mb(row.peak_pss_kb)} | {_fmt_kb_as_mb(row.peak_pss_delta_kb)} "
                f"| {_fmt_mib(row.peak_vram_mib)} | {_fmt_mib(row.peak_vram_delta_mib)} "
                f"| {_fmt_pct(row.peak_gpu_util_pct)} | {_fmt_pct(row.peak_cpu_pct)} "
                f"| {_fmt(row.ttft)} "
                f"| {row.observed_profile} |"
            )
        else:
            lines.append(
                f"| {row.session_id} | {row.rep} | {row.duration:.1f} "
                f"| {row.wall_seconds:.2f} | {row.throughput:.2f}x "
                f"| {_fmt_kb_as_mb(row.peak_pss_kb)} | {_fmt_kb_as_mb(row.peak_pss_delta_kb)} "
                f"| {_fmt_mib(row.peak_vram_mib)} | {_fmt_mib(row.peak_vram_delta_mib)} "
                f"| {_fmt_pct(row.peak_gpu_util_pct)} | {_fmt_pct(row.peak_cpu_pct)} "
                f"| {row.observed_profile} |"
            )

    return lines


def _fmt(value: float | None, decimals: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{decimals}f}"


def _fmt_kb_as_mb(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value / 1024.0:.0f} MB"


def _fmt_mib(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.0f} MiB"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}%"


def render_stdout(report: BenchReport) -> None:
    """Render the report to stdout using rich if available, else plain text."""
    try:
        _render_stdout_rich(report)
    except ImportError:
        _render_stdout_plain(report)


def _render_stdout_rich(report: BenchReport) -> None:
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print(Panel(_rich_header(report), title="Benchmark Report"))
    _rich_quality_table(console, report)
    for table in report.schema_quality_tables:
        _rich_schema_quality_table(console, table)
    _rich_segment_shape_table(console, report)
    _rich_performance_table(console, report)


def _rich_segment_shape_table(console: Any, report: BenchReport) -> None:
    if not (report.segment_shape_rows or report.segment_shape_combined):
        return
    from rich.table import Table

    st = Table(title="Segment Shape", show_lines=True)
    for col in ("session", "segments", "median words/seg", "single-word segs"):
        st.add_column(col)
    for row in report.segment_shape_rows:
        st.add_row(
            row.session_id,
            str(row.segment_count),
            _fmt_median(row.median_words_per_segment),
            str(row.single_word_segment_count),
        )
    c = report.segment_shape_combined
    if c is not None:
        st.add_row(
            f"[bold]{c.session_id}[/bold]",
            f"[bold]{c.segment_count}[/bold]",
            f"[bold]{_fmt_median(c.median_words_per_segment)}[/bold]",
            f"[bold]{c.single_word_segment_count}[/bold]",
        )
    console.print(st)
    console.print(f"  * {_SEGMENT_SHAPE_NOTE}")


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

    qt = Table(title="Quality Results", show_lines=True)
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


def _rich_schema_quality_table(console: object, table: QualitySchemaTable) -> None:
    from rich.table import Table

    qt = Table(title=table.title, show_lines=True)
    for col in ("session", "duration", "cpWER", "ORC-WER", "DI-cpWER"):
        qt.add_column(col)
    for row in table.rows:
        qt.add_row(
            row.session_id,
            f"{row.duration:.1f}",
            _fmt(row.cpwer),
            _fmt(row.orcwer),
            _fmt(row.dicpwer),
        )
    if table.combined is not None:
        c = table.combined
        qt.add_row(
            f"[bold]{c.session_id}[/bold]",
            f"{c.duration:.1f}",
            _fmt(c.cpwer),
            _fmt(c.orcwer),
            _fmt(c.dicpwer),
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


def _render_stdout_plain(report: BenchReport) -> None:
    print("# Benchmark Report")
    print(render_markdown(report))
