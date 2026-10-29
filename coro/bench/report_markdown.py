"""GFM markdown renderer for the benchmark report.

Quality figures are always rendered as two separately labelled tables, one per
metric lane, so no figure is readable without knowing which protocol produced
it. See ADR 0011.
"""

from __future__ import annotations

from pathlib import Path

from coro.bench.models.report import BenchReport, QualityRow

RAW_LANE_LABEL = "Raw Metric Lane"
"""Heading suffix for the un-normalized quality table."""

NORMALIZED_LANE_LABEL = "Normalized Metric Lane"
"""Heading suffix for the normalized quality table."""

RAW_LANE_CAPTION = (
    "Reference STM and Hypothesis STM text scored verbatim — punctuation, "
    "casing and diacritics all count. Catches regressions the normalizer hides; "
    "not comparable to published WER. DER is lane-independent and reported here only."
)

NORMALIZED_LANE_CAPTION = (
    "Scored after applying the Basic Text Normalizer identically to the Reference "
    "STM and the Hypothesis STM: bracketed and parenthesised content removed, "
    "lowercased, symbols and punctuation mapped to spaces, whitespace collapsed, "
    "diacritics preserved. This is the lane comparable to published WER."
)


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

    if _has_wder(report):
        lines.append("")
        lines += _wder_table_md(report)

    if report.normalized_quality_rows or report.normalized_quality_combined:
        lines.append("")
        lines += _normalized_quality_table_md(report)

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
    lines.append(f"## Quality Results — {RAW_LANE_LABEL}")
    lines.append("")
    lines.append(RAW_LANE_CAPTION)
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


def _wder_rows(report: BenchReport) -> list[QualityRow]:
    """Quality rows that carry a WDER value, combined row appended last."""
    rows = [r for r in report.quality_rows if r.wder is not None]
    combined = report.quality_combined
    if combined is not None and combined.wder is not None:
        rows.append(combined)
    return rows


def _has_wder(report: BenchReport) -> bool:
    return bool(_wder_rows(report))


def _wder_table_md(report: BenchReport) -> list[str]:
    lines: list[str] = []
    lines.append(f"## Speaker Attribution (WDER) — {RAW_LANE_LABEL}")
    lines.append("")
    lines.append(
        "Word Diarization Error Rate — speaker errors over the words that exist "
        "in both transcripts (correct + substituted). Insertions and deletions "
        "are excluded, so this is blind to segmentation and undiluted by the ASR "
        "error floor. `WDER-claimed` is precision where a real speaker was "
        "committed to; `abstention` is the share of scored words left unknown. "
        "`WDER = WDER-claimed x (1 - abstention) + abstention`. Normalization "
        "changes which words are scored, so the Normalized Metric Lane's WDER is "
        "reported in that lane's own table."
    )
    lines.append("")
    lines.append("| session | WDER | WDER-claimed | abstention |")
    lines.append("|---------|------|--------------|------------|")

    for row in _wder_rows(report):
        label = row.session_id
        if row is report.quality_combined:
            label = f"**{label}**"
        lines.append(
            f"| {label} | {_fmt(row.wder)} "
            f"| {_fmt(row.wder_claimed)} | {_fmt(row.abstention_rate)} |"
        )

    return lines


def _normalized_quality_table_md(report: BenchReport) -> list[str]:
    lines: list[str] = []
    lines.append(f"## Quality Results — {NORMALIZED_LANE_LABEL}")
    lines.append("")
    lines.append(NORMALIZED_LANE_CAPTION)
    lines.append("")
    lines.append("| session | duration | cpWER | ORC-WER | DI-cpWER | WDER |")
    lines.append("|---------|----------|-------|---------|----------|------|")

    for row in report.normalized_quality_rows:
        lines.append(
            f"| {row.session_id} | {row.duration:.1f} "
            f"| {_fmt(row.cpwer)} | {_fmt(row.orcwer)} | {_fmt(row.dicpwer)} "
            f"| {_fmt(row.wder)} |"
        )

    if report.normalized_quality_combined is not None:
        c = report.normalized_quality_combined
        lines.append(
            f"| **{c.session_id}** | {c.duration:.1f} "
            f"| {_fmt(c.cpwer)} | {_fmt(c.orcwer)} | {_fmt(c.dicpwer)} "
            f"| {_fmt(c.wder)} |"
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
