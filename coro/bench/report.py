"""Benchmark report builder: reads run artifacts into a ``BenchReport`` model.

The renderers live in ``coro.bench.report_markdown`` (GFM markdown) and
``coro.bench.report_stdout`` (terminal).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from coro.bench.models.artifacts import (
    DiarizationJson,
    ManifestJson,
    PerformanceSummaryJson,
    PerItemJson,
    QualitySummaryJson,
)
from coro.bench.models.report import BenchReport, PerformanceRow, QualityRow


def build_report(out_dir: Path) -> BenchReport:
    """Read artifacts from out_dir and construct a BenchReport model."""
    manifest = _read_json(out_dir / "manifest.json", ManifestJson())

    subcommand = manifest.get("subcommand", "all")
    timestamp = manifest.get("timestamp", "")
    git_sha = manifest.get("git_sha", "unknown")
    cli_args = manifest.get("cli_args", [])
    versions = manifest.get("versions", {})

    server_health = manifest.get("server_health", {})
    startup = server_health.get("startup_selection", {})
    # /health reports provider/model under *_provider keys; accept the older
    # *_backend names too so the report records the run config either way.
    server_config: dict[str, str | bool] = {
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
        normalized_quality_rows,
        normalized_quality_combined,
        quality_footnotes,
    ) = _load_quality(out_dir)
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
        normalized_quality_rows=normalized_quality_rows,
        normalized_quality_combined=normalized_quality_combined,
        quality_footnotes=quality_footnotes,
        performance_rows=performance_rows,
        versions=versions,
        cli_args=cli_args,
    )


def _load_quality(
    out_dir: Path,
) -> tuple[list[QualityRow], QualityRow | None, list[QualityRow], QualityRow | None, list[str]]:
    summary_path = out_dir / "quality" / "summary.json"
    if not summary_path.exists():
        return [], None, [], None, []

    summary = _read_json(summary_path, QualitySummaryJson())
    combined_data = summary.get("combined") or {}

    rows: list[QualityRow] = []
    normalized_rows: list[QualityRow] = []
    footnotes: list[str] = []

    for item in summary.get("per_item", []):
        footnotes.extend(_item_footnotes(item))
        rows.append(_raw_row(item))
        normalized = _normalized_row(item)
        if normalized is not None:
            normalized_rows.append(normalized)

    combined = _combined_row(combined_data, rows) if combined_data else None
    normalized_combined = _normalized_combined_row(
        _block(combined_data, "normalized"), normalized_rows
    )
    return rows, combined, normalized_rows, normalized_combined, footnotes


def _is_error_item(item: PerItemJson) -> bool:
    """Diarization-only items (e.g. VoxConverse) carry a valid DER but no WER.

    A missing cpwer is expected for them and must not be flagged as an error.
    """
    if item.get("diarization_only"):
        return False
    return bool(item.get("error")) or item.get("cpwer") is None


def _error_message(item: PerItemJson) -> str:
    error = item.get("error")
    return str(error) if error else "unknown error"


def _item_footnotes(item: PerItemJson) -> list[str]:
    session_id = item.get("session_id", "")
    footnotes: list[str] = []
    diar = item.get("diarization") or DiarizationJson()
    if diar.get("degenerate"):
        footnotes.append(
            f"WARNING for {session_id}: degenerate diarization "
            f"({diar.get('hyp_speakers')} hyp speaker(s) vs "
            f"{diar.get('ref_speakers')} ref) — speaker-blind WER (ORC-WER) "
            f"will look good regardless; check DER/cpWER."
        )
    if _is_error_item(item):
        footnotes.append(f"ERROR for {session_id}: {_error_message(item)}")
    return footnotes


def _raw_row(item: PerItemJson) -> QualityRow:
    session_id = item.get("session_id", "")
    duration = float(item.get("audio_seconds", 0.0))
    if _is_error_item(item):
        return QualityRow(
            session_id=session_id,
            duration=duration,
            cpwer=None,
            orcwer=None,
            dicpwer=None,
            der=None,
            error=_error_message(item),
        )
    return QualityRow(
        session_id=session_id,
        duration=duration,
        cpwer=_wer_val(item.get("cpwer")),
        orcwer=_wer_val(item.get("orcwer")),
        dicpwer=_wer_val(item.get("dicpwer")),
        der=_wer_val(item.get("der")),
        wder=_wer_val(item.get("wder")),
        wder_claimed=_wer_val(item.get("wder_claimed")),
        abstention_rate=_wer_val(item.get("abstention_rate")),
    )


def _normalized_row(item: PerItemJson) -> QualityRow | None:
    if _is_error_item(item) or item.get("normalized_cpwer") is None:
        return None
    return QualityRow(
        session_id=item.get("session_id", ""),
        duration=float(item.get("audio_seconds", 0.0)),
        cpwer=_wer_val(item.get("normalized_cpwer")),
        orcwer=_wer_val(item.get("normalized_orcwer")),
        dicpwer=_wer_val(item.get("normalized_dicpwer")),
        der=None,
        wder=_wer_val(item.get("normalized_wder")),
    )


def _combined_row(combined_data: Mapping[str, object], rows: list[QualityRow]) -> QualityRow:
    return QualityRow(
        session_id="COMBINED",
        duration=sum(r.duration for r in rows),
        cpwer=_nested_wer(combined_data, "cpwer"),
        orcwer=_nested_wer(combined_data, "orcwer"),
        dicpwer=_nested_wer(combined_data, "dicpwer"),
        der=_nested_der(combined_data),
        wder=_nested_field(combined_data, "wder", "wder"),
        wder_claimed=_nested_field(combined_data, "wder", "wder_claimed"),
        abstention_rate=_nested_field(combined_data, "wder", "abstention_rate"),
    )


def _normalized_combined_row(
    normalized_data: Mapping[str, object], normalized_rows: list[QualityRow]
) -> QualityRow | None:
    if not normalized_data:
        return None
    return QualityRow(
        session_id="COMBINED",
        duration=sum(r.duration for r in normalized_rows),
        cpwer=_nested_wer(normalized_data, "cpwer"),
        orcwer=_nested_wer(normalized_data, "orcwer"),
        dicpwer=_nested_wer(normalized_data, "dicpwer"),
        der=None,
        wder=_nested_field(normalized_data, "wder", "wder"),
    )


def _block(data: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Return the nested metric block under ``key``, or an empty one."""
    val = data.get(key)
    return val if isinstance(val, dict) else {}


def _wer_val(v: object) -> float | None:
    if not isinstance(v, (int, float, str)):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _nested_wer(data: Mapping[str, object], key: str) -> float | None:
    val = data.get(key)
    if isinstance(val, dict):
        return _wer_val(_block(data, key).get("wer"))
    return _wer_val(val)


def _nested_field(data: Mapping[str, object], block: str, key: str) -> float | None:
    """Read one float out of a nested metric block, tolerating absent blocks."""
    return _wer_val(_block(data, block).get(key))


def _nested_der(data: Mapping[str, object]) -> float | None:
    val = data.get("der")
    if isinstance(val, dict):
        return _wer_val(_block(data, "der").get("der"))
    return _wer_val(val)


def _load_performance(out_dir: Path, *, stream: bool) -> list[PerformanceRow]:
    summary_path = out_dir / "performance" / "summary.json"
    if not summary_path.exists():
        return []

    summary = _read_json(summary_path, PerformanceSummaryJson())
    per_rep = summary.get("per_rep", [])
    rows: list[PerformanceRow] = []

    for rep_data in per_rep:
        item_id = rep_data.get("item_id", "")
        rep = rep_data.get("rep", 1)
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


def _read_json[T](path: Path, empty: T) -> T:
    """Parse a JSON artifact, or return ``empty`` when it is unreadable or malformed."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return empty
