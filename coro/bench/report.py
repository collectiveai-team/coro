"""Benchmark report builder: reads run artifacts into a ``BenchReport`` model.

The renderers live in ``coro.bench.report_markdown`` (GFM markdown) and
``coro.bench.report_stdout`` (terminal).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from coro.bench.models.report import BenchReport, PerformanceRow, QualityRow


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

    summary = _read_json(summary_path)
    per_item = summary.get("per_item", [])
    combined_data = summary.get("combined", {})

    rows: list[QualityRow] = []
    normalized_rows: list[QualityRow] = []
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
                    wder=_wer_val(item.get("wder")),
                    wder_claimed=_wer_val(item.get("wder_claimed")),
                    abstention_rate=_wer_val(item.get("abstention_rate")),
                )
            )
            if item.get("normalized_cpwer") is not None:
                normalized_rows.append(
                    QualityRow(
                        session_id=session_id,
                        duration=duration,
                        cpwer=_wer_val(item.get("normalized_cpwer")),
                        orcwer=_wer_val(item.get("normalized_orcwer")),
                        dicpwer=_wer_val(item.get("normalized_dicpwer")),
                        der=None,
                        wder=_wer_val(item.get("normalized_wder")),
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
            wder=_nested_field(combined_data, "wder", "wder"),
            wder_claimed=_nested_field(combined_data, "wder", "wder_claimed"),
            abstention_rate=_nested_field(combined_data, "wder", "abstention_rate"),
        )

    normalized_combined: QualityRow | None = None
    normalized_combined_data = combined_data.get("normalized", {})
    if normalized_combined_data:
        normalized_combined = QualityRow(
            session_id="COMBINED",
            duration=sum(r.duration for r in normalized_rows),
            cpwer=_nested_wer(normalized_combined_data, "cpwer"),
            orcwer=_nested_wer(normalized_combined_data, "orcwer"),
            dicpwer=_nested_wer(normalized_combined_data, "dicpwer"),
            der=None,
            wder=_nested_field(normalized_combined_data, "wder", "wder"),
        )

    return rows, combined, normalized_rows, normalized_combined, footnotes


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


def _nested_field(data: dict, block: str, key: str) -> float | None:
    """Read one float out of a nested metric block, tolerating absent blocks."""
    val = data.get(block)
    if not isinstance(val, dict):
        return None
    return _wer_val(val.get(key))


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
    except Exception:  # noqa: BLE001
        empty: dict[str, Any] = {}
        return empty
