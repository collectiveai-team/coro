"""Pure functions over Resource CSVs for performance aggregation."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any


def _to_float(value: str | None) -> float | None:
    """Parse ``value`` as a float, returning ``None`` for empty/invalid input."""
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_resource_rows(csv_path: Path) -> dict[str, Any]:
    """Read a Resource CSV into per-column value lists and run-scalar fields."""
    series: dict[str, list[float]] = {
        "pss_kb": [],
        "cpu_pct": [],
        "server_vram_mib": [],
        "gpu_util_pct": [],
    }
    scalars: dict[str, Any] = {"observed_hardware_profile": ""}
    scalar_keys = (
        "wall_seconds",
        "transcription_throughput",
        "audio_seconds",
        "time_to_first_delta_s",
        "baseline_pss_kb",
        "baseline_vram_mib",
    )
    have_scalars = False

    with csv_path.open() as f:
        for row in csv.DictReader(f):
            for column, values in series.items():
                parsed = _to_float(row.get(column))
                if parsed is not None:
                    values.append(parsed)
            if not have_scalars:
                have_scalars = True
                for key in scalar_keys:
                    scalars[key] = _to_float(row.get(key))
                scalars["observed_hardware_profile"] = row.get("observed_hardware_profile", "")

    return {"series": series, "scalars": scalars}


def _peak_with_baseline(
    result: dict[str, Any],
    values: list[float],
    baseline: float | None,
    peak_key: str,
    baseline_key: str,
    delta_key: str,
) -> None:
    """Record the peak of ``values`` plus baseline-corrected delta when present."""
    if not values:
        return
    peak = max(values)
    result[peak_key] = peak
    if baseline is not None:
        result[baseline_key] = baseline
        result[delta_key] = max(0.0, peak - baseline)


def compute_per_rep_summary(csv_path: Path) -> dict[str, Any]:
    parsed = _parse_resource_rows(csv_path)
    series = parsed["series"]
    scalars = parsed["scalars"]

    result: dict[str, Any] = {}
    _peak_with_baseline(
        result,
        series["pss_kb"],
        scalars.get("baseline_pss_kb"),
        "peak_pss_kb",
        "baseline_pss_kb",
        "peak_pss_delta_kb",
    )
    _peak_with_baseline(
        result,
        series["server_vram_mib"],
        scalars.get("baseline_vram_mib"),
        "peak_vram_mib",
        "baseline_vram_mib",
        "peak_vram_delta_mib",
    )
    if series["cpu_pct"]:
        result["peak_cpu_pct"] = max(series["cpu_pct"])
    if series["gpu_util_pct"]:
        result["peak_gpu_util_pct"] = max(series["gpu_util_pct"])
    for key in (
        "wall_seconds",
        "transcription_throughput",
        "audio_seconds",
        "time_to_first_delta_s",
    ):
        if scalars.get(key) is not None:
            result[key] = scalars[key]
    result["observed_hardware_profile"] = scalars["observed_hardware_profile"] or "cpu-only"
    return result


def aggregate_across_reps(per_rep_summaries: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metrics = [
        "transcription_throughput",
        "peak_pss_kb",
        "peak_pss_delta_kb",
        "peak_cpu_pct",
        "peak_vram_mib",
        "peak_vram_delta_mib",
        "peak_gpu_util_pct",
        "time_to_first_delta_s",
    ]
    result: dict[str, dict[str, float]] = {}

    for metric in metrics:
        values = [s[metric] for s in per_rep_summaries if metric in s and s[metric] is not None]
        if not values:
            continue
        values = [float(v) for v in values]
        entry: dict[str, float] = {}
        entry["median"] = float(statistics.median(values))
        entry["min"] = float(min(values))
        entry["max"] = float(max(values))
        entry["mean"] = float(statistics.mean(values))
        if len(values) >= 2:
            entry["stddev"] = float(statistics.stdev(values))
        else:
            entry["stddev"] = 0.0
        result[metric] = entry

    return result


def write_performance_summary(
    out_dir: Path,
    per_item_reps: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    per_rep_rows: list[dict[str, Any]] = []
    per_item_agg: dict[str, Any] = {}
    total_wall = 0.0
    total_audio = 0.0

    for item_id, rep_summaries in per_item_reps.items():
        for i, rep_summary in enumerate(rep_summaries):
            row = {
                "item_id": item_id,
                "rep": i + 1,
            }
            row.update(rep_summary)
            per_rep_rows.append(row)
            ws = rep_summary.get("wall_seconds", 0)
            if ws is not None:
                total_wall += float(ws)
            aud = rep_summary.get("audio_seconds", 0)
            if aud is not None:
                total_audio += float(aud)

        agg = aggregate_across_reps(rep_summaries)
        per_item_agg[item_id] = agg

    workload_set_size = len(per_item_reps)

    summary = {
        "per_rep": per_rep_rows,
        "per_item_aggregation": per_item_agg,
        "run_totals": {
            "total_wall_seconds": round(total_wall, 3),
            "total_audio_seconds": round(total_audio, 3),
            "workload_set_size": workload_set_size,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
