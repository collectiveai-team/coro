"""Pure functions over Resource CSVs for performance aggregation."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from asr_diar_server.bench.schema import RESOURCE_FIELDNAMES


def compute_per_rep_summary(csv_path: Path) -> dict[str, Any]:
    pss_values: list[float] = []
    cpu_values: list[float] = []
    wall_seconds: float | None = None
    throughput: float | None = None
    audio_seconds: float | None = None
    profile: str = ""

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pss_values.append(float(row.get("pss_kb", 0) or 0))
            except ValueError:
                pass
            try:
                cpu_values.append(float(row.get("cpu_pct", 0) or 0))
            except ValueError:
                pass
            if wall_seconds is None:
                ws = row.get("wall_seconds", "")
                if ws != "":
                    try:
                        wall_seconds = float(ws)
                    except ValueError:
                        pass
                tp = row.get("transcription_throughput", "")
                if tp != "":
                    try:
                        throughput = float(tp)
                    except ValueError:
                        pass
                aud = row.get("audio_seconds", "")
                if aud != "":
                    try:
                        audio_seconds = float(aud)
                    except ValueError:
                        pass
                profile = row.get("observed_hardware_profile", "")

    result: dict[str, Any] = {}
    if pss_values:
        result["peak_pss_kb"] = max(pss_values)
    if cpu_values:
        result["peak_cpu_pct"] = max(cpu_values)
    if wall_seconds is not None:
        result["wall_seconds"] = wall_seconds
    if throughput is not None:
        result["transcription_throughput"] = throughput
    if audio_seconds is not None:
        result["audio_seconds"] = audio_seconds
    result["observed_hardware_profile"] = profile or "cpu-only"
    return result


def aggregate_across_reps(per_rep_summaries: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metrics = ["transcription_throughput", "peak_pss_kb", "peak_cpu_pct"]
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
