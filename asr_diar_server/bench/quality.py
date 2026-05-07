"""Quality Benchmark scoring: MeetEval Metric Set and run-level summary."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _require_meeteval():
    try:
        import meeteval
        return meeteval
    except ImportError:
        print(
            "Error: meeteval is required for quality scoring.\n"
            "Install with: pip install asr-diar-server[bench]",
            file=sys.stderr,
        )
        sys.exit(1)


def _parse_stm(path: Path) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for line in path.read_text().strip().splitlines():
        parts = line.strip().split(maxsplit=5)
        if len(parts) < 6:
            continue
        segments.append({
            "recording_id": parts[0],
            "channel": parts[1],
            "speaker": parts[2],
            "start_time": float(parts[3]),
            "end_time": float(parts[4]),
            "text": parts[5],
        })
    return segments


def _wer_to_dict(result) -> dict[str, Any]:
    return {
        "wer": result.wer,
        "errors": result.errors,
        "length": result.length,
        "insertions": result.insertions,
        "deletions": result.deletions,
        "substitutions": result.substitutions,
    }


def _der_to_dict(result) -> dict[str, Any]:
    return {
        "der": result.der,
        "false_alarm": result.false_alarm,
        "missed_detection": result.missed_detection,
        "speaker_error": result.speaker_error,
        "total_speech": result.total_speech,
    }


def score_item(
    ref_stm_path: Path,
    hyp_stm_path: Path,
    *,
    der_collar: float = 0.0,
    der_regions: str = "all",
) -> dict[str, Any]:
    meeteval = _require_meeteval()

    try:
        ref = _parse_stm(ref_stm_path)
        hyp = _parse_stm(hyp_stm_path)

        raw: dict[str, Any] = {}
        metrics: dict[str, Any] = {}

        raw["siwer"] = meeteval.wer.siwer(ref, hyp)
        metrics["siwer"] = _wer_to_dict(raw["siwer"])
        raw["cpwer"] = meeteval.wer.cpwer(ref, hyp)
        metrics["cpwer"] = _wer_to_dict(raw["cpwer"])
        raw["orcwer"] = meeteval.wer.greedy_orcwer(ref, hyp)
        metrics["orcwer"] = _wer_to_dict(raw["orcwer"])
        raw["dicpwer"] = meeteval.wer.greedy_dicpwer(ref, hyp)
        metrics["dicpwer"] = _wer_to_dict(raw["dicpwer"])
        raw["der"] = meeteval.der.md_eval_22(
            ref, hyp, collar=der_collar, regions=der_regions
        )
        metrics["der"] = _der_to_dict(raw["der"])

        return {"metrics": metrics, "_raw": raw}
    except Exception as exc:
        return {
            "metrics": None,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }


WER_METRIC_KEYS = ("siwer", "cpwer", "orcwer", "dicpwer")


def combine_items(item_results: list[dict[str, Any]]) -> dict[str, Any]:
    meeteval = _require_meeteval()

    succeeded = [r for r in item_results if r.get("metrics") is not None]
    failed = [r for r in item_results if r.get("metrics") is None]

    combined: dict[str, Any] = {}
    for key in WER_METRIC_KEYS:
        raw_objects = [r["_raw"][key] for r in succeeded]
        if raw_objects:
            combined_result = meeteval.wer.combine_error_rates(*raw_objects)
            combined[key] = _wer_to_dict(combined_result)
        else:
            combined[key] = None

    if succeeded:
        combined["der"] = _der_to_dict(succeeded[0]["_raw"]["der"])
    else:
        combined["der"] = None

    per_item = []
    for r in item_results:
        entry: dict[str, Any] = {"session_id": r.get("session_id", "")}
        if r.get("metrics") is not None:
            for key in WER_METRIC_KEYS:
                entry[key] = r["metrics"][key]["wer"]
            entry["der"] = r["metrics"]["der"]["der"]
        per_item.append(entry)

    return {
        "workload_set": [r.get("session_id", "") for r in item_results],
        "n_succeeded": len(succeeded),
        "n_failed": len(failed),
        "combined": combined,
        "per_item": per_item,
    }
