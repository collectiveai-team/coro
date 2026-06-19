"""Quality Benchmark scoring: MeetEval Metric Set and run-level summary."""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path
import re
import string
from typing import Any


def _require_meeteval():
    try:
        import meeteval

        return meeteval
    except ImportError:
        print(
            "Error: meeteval is required for quality scoring.\n"
            "Install with: pip install coro[bench]",
            file=sys.stderr,
        )
        sys.exit(1)


def _wer_to_dict(result) -> dict[str, Any]:
    """Convert a meeteval WER result object to a plain dict.

    meeteval 0.4.x uses `error_rate` instead of `wer` as the attribute name.
    """
    return {
        "wer": result.error_rate,
        "errors": result.errors,
        "length": result.length,
        "insertions": result.insertions,
        "deletions": result.deletions,
        "substitutions": result.substitutions,
    }


def _der_to_dict(result) -> dict[str, Any]:
    """Convert a meeteval DER result object to a plain dict.

    meeteval 0.4.x DiaErrorRate uses `*_speaker_time` field names and
    returns Decimal values; cast to float for JSON serialisability.
    """
    return {
        "der": float(result.error_rate),
        "false_alarm": float(result.falarm_speaker_time),
        "missed_detection": float(result.missed_speaker_time),
        "speaker_error": float(result.speaker_error_time),
        "total_speech": float(result.scored_speaker_time),
    }


def _combine_multifile(meeteval, results: dict) -> Any:
    """Combine per-session meeteval results into a single aggregate."""
    return meeteval.wer.combine_error_rates(*results.values())


_PUNCTUATION_TRANS = str.maketrans("", "", string.punctuation)


def _normalize_transcript_text(text: str) -> str:
    """Remove punctuation and collapse repeated whitespace in transcript text."""
    no_punctuation = text.translate(_PUNCTUATION_TRANS)
    return re.sub(r"\s+", " ", no_punctuation).strip()


def is_diarization_only_stm(path: Path) -> bool:
    """Return True when every STM line's text is the diarization-only sentinel.

    Such references (e.g. VoxConverse RTTM converted via rttm_to_stm) carry
    speaker turns but no transcript, so only DER is meaningful.
    """
    from coro.bench.stm import DIARIZATION_ONLY_TEXT

    saw_line = False
    for line in path.read_text().splitlines():
        parts = line.strip().split(maxsplit=5)
        if len(parts) < 6:
            continue
        saw_line = True
        if parts[5].strip() != DIARIZATION_ONLY_TEXT:
            return False
    return saw_line


def _count_stm_speakers(path: Path) -> int:
    """Count distinct speaker labels (column 3) in an STM file."""
    speakers: set[str] = set()
    for line in path.read_text().splitlines():
        parts = line.strip().split(maxsplit=5)
        if len(parts) >= 3:
            speakers.add(parts[2])
    return len(speakers)


def diarization_sanity(ref_stm_path: Path, hyp_stm_path: Path) -> dict[str, Any]:
    """Flag degenerate diarization: a single hyp speaker against multi-speaker ref.

    A diarization-invariant metric like ORC-WER stays low even when every word
    collapses onto one speaker, so this check surfaces the failure the WER
    headline would otherwise hide.
    """
    ref_speakers = _count_stm_speakers(ref_stm_path)
    hyp_speakers = _count_stm_speakers(hyp_stm_path)
    degenerate = hyp_speakers <= 1 and ref_speakers > 1
    return {
        "ref_speakers": ref_speakers,
        "hyp_speakers": hyp_speakers,
        "degenerate": degenerate,
    }


def _write_normalized_stm(src: Path, dst: Path) -> None:
    """Write an STM file with only the transcript text field normalized."""
    lines: list[str] = []
    for line in src.read_text().splitlines():
        parts = line.strip().split(maxsplit=5)
        if len(parts) < 6:
            continue
        text = _normalize_transcript_text(parts[5])
        if not text:
            continue
        lines.append(" ".join([*parts[:5], text]))
    dst.write_text("\n".join(lines) + ("\n" if lines else ""))


def score_item(
    ref_stm_path: Path,
    hyp_stm_path: Path,
    *,
    der_collar: float = 0.0,
    der_regions: str = "all",
) -> dict[str, Any]:
    """Score one hypothesis STM against the reference STM.

    Passes file paths directly to meeteval so it handles STM parsing
    internally. The multifile API returns dict[session_id -> result];
    results are combined across sessions with combine_error_rates.

    siWER (SISO-WER) is omitted because AMI data has multiple speakers
    per session, making (session, speaker) pairs non-unique — a hard
    requirement of siWER.
    """
    meeteval = _require_meeteval()

    try:
        raw: dict[str, Any] = {}
        metrics: dict[str, Any] = {}

        # Diarization-only references (speaker turns, no transcript) can only be
        # scored for DER; computing WER against a sentinel transcript would be
        # meaningless, so it is skipped.
        diarization_only = is_diarization_only_stm(ref_stm_path)

        if not diarization_only:
            raw["cpwer"] = _combine_multifile(
                meeteval, meeteval.wer.cpwer(ref_stm_path, hyp_stm_path)
            )
            metrics["cpwer"] = _wer_to_dict(raw["cpwer"])

            raw["orcwer"] = _combine_multifile(
                meeteval, meeteval.wer.greedy_orcwer(ref_stm_path, hyp_stm_path)
            )
            metrics["orcwer"] = _wer_to_dict(raw["orcwer"])

            raw["dicpwer"] = _combine_multifile(
                meeteval, meeteval.wer.greedy_dicpwer(ref_stm_path, hyp_stm_path)
            )
            metrics["dicpwer"] = _wer_to_dict(raw["dicpwer"])

            with tempfile.TemporaryDirectory(prefix="coro-quality-") as tmp:
                tmp_dir = Path(tmp)
                normalized_ref = tmp_dir / ref_stm_path.name
                normalized_hyp = tmp_dir / hyp_stm_path.name
                _write_normalized_stm(ref_stm_path, normalized_ref)
                _write_normalized_stm(hyp_stm_path, normalized_hyp)

                normalized_metrics: dict[str, Any] = {}
                raw["normalized_cpwer"] = _combine_multifile(
                    meeteval, meeteval.wer.cpwer(normalized_ref, normalized_hyp)
                )
                normalized_metrics["cpwer"] = _wer_to_dict(raw["normalized_cpwer"])

                raw["normalized_orcwer"] = _combine_multifile(
                    meeteval, meeteval.wer.greedy_orcwer(normalized_ref, normalized_hyp)
                )
                normalized_metrics["orcwer"] = _wer_to_dict(raw["normalized_orcwer"])

                raw["normalized_dicpwer"] = _combine_multifile(
                    meeteval, meeteval.wer.greedy_dicpwer(normalized_ref, normalized_hyp)
                )
                normalized_metrics["dicpwer"] = _wer_to_dict(raw["normalized_dicpwer"])
                metrics["normalized"] = normalized_metrics

        der_results = meeteval.der.md_eval_22(
            ref_stm_path,
            hyp_stm_path,
            collar=der_collar,
            regions=der_regions,
        )
        raw["der"] = _combine_multifile(meeteval, der_results)
        metrics["der"] = _der_to_dict(raw["der"])

        return {
            "metrics": metrics,
            "_raw": raw,
            "diarization_only": diarization_only,
            "diarization": diarization_sanity(ref_stm_path, hyp_stm_path),
        }

    except Exception as exc:
        # Print full traceback to stderr so the operator can see the real cause
        # without having to dig into the JSON artifact.
        print(
            f"[bench/quality] scoring failed for {hyp_stm_path.name}:\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        return {
            "metrics": None,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }


# sisower removed: SISO-WER requires unique (session, speaker) pairs,
# which AMI multi-speaker meetings do not satisfy.
WER_METRIC_KEYS = ("cpwer", "orcwer", "dicpwer")


def _combine_raw_key(meeteval, succeeded: list[dict], raw_key: str) -> dict[str, Any] | None:
    """Aggregate one ``_raw`` key (WER or DER) across all succeeded items."""
    raw_objects = [r["_raw"][raw_key] for r in succeeded if raw_key in r.get("_raw", {})]
    if not raw_objects:
        return None
    combined = meeteval.wer.combine_error_rates(*raw_objects)
    converter = _der_to_dict if raw_key == "der" else _wer_to_dict
    return converter(combined)


def _combined_metrics(meeteval, succeeded: list[dict]) -> dict[str, Any]:
    """Build the workload-level combined metric block from succeeded items."""
    combined: dict[str, Any] = {
        key: _combine_raw_key(meeteval, succeeded, key) for key in WER_METRIC_KEYS
    }
    combined["normalized"] = {
        key: _combine_raw_key(meeteval, succeeded, f"normalized_{key}") for key in WER_METRIC_KEYS
    }
    combined["der"] = _combine_raw_key(meeteval, succeeded, "der")
    return combined


def _per_item_entry(result: dict[str, Any]) -> dict[str, Any]:
    """Flatten one item's metrics + diarization sanity into a summary row."""
    entry: dict[str, Any] = {"session_id": result.get("session_id", "")}
    if "audio_seconds" in result:
        entry["audio_seconds"] = result["audio_seconds"]
    if result.get("diarization_only"):
        entry["diarization_only"] = True
    if result.get("diarization") is not None:
        entry["diarization"] = result["diarization"]
    metrics = result.get("metrics")
    if metrics is not None:
        for key in WER_METRIC_KEYS:
            if metrics.get(key) is not None:
                entry[key] = metrics[key]["wer"]
        if metrics.get("der") is not None:
            entry["der"] = metrics["der"]["der"]
        normalized = metrics.get("normalized", {})
        for key in WER_METRIC_KEYS:
            if normalized.get(key) is not None:
                entry[f"normalized_{key}"] = normalized[key]["wer"]
    return entry


def combine_items(item_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-item score_item results into a workload-level summary."""
    meeteval = _require_meeteval()

    succeeded = [r for r in item_results if r.get("metrics") is not None]
    failed = [r for r in item_results if r.get("metrics") is None]
    n_degenerate = sum(1 for r in item_results if (r.get("diarization") or {}).get("degenerate"))

    return {
        "workload_set": [r.get("session_id", "") for r in item_results],
        "n_succeeded": len(succeeded),
        "n_failed": len(failed),
        "n_degenerate_diarization": n_degenerate,
        "combined": _combined_metrics(meeteval, succeeded),
        "per_item": [_per_item_entry(r) for r in item_results],
    }
