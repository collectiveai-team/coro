"""Combined quality+performance workload for the 'all' subcommand."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from asr_diar_server.bench.performance import (
    compute_per_rep_summary,
    write_performance_summary,
)
from asr_diar_server.bench.sampling import Sampler, sample_resource_baseline
from asr_diar_server.bench.transport import transcribe_audio, transcribe_audio_sse


def run_all_workload(
    *,
    items: list[dict[str, Any]],
    base_url: str,
    out_dir: Path,
    reps: int,
    server_pid: int,
    sample_fn: Any | None = None,
    sample_interval: float = 0.25,
    cli_args: list[str] | None = None,
    der_collar: float = 0.0,
    der_regions: str = "all",
    warmup_audio: Path | None = None,
    stream: bool = False,
) -> None:
    import time

    from asr_diar_server.bench.orchestrate import (
        _audio_duration,
        _fetch_health,
        _infer_hw_profile,
        _write_hyp,
        _write_manifest,
        _write_ref,
    )

    resp_dir = out_dir / "responses"
    perf_dir = out_dir / "performance"
    hyp_dir = out_dir / "hyp"
    ref_dir = out_dir / "ref"
    resp_dir.mkdir(parents=True, exist_ok=True)
    perf_dir.mkdir(parents=True, exist_ok=True)
    hyp_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    server_health = _fetch_health(base_url)

    if warmup_audio is not None:
        transcribe_audio(base_url, warmup_audio)
    memory_baseline = sample_resource_baseline(server_pid, sample_fn=sample_fn)

    per_item_reps: dict[str, list[dict[str, Any]]] = {}

    for item in items:
        item_id = item["item_id"]
        audio_path = item["audio_path"]
        ref_stm_path = item.get("ref_stm_path")
        audio_seconds = _audio_duration(audio_path)
        item["audio_seconds"] = round(audio_seconds, 3)
        rep_summaries: list[dict[str, Any]] = []

        for rep in range(1, reps + 1):
            sampler = Sampler(
                pid=server_pid,
                interval=sample_interval,
                sample_fn=sample_fn,
            )
            sampler.start()

            req_start = time.monotonic()
            if stream:
                result, ttft = transcribe_audio_sse(base_url, audio_path)
            else:
                result = transcribe_audio(base_url, audio_path)
                ttft = None
            wall_seconds = time.monotonic() - req_start

            sampler.stop()

            throughput = audio_seconds / wall_seconds if wall_seconds > 0 and audio_seconds > 0 else None
            hw_profile = _infer_hw_profile(sampler.samples)

            sampler.backfill(
                wall_seconds=round(wall_seconds, 3),
                audio_seconds=round(audio_seconds, 3),
                transcription_throughput=round(throughput, 6) if throughput else "",
                time_to_first_delta_s=round(ttft, 6) if ttft is not None else "",
                observed_hardware_profile=hw_profile,
                **memory_baseline,
            )

            csv_path = perf_dir / f"resource_{item_id}_rep{rep}.csv"
            sampler.write_csv(csv_path)

            resp_path = resp_dir / f"{item_id}_rep{rep}.json"
            resp_path.write_text(json.dumps(result, indent=2))

            if rep == 1 and ref_stm_path is not None:
                _write_hyp(hyp_dir, item_id, result)
                _write_ref(ref_dir, item_id, ref_stm_path)

            rep_summary = compute_per_rep_summary(csv_path)
            rep_summary.setdefault("wall_seconds", round(wall_seconds, 3))
            rep_summary.setdefault("audio_seconds", round(audio_seconds, 3))
            rep_summary.setdefault(
                "transcription_throughput",
                round(throughput, 6) if throughput else 0.0,
            )
            rep_summary.setdefault("observed_hardware_profile", hw_profile)
            rep_summary.setdefault("baseline_pss_kb", memory_baseline.get("baseline_pss_kb", ""))
            rep_summary.setdefault("baseline_vram_mib", memory_baseline.get("baseline_vram_mib", ""))
            if ttft is not None:
                rep_summary["time_to_first_delta_s"] = round(ttft, 6)
            rep_summaries.append(rep_summary)

        per_item_reps[item_id] = rep_summaries

    write_performance_summary(perf_dir, per_item_reps)

    _run_quality_scoring_with_skip(
        out_dir=out_dir,
        items=items,
        der_collar=der_collar,
        der_regions=der_regions,
    )

    _write_manifest(
        out_dir=out_dir,
        items=items,
        server_health=server_health,
        cli_args=cli_args,
        reps=reps,
        subcommand="all",
        stream=stream,
        warmup=warmup_audio is not None,
    )


def _run_quality_scoring_with_skip(
    *,
    out_dir: Path,
    items: list[dict[str, Any]],
    der_collar: float,
    der_regions: str,
) -> None:
    from asr_diar_server.bench.quality import combine_items, score_item

    quality_dir = out_dir / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)

    hyp_dir = out_dir / "hyp"
    ref_dir = out_dir / "ref"

    item_results: list[dict[str, Any]] = []
    n_skipped = 0

    for item in items:
        item_id = item["item_id"]
        ref_stm_path = item.get("ref_stm_path")

        if ref_stm_path is None:
            n_skipped += 1
            continue

        hyp_path = hyp_dir / f"{item_id}.hyp.stm"
        ref_path = ref_dir / f"{item_id}.ref.stm"

        scored = score_item(
            ref_path,
            hyp_path,
            der_collar=der_collar,
            der_regions=der_regions,
        )

        scored["session_id"] = item_id
        scored["audio_seconds"] = item.get("audio_seconds", 0.0)

        raw = scored.pop("_raw", None)

        artifact: dict[str, Any] = {
            "session_id": item_id,
            "audio_seconds": item.get("audio_seconds", 0.0),
            "metrics": scored["metrics"],
        }
        if scored.get("error"):
            artifact["error"] = scored["error"]

        (quality_dir / f"{item_id}.json").write_text(json.dumps(artifact, indent=2))

        scored["_raw"] = raw
        item_results.append(scored)

    if item_results:
        summary = combine_items(item_results)
    else:
        summary = {
            "workload_set": [],
            "n_succeeded": 0,
            "n_failed": 0,
            "combined": {},
            "per_item": [],
        }
    summary["n_skipped"] = n_skipped
    (quality_dir / "summary.json").write_text(json.dumps(summary, indent=2))
