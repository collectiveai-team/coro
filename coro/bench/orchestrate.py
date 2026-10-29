"""Benchmark run orchestration: workload execution, artifact writing, manifest."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from coro.bench.models.performance import PerRepSummary
from coro.bench.models.resource import ResourceBaseline
from coro.bench.performance import (
    compute_per_rep_summary,
    write_performance_summary,
)
from coro.bench.quality_artifacts import run_quality_scoring, run_quality_scoring_with_skip
from coro.bench.run_artifacts import (
    audio_duration,
    fetch_health,
    infer_hw_profile,
    write_hyp,
    write_manifest,
    write_ref,
)
from coro.bench.sampling import SampleFn, Sampler, sample_resource_baseline
from coro.bench.transport import TranscribeFn, select_transport

if TYPE_CHECKING:
    from coro.bench.models.response import TranscriptResponse
    from coro.bench.models.workload import WorkloadItem


def _finalize_rep_summary(
    rep_summary: PerRepSummary,
    *,
    wall_seconds: float,
    audio_seconds: float,
    throughput: float | None,
    hw_profile: str,
    memory_baseline: ResourceBaseline,
    ttft: float | None,
) -> None:
    """Fill in run-scalar fields the CSV-derived summary did not already set."""
    if rep_summary.wall_seconds is None:
        rep_summary.wall_seconds = round(wall_seconds, 3)
    if rep_summary.audio_seconds is None:
        rep_summary.audio_seconds = round(audio_seconds, 3)
    if rep_summary.transcription_throughput is None:
        rep_summary.transcription_throughput = round(throughput, 6) if throughput else 0.0
    if rep_summary.baseline_pss_kb is None:
        rep_summary.baseline_pss_kb = memory_baseline.baseline_pss_kb
    if rep_summary.baseline_vram_mib is None:
        rep_summary.baseline_vram_mib = memory_baseline.baseline_vram_mib
    if ttft is not None:
        rep_summary.time_to_first_delta_s = round(ttft, 6)


@dataclass(frozen=True)
class _SampledRun:
    """What every measured rep shares: the transport, the server and the sampler setup."""

    transcribe: TranscribeFn
    base_url: str
    server_pid: int
    sample_fn: SampleFn | None
    sample_interval: float
    memory_baseline: ResourceBaseline
    perf_dir: Path


def _measure_rep(
    run: _SampledRun,
    *,
    item_id: str,
    audio_path: Path,
    audio_seconds: float,
    rep: int,
) -> tuple[TranscriptResponse, PerRepSummary]:
    """Transcribe one item once while sampling the server, and write its Resource CSV."""
    sampler = Sampler(pid=run.server_pid, interval=run.sample_interval, sample_fn=run.sample_fn)
    sampler.start()

    req_start = time.monotonic()
    result, ttft = run.transcribe(run.base_url, audio_path)
    wall_seconds = time.monotonic() - req_start

    sampler.stop()

    throughput = audio_seconds / wall_seconds if wall_seconds > 0 and audio_seconds > 0 else None
    hw_profile = infer_hw_profile(sampler.samples)

    sampler.backfill(
        wall_seconds=round(wall_seconds, 3),
        audio_seconds=round(audio_seconds, 3),
        transcription_throughput=round(throughput, 6) if throughput else "",
        time_to_first_delta_s=round(ttft, 6) if ttft is not None else "",
        observed_hardware_profile=hw_profile,
        **asdict(run.memory_baseline),
    )

    csv_path = run.perf_dir / f"resource_{item_id}_rep{rep}.csv"
    sampler.write_csv(csv_path)

    rep_summary = compute_per_rep_summary(csv_path)
    _finalize_rep_summary(
        rep_summary,
        wall_seconds=wall_seconds,
        audio_seconds=audio_seconds,
        throughput=throughput,
        hw_profile=hw_profile,
        memory_baseline=run.memory_baseline,
        ttft=ttft,
    )
    return result, rep_summary


def _make_dirs(out_dir: Path, *names: str) -> None:
    for name in names:
        (out_dir / name).mkdir(parents=True, exist_ok=True)


def run_workload(
    *,
    items: list[WorkloadItem],
    base_url: str,
    out_dir: Path,
    reps: int,
    subcommand: str,
    cli_args: list[str] | None = None,
    der_collar: float = 0.0,
    der_regions: str = "all",
    deepgram: bool = False,
    language: str | None = None,
) -> None:
    resp_dir = out_dir / "responses"
    hyp_dir = out_dir / "hyp"
    ref_dir = out_dir / "ref"
    _make_dirs(out_dir, "responses", "hyp", "ref")

    transcribe = select_transport(deepgram=deepgram, language=language)
    server_health = fetch_health(base_url)

    for item in items:
        item_id = item["item_id"]
        audio_path = item["audio_path"]
        ref_stm_path = item.get("ref_stm_path")
        if not item.get("audio_seconds"):
            item["audio_seconds"] = round(audio_duration(audio_path), 3)

        for rep in range(1, reps + 1):
            result, _ = transcribe(base_url, audio_path)
            resp_path = resp_dir / f"{item_id}_rep{rep}.json"
            resp_path.write_text(json.dumps(result, indent=2))

            if rep == 1 and ref_stm_path is not None:
                write_hyp(hyp_dir, item_id, result)
                write_ref(ref_dir, item_id, ref_stm_path)

    if subcommand == "quality":
        run_quality_scoring(
            out_dir=out_dir,
            items=items,
            der_collar=der_collar,
            der_regions=der_regions,
        )

    write_manifest(
        out_dir=out_dir,
        items=items,
        server_health=server_health,
        cli_args=cli_args,
        reps=reps,
        subcommand=subcommand,
        deepgram=deepgram,
        language=language,
    )


def run_all_workload(
    *,
    items: list[WorkloadItem],
    base_url: str,
    out_dir: Path,
    reps: int,
    server_pid: int,
    sample_fn: SampleFn | None = None,
    sample_interval: float = 0.25,
    cli_args: list[str] | None = None,
    der_collar: float = 0.0,
    der_regions: str = "all",
    warmup_audio: Path | None = None,
    stream: bool = False,
    deepgram: bool = False,
    language: str | None = None,
) -> None:
    resp_dir = out_dir / "responses"
    perf_dir = out_dir / "performance"
    hyp_dir = out_dir / "hyp"
    ref_dir = out_dir / "ref"
    _make_dirs(out_dir, "responses", "performance", "hyp", "ref")

    transcribe = select_transport(stream=stream, deepgram=deepgram, language=language)
    server_health = fetch_health(base_url)

    # Warm up over the same endpoint the run will measure, so the first
    # measured request is not the one paying that endpoint's setup cost.
    if warmup_audio is not None:
        transcribe(base_url, warmup_audio)
    run = _SampledRun(
        transcribe=transcribe,
        base_url=base_url,
        server_pid=server_pid,
        sample_fn=sample_fn,
        sample_interval=sample_interval,
        memory_baseline=sample_resource_baseline(server_pid, sample_fn=sample_fn),
        perf_dir=perf_dir,
    )

    per_item_reps: dict[str, list[PerRepSummary]] = {}

    for item in items:
        item_id = item["item_id"]
        audio_path = item["audio_path"]
        ref_stm_path = item.get("ref_stm_path")
        audio_seconds = audio_duration(audio_path)
        item["audio_seconds"] = round(audio_seconds, 3)
        rep_summaries: list[PerRepSummary] = []

        for rep in range(1, reps + 1):
            result, rep_summary = _measure_rep(
                run, item_id=item_id, audio_path=audio_path, audio_seconds=audio_seconds, rep=rep
            )

            resp_path = resp_dir / f"{item_id}_rep{rep}.json"
            resp_path.write_text(json.dumps(result, indent=2))

            if rep == 1 and ref_stm_path is not None:
                write_hyp(hyp_dir, item_id, result)
                write_ref(ref_dir, item_id, ref_stm_path)

            rep_summaries.append(rep_summary)

        per_item_reps[item_id] = rep_summaries

    write_performance_summary(perf_dir, per_item_reps)

    run_quality_scoring_with_skip(
        out_dir=out_dir,
        items=items,
        der_collar=der_collar,
        der_regions=der_regions,
    )

    write_manifest(
        out_dir=out_dir,
        items=items,
        server_health=server_health,
        cli_args=cli_args,
        reps=reps,
        subcommand="all",
        stream=stream,
        deepgram=deepgram,
        language=language,
        warmup=warmup_audio is not None,
    )


def run_performance_workload(
    *,
    items: list[WorkloadItem],
    base_url: str,
    out_dir: Path,
    reps: int,
    server_pid: int,
    sample_fn: SampleFn | None = None,
    sample_interval: float = 0.25,
    cli_args: list[str] | None = None,
    stream: bool = False,
    deepgram: bool = False,
    language: str | None = None,
    warmup_audio: Path | None = None,
) -> None:
    resp_dir = out_dir / "responses"
    perf_dir = out_dir / "performance"
    _make_dirs(out_dir, "responses", "performance")

    transcribe = select_transport(stream=stream, deepgram=deepgram, language=language)
    server_health = fetch_health(base_url)
    # Warm up over the same endpoint the run will measure.
    if warmup_audio is not None:
        transcribe(base_url, warmup_audio)
    run = _SampledRun(
        transcribe=transcribe,
        base_url=base_url,
        server_pid=server_pid,
        sample_fn=sample_fn,
        sample_interval=sample_interval,
        memory_baseline=sample_resource_baseline(server_pid, sample_fn=sample_fn),
        perf_dir=perf_dir,
    )
    per_item_reps: dict[str, list[PerRepSummary]] = {}

    for item in items:
        item_id = item["item_id"]
        audio_path = item["audio_path"]
        audio_seconds = audio_duration(audio_path)
        rep_summaries: list[PerRepSummary] = []

        for rep in range(1, reps + 1):
            result, rep_summary = _measure_rep(
                run, item_id=item_id, audio_path=audio_path, audio_seconds=audio_seconds, rep=rep
            )
            resp_path = resp_dir / f"{item_id}_rep{rep}.json"
            resp_path.write_text(json.dumps(result, indent=2))
            rep_summaries.append(rep_summary)

        per_item_reps[item_id] = rep_summaries

    write_performance_summary(perf_dir, per_item_reps)

    write_manifest(
        out_dir=out_dir,
        items=items,
        server_health=server_health,
        cli_args=cli_args,
        reps=reps,
        subcommand="performance",
        stream=stream,
        deepgram=deepgram,
        language=language,
        warmup=warmup_audio is not None,
    )
