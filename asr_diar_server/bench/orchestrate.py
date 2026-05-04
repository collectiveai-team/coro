"""Benchmark run orchestration: workload execution, artifact writing, manifest."""

from __future__ import annotations

import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from asr_diar_server.bench.performance import (
    compute_per_rep_summary,
    write_performance_summary,
)
from asr_diar_server.bench.sampling import Sampler
from asr_diar_server.bench.stm import hyp_segments_to_stm
from asr_diar_server.bench.transport import transcribe_audio


def run_workload(
    *,
    items: list[dict[str, Any]],
    base_url: str,
    out_dir: Path,
    reps: int,
    subcommand: str,
    cli_args: list[str] | None = None,
    der_collar: float = 0.0,
    der_regions: str = "all",
) -> None:
    resp_dir = out_dir / "responses"
    hyp_dir = out_dir / "hyp"
    ref_dir = out_dir / "ref"
    resp_dir.mkdir(parents=True, exist_ok=True)
    hyp_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    server_health = _fetch_health(base_url)

    for item in items:
        item_id = item["item_id"]
        audio_path = item["audio_path"]
        ref_stm_path = item.get("ref_stm_path")

        for rep in range(1, reps + 1):
            result = transcribe_audio(base_url, audio_path)
            resp_path = resp_dir / f"{item_id}_rep{rep}.json"
            resp_path.write_text(json.dumps(result, indent=2))

            if rep == 1 and ref_stm_path is not None:
                _write_hyp(hyp_dir, item_id, result)
                _write_ref(ref_dir, item_id, ref_stm_path)

    if subcommand == "quality":
        _run_quality_scoring(
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
        subcommand=subcommand,
    )


def _run_quality_scoring(
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

    for item in items:
        item_id = item["item_id"]
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

    summary = combine_items(item_results)
    (quality_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def _fetch_health(base_url: str) -> dict[str, Any]:
    import urllib.request

    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/health", timeout=10) as resp:
            return json.loads(resp.read())
    except Exception:
        return {}


def _write_hyp(hyp_dir: Path, item_id: str, result: dict[str, Any]) -> None:
    segments = result.get("segments", [])
    stm_text = hyp_segments_to_stm(segments, item_id)
    if stm_text:
        (hyp_dir / f"{item_id}.hyp.stm").write_text(stm_text)


def _write_ref(ref_dir: Path, item_id: str, ref_stm_path: Path) -> None:
    dst = ref_dir / f"{item_id}.ref.stm"
    if not dst.exists():
        shutil.copy2(ref_stm_path, dst)


def _write_manifest(
    *,
    out_dir: Path,
    items: list[dict[str, Any]],
    server_health: dict[str, Any],
    cli_args: list[str] | None,
    reps: int,
    subcommand: str,
) -> None:
    git_sha = _git_sha()
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "git_sha": git_sha,
        "cli_args": cli_args or [],
        "subcommand": subcommand,
        "reps": reps,
        "workload_set": [
            {
                "item_id": it["item_id"],
                "audio_path": str(it["audio_path"]),
                "ref_stm_path": str(it.get("ref_stm_path")) if it.get("ref_stm_path") else None,
            }
            for it in items
        ],
        "server_health": server_health,
        "versions": _collect_versions(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _git_sha() -> str:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _collect_versions() -> dict[str, str]:
    import importlib.metadata
    import subprocess

    versions: dict[str, str] = {}
    for pkg in ("asr_diar_server", "meeteval"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            pass

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        first_line = result.stdout.splitlines()[0] if result.stdout else ""
        versions["ffmpeg"] = first_line
    except Exception:
        pass

    return versions


def run_performance_workload(
    *,
    items: list[dict[str, Any]],
    base_url: str,
    out_dir: Path,
    reps: int,
    server_pid: int,
    sample_fn: Any | None = None,
    sample_interval: float = 0.25,
    cli_args: list[str] | None = None,
) -> None:
    resp_dir = out_dir / "responses"
    perf_dir = out_dir / "performance"
    resp_dir.mkdir(parents=True, exist_ok=True)
    perf_dir.mkdir(parents=True, exist_ok=True)

    server_health = _fetch_health(base_url)
    per_item_reps: dict[str, list[dict[str, Any]]] = {}

    for item in items:
        item_id = item["item_id"]
        audio_path = item["audio_path"]
        audio_seconds = _audio_duration(audio_path)
        rep_summaries: list[dict[str, Any]] = []

        for rep in range(1, reps + 1):
            sampler = Sampler(
                pid=server_pid,
                interval=sample_interval,
                sample_fn=sample_fn,
            )
            sampler.start()

            req_start = __import__("time").monotonic()
            result = transcribe_audio(base_url, audio_path)
            wall_seconds = __import__("time").monotonic() - req_start

            sampler.stop()

            throughput = audio_seconds / wall_seconds if wall_seconds > 0 and audio_seconds > 0 else None
            hw_profile = _infer_hw_profile(sampler.samples)

            sampler.backfill(
                wall_seconds=round(wall_seconds, 3),
                audio_seconds=round(audio_seconds, 3),
                transcription_throughput=round(throughput, 6) if throughput else "",
                time_to_first_delta_s="",
                observed_hardware_profile=hw_profile,
            )

            csv_path = perf_dir / f"resource_{item_id}_rep{rep}.csv"
            sampler.write_csv(csv_path)

            resp_path = resp_dir / f"{item_id}_rep{rep}.json"
            resp_path.write_text(json.dumps(result, indent=2))

            rep_summary = compute_per_rep_summary(csv_path)
            rep_summary.setdefault("wall_seconds", round(wall_seconds, 3))
            rep_summary.setdefault("audio_seconds", round(audio_seconds, 3))
            rep_summary.setdefault(
                "transcription_throughput",
                round(throughput, 6) if throughput else 0.0,
            )
            rep_summary.setdefault("observed_hardware_profile", hw_profile)
            rep_summaries.append(rep_summary)

        per_item_reps[item_id] = rep_summaries

    write_performance_summary(perf_dir, per_item_reps)

    _write_manifest(
        out_dir=out_dir,
        items=items,
        server_health=server_health,
        cli_args=cli_args,
        reps=reps,
        subcommand="performance",
    )


def _audio_duration(audio_path: Path) -> float:
    import subprocess

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _infer_hw_profile(samples: list[dict[str, Any]]) -> str:
    for row in samples:
        vram = row.get("server_vram_mib", "")
        if vram not in ("", None):
            return "cpu+gpu"
    return "cpu-only"
