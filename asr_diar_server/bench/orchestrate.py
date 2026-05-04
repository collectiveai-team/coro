"""Benchmark run orchestration: workload execution, artifact writing, manifest."""

from __future__ import annotations

import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

    _write_manifest(
        out_dir=out_dir,
        items=items,
        server_health=server_health,
        cli_args=cli_args,
        reps=reps,
        subcommand=subcommand,
    )


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
