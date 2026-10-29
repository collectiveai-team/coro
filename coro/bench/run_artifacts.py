"""Benchmark run artifacts: STM copies, the run manifest and its environment probes."""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import urllib.request
from collections.abc import Mapping
from datetime import datetime, UTC
from pathlib import Path
from typing import TYPE_CHECKING

from coro.bench.errors import ServerUnreachableError
from coro.bench.stm import hyp_response_to_stm
from coro.bench.transport import _is_connection_refused

if TYPE_CHECKING:
    from coro.bench.models.response import HealthResponse, TranscriptResponse
    from coro.bench.models.workload import WorkloadItem
    from coro.bench.sampling import ResourceRowTable


def fetch_health(base_url: str) -> HealthResponse:
    """Fetch and parse /health JSON from the server.

    Raises ServerUnreachableError if the TCP connection is refused, so the
    bench fails fast with a clear actionable message instead of crashing
    later inside transcribe_audio with a raw urllib stack trace.

    Other failures (e.g. 404, JSON parse errors) are swallowed and return an
    empty mapping to preserve the previous behavior of treating /health as
    optional.
    """
    try:
        # The benchmark's own --server-url, an http(s) URL by construction.
        health_url = f"{base_url.rstrip('/')}/health"
        with urllib.request.urlopen(health_url, timeout=10) as resp:  # noqa: S310
            return json.loads(resp.read())
    except Exception as exc:
        if _is_connection_refused(exc):
            raise ServerUnreachableError(base_url, cause=exc) from exc
        empty: HealthResponse = {}
        return empty


def write_hyp(hyp_dir: Path, item_id: str, result: TranscriptResponse) -> None:
    stm_text = hyp_response_to_stm(result, item_id)
    if stm_text:
        (hyp_dir / f"{item_id}.hyp.stm").write_text(stm_text)


def write_ref(ref_dir: Path, item_id: str, ref_stm_path: Path) -> None:
    """Copy the Reference STM into the run's ``ref/`` directory, overwriting any stale copy."""
    shutil.copy2(ref_stm_path, ref_dir / f"{item_id}.ref.stm")


def write_manifest(
    *,
    out_dir: Path,
    items: list[WorkloadItem],
    server_health: HealthResponse,
    cli_args: list[str] | None,
    reps: int,
    subcommand: str,
    stream: bool = False,
    deepgram: bool = False,
    language: str | None = None,
    warmup: bool = False,
) -> None:
    git_sha = _git_sha()
    manifest = {
        "timestamp": datetime.now(UTC).isoformat(),
        "hostname": platform.node(),
        "git_sha": git_sha,
        "cli_args": cli_args or [],
        "subcommand": subcommand,
        "reps": reps,
        "warmup": warmup,
        "stream": stream,
        # Which wire surface produced the responses: the Deepgram endpoint
        # carries per-word speakers, the OpenAI one only a segment summary.
        "deepgram": deepgram,
        # Experimental decoder condition. None deliberately means automatic
        # language detection and must remain distinguishable from forced runs.
        "language": language,
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
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 - system git by name
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip()


def _collect_versions() -> Mapping[str, str]:
    versions: dict[str, str] = {}
    for pkg in ("coro", "meeteval"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[pkg] = importlib.metadata.version(pkg)

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],  # noqa: S607 - system ffmpeg by name
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return versions
    versions["ffmpeg"] = result.stdout.splitlines()[0] if result.stdout else ""
    return versions


def audio_duration(audio_path: Path) -> float:
    """Return the duration of ``audio_path`` from ``ffprobe``, or 0.0 when it cannot tell."""
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv plus a local path, no shell
            [  # noqa: S607 - system ffprobe by name
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0.0


def infer_hw_profile(samples: list[ResourceRowTable]) -> str:
    """Infer hardware profile from collected resource samples.

    Returns ``"cpu+gpu"`` if any sample contains non-empty GPU metrics
    (either VRAM usage or GPU utilisation), otherwise ``"cpu-only"``.
    """
    for row in samples:
        vram = row.get("server_vram_mib", "")
        util = row.get("gpu_util_pct", "")
        if vram not in ("", None) or util not in ("", None):
            return "cpu+gpu"
    return "cpu-only"
