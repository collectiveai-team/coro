"""Benchmark run orchestration — migrated from tools/bench_asr.py.

This module contains the full Resource Benchmark implementation.
Heavy imports (psutil, requests, etc.) are deferred to this module
so that importing asr_diar_server.bench.cli stays lightweight.
"""

from __future__ import annotations

import argparse
import csv
import os
import string
import subprocess
import sys
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any


CLOCK_TICKS = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2

from asr_diar_server.bench.schema import RESOURCE_FIELDNAMES  # noqa: E402


def _get_process_tree_pids(root_pid: int) -> set[int]:
    """Return all PIDs in the process tree rooted at root_pid."""
    pids = {root_pid}
    try:
        result = subprocess.run(
            ["pgrep", "-P", str(root_pid)],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.strip().splitlines():
            child = int(line)
            pids |= _get_process_tree_pids(child)
    except Exception:
        pass
    return pids


def _read_proc_smaps_rollup(pid: int) -> dict[str, int]:
    try:
        path = f"/proc/{pid}/smaps_rollup"
        data: dict[str, int] = {}
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0].endswith(":"):
                    key = parts[0][:-1]
                    try:
                        data[key] = int(parts[1])
                    except ValueError:
                        pass
        return data
    except Exception:
        return {}


def _read_proc_io(pid: int) -> dict[str, int]:
    try:
        data: dict[str, int] = {}
        with open(f"/proc/{pid}/io") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    try:
                        data[parts[0].rstrip(":")] = int(parts[1])
                    except ValueError:
                        pass
        return data
    except Exception:
        return {}


def _read_proc_stat(pid: int) -> dict[str, Any]:
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().split()
        return {
            "utime": int(fields[13]),
            "stime": int(fields[14]),
            "num_threads": int(fields[19]),
        }
    except Exception:
        return {}


def _read_proc_status(pid: int) -> dict[str, int]:
    try:
        data: dict[str, int] = {}
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0].endswith(":"):
                    key = parts[0][:-1]
                    if key in ("VmRSS", "VmSize"):
                        try:
                            data[key] = int(parts[1])
                        except ValueError:
                            pass
        return data
    except Exception:
        return {}


def sample_process_tree(root_pid: int) -> dict[str, Any]:
    """Sample resource metrics for the full Server Process Tree."""
    pids = _get_process_tree_pids(root_pid)
    total_pss = total_uss = total_rss = total_vsz = 0
    total_utime = total_stime = total_threads = 0
    total_rchar = total_wchar = total_read_bytes = total_write_bytes = 0

    for pid in pids:
        smaps = _read_proc_smaps_rollup(pid)
        total_pss += smaps.get("Pss", 0)
        total_uss += smaps.get("Private_Clean", 0) + smaps.get("Private_Dirty", 0)
        io = _read_proc_io(pid)
        total_rchar += io.get("rchar", 0)
        total_wchar += io.get("wchar", 0)
        total_read_bytes += io.get("read_bytes", 0)
        total_write_bytes += io.get("write_bytes", 0)
        stat = _read_proc_stat(pid)
        total_utime += stat.get("utime", 0)
        total_stime += stat.get("stime", 0)
        total_threads += stat.get("num_threads", 0)
        status = _read_proc_status(pid)
        total_rss += status.get("VmRSS", 0)
        total_vsz += status.get("VmSize", 0)

    return {
        "pids": pids,
        "pss_kb": total_pss,
        "uss_kb": total_uss,
        "rss_kb": total_rss,
        "vsz_kb": total_vsz,
        "cpu_user_s": total_utime / CLOCK_TICKS,
        "cpu_system_s": total_stime / CLOCK_TICKS,
        "rchar": total_rchar,
        "wchar": total_wchar,
        "read_bytes": total_read_bytes,
        "write_bytes": total_write_bytes,
        "thread_count": total_threads,
    }


def _normalize_text_wer(text: str) -> str:
    """Spanish-friendly WER normalization: lowercase, NFC, strip punct, collapse ws."""
    text = unicodedata.normalize("NFC", text.lower())
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def find_server_pid(server_pid: int | None, server_match: str) -> int:
    """Locate the server PID from explicit value or pgrep pattern."""
    if server_pid is not None:
        try:
            os.kill(server_pid, 0)
        except PermissionError:
            pass
        except ProcessLookupError:
            print(f"Error: server PID {server_pid} is not running", file=sys.stderr)
            sys.exit(1)
        return server_pid

    result = subprocess.run(["pgrep", "-f", server_match], capture_output=True, text=True)
    pids = result.stdout.strip().splitlines()
    if not pids:
        print(
            f"Error: could not find server matching '{server_match}'. "
            "Set --server-pid or --server-match.",
            file=sys.stderr,
        )
        sys.exit(1)
    if len(pids) > 1:
        print(
            f"Warning: {len(pids)} processes matching '{server_match}', using {pids[0]}.",
            file=sys.stderr,
        )
    return int(pids[0])


def run_benchmark(args: argparse.Namespace) -> None:
    """Execute the Resource Benchmark as configured by args."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root_pid = find_server_pid(getattr(args, "server_pid", None), args.server_match)
    audio_path = Path(args.audio)

    # Determine audio duration
    ffprobe_result = subprocess.run(
        [
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
    try:
        audio_seconds = float(ffprobe_result.stdout.strip())
    except ValueError:
        audio_seconds = 0.0

    rep_results = []
    for rep in range(1, args.reps + 1):
        result = _run_rep(args, rep, out_dir, root_pid, audio_path, audio_seconds)
        rep_results.append(result)

    _write_summary(args, out_dir, rep_results, audio_seconds)


def _run_rep(
    args, rep: int, out_dir: Path, root_pid: int, audio_path: Path, audio_seconds: float
) -> dict:
    """Run one benchmark repetition and write Resource CSV."""
    resource_csv_path = out_dir / f"resource_{rep}.csv"
    response_path = out_dir / f"response_{rep}.{args.mode}"
    curl_metrics_path = out_dir / f"curl_metrics_{rep}.txt"

    samples: list[dict] = []
    prev_sample: dict[str, Any] | None = None
    prev_pids: set[int] = set()
    sampling_warning = False
    stop_event = threading.Event()
    start_time = time.monotonic()

    def _sample_loop():
        nonlocal prev_sample, prev_pids, sampling_warning
        while not stop_event.is_set():
            ts = time.time()
            elapsed = time.monotonic() - start_time
            sample = sample_process_tree(root_pid)
            new_pids = sample["pids"] - prev_pids
            gone_pids = prev_pids - sample["pids"]
            prev_pids.update(sample["pids"])

            cpu_pct = 0.0
            sample_dt = args.sample_interval
            if prev_sample is not None:
                dt = elapsed - float(prev_sample.get("elapsed_s", elapsed))
                if dt > 0:
                    du = sample["cpu_user_s"] - prev_sample.get("cpu_user_s", 0.0)
                    ds = sample["cpu_system_s"] - prev_sample.get("cpu_system_s", 0.0)
                    cpu_pct = 100.0 * (du + ds) / dt
                    sample_dt = dt
                    if dt > args.sample_interval * 2:
                        sampling_warning = True

            row: dict[str, Any] = {
                "ts_epoch": round(ts, 3),
                "elapsed_s": round(elapsed, 3),
                "sample_dt_s": round(sample_dt, 3),
                "root_pid": root_pid,
                "process_count": len(sample["pids"]),
                "new_pids": len(new_pids),
                "gone_pids": len(gone_pids),
                "rss_kb": sample["rss_kb"],
                "pss_kb": sample["pss_kb"],
                "uss_kb": sample["uss_kb"],
                "vsz_kb": sample["vsz_kb"],
                "cpu_user_s": round(sample["cpu_user_s"], 3),
                "cpu_system_s": round(sample["cpu_system_s"], 3),
                "cpu_total_s": round(sample["cpu_user_s"] + sample["cpu_system_s"], 3),
                "cpu_pct": round(cpu_pct, 2),
                "thread_count": sample["thread_count"],
                "io_rchar_bytes": sample["rchar"],
                "io_wchar_bytes": sample["wchar"],
                "io_read_bytes": sample["read_bytes"],
                "io_write_bytes": sample["write_bytes"],
                "io_rchar_bps": 0.0,
                "io_wchar_bps": 0.0,
                "io_read_bps": 0.0,
                "io_write_bps": 0.0,
                "server_vram_mib": "",
                "total_gpu_mem_mib": "",
                "total_gpu_used_mib": "",
                "gpu_util_pct": "",
                "observed_hardware_profile": "cpu-only",
                "audio_seconds": audio_seconds,
                "wall_seconds": "",
                "transcription_throughput": "",
                "wer": "",
                "der": "",
                "der_collar_s": args.der_collar,
                "der_skip_overlap": args.der_skip_overlap,
                "wer_normalization": "spanish-friendly",
                "sampling_warning": "",
            }
            if prev_sample is not None:
                dt = float(row["sample_dt_s"])
                if dt > 0:
                    row["io_rchar_bps"] = round(
                        (sample["rchar"] - prev_sample.get("io_rchar_bytes", 0)) / dt, 1
                    )
                    row["io_wchar_bps"] = round(
                        (sample["wchar"] - prev_sample.get("io_wchar_bytes", 0)) / dt, 1
                    )
                    row["io_read_bps"] = round(
                        (sample["read_bytes"] - prev_sample.get("io_read_bytes", 0)) / dt, 1
                    )
                    row["io_write_bps"] = round(
                        (sample["write_bytes"] - prev_sample.get("io_write_bytes", 0)) / dt, 1
                    )

            prev_sample = {
                "elapsed_s": elapsed,
                "cpu_user_s": sample["cpu_user_s"],
                "cpu_system_s": sample["cpu_system_s"],
                "io_rchar_bytes": sample["rchar"],
                "io_wchar_bytes": sample["wchar"],
                "io_read_bytes": sample["read_bytes"],
                "io_write_bytes": sample["write_bytes"],
            }
            samples.append(row)
            stop_event.wait(args.sample_interval)

    sampler = threading.Thread(target=_sample_loop, daemon=True)
    sampler.start()

    # Run the benchmark request
    req_start = time.monotonic()
    try:
        if args.mode == "sse":
            curl_cmd = [
                "curl",
                "-s",
                "-X",
                "POST",
                args.url,
                "-F",
                f"file=@{audio_path}",
                "-F",
                "stream=true",
                "-w",
                "@-",
                "--no-buffer",
                "-o",
                str(response_path),
            ]
        else:
            curl_cmd = [
                "curl",
                "-s",
                "-X",
                "POST",
                args.url,
                "-F",
                f"file=@{audio_path}",
                "-w",
                "time_total=%{time_total}\nhttp_code=%{http_code}",
                "-o",
                str(response_path),
            ]
        result = subprocess.run(curl_cmd, capture_output=True, text=True)
        curl_metrics_path.write_text(result.stdout + result.stderr)
    except Exception as exc:
        curl_metrics_path.write_text(str(exc))

    wall_seconds = time.monotonic() - req_start
    stop_event.set()
    sampler.join()

    throughput = audio_seconds / wall_seconds if wall_seconds > 0 and audio_seconds > 0 else None

    # Backfill aggregate fields into all rows (Backfilled Quality Column)
    for row in samples:
        row["wall_seconds"] = round(wall_seconds, 3)
        row["transcription_throughput"] = round(throughput, 6) if throughput else ""
        row["sampling_warning"] = "true" if sampling_warning else ""

    # Write Resource CSV
    with resource_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESOURCE_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(samples)

    return {"wall_seconds": wall_seconds, "throughput": throughput, "wer": None, "der": None}


def compute_peaks(csv_path: Path) -> dict[str, Any]:
    """Compute peak memory/CPU from a Resource CSV."""
    peaks: dict[str, Any] = {}
    try:
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            pss_values = []
            cpu_values = []
            for row in reader:
                try:
                    pss_values.append(float(row.get("pss_kb", 0) or 0))
                    cpu_values.append(float(row.get("cpu_pct", 0) or 0))
                except ValueError:
                    pass
        if pss_values:
            peaks["peak_pss_kb"] = max(pss_values)
        if cpu_values:
            peaks["peak_cpu_pct"] = max(cpu_values)
    except Exception:
        pass
    return peaks


def _write_summary(args, out_dir: Path, rep_results: list[dict], audio_seconds: float) -> None:
    """Write a human-readable summary.txt."""
    summary_lines = [
        f"audio={args.audio}",
        f"mode={args.mode}",
        f"reps={args.reps}",
        f"url={args.url}",
        f"audio_seconds={audio_seconds:.3f}",
        f"wer_normalization={'spanish-friendly' if args.reference_transcript else ''}",
        f"wall_total_seconds={sum(float(r['wall_seconds']) for r in rep_results):.3f}",
        "",
    ]
    for rep in range(1, args.reps + 1):
        result = rep_results[rep - 1]
        summary_lines.append(f"wall_seconds_rep{rep}={result['wall_seconds']:.3f}")
        if result["throughput"] is not None:
            summary_lines.append(f"transcription_throughput_rep{rep}={result['throughput']:.6f}")
        summary_lines.append(f"[rep {rep} peaks]")
        peaks = compute_peaks(out_dir / f"resource_{rep}.csv")
        for k, v in peaks.items():
            summary_lines.append(f"{k}={v}")
        summary_lines.append("")

    summary_lines.append("[outputs]")
    for rep in range(1, args.reps + 1):
        summary_lines.append(f"resource_csv_rep{rep}={out_dir}/resource_{rep}.csv")
        summary_lines.append(f"response_rep{rep}={out_dir}/response_{rep}.{args.mode}")
        summary_lines.append(f"curl_metrics_rep{rep}={out_dir}/curl_metrics_{rep}.txt")

    summary_text = "\n".join(summary_lines)
    summary_path = out_dir / "summary.txt"
    summary_path.write_text(summary_text)
    print(summary_text)
    print(f"\nSummary written to: {summary_path}")
