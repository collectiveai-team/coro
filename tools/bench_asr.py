#!/usr/bin/env python3
"""ASR endpoint benchmark tool."""

import argparse
import csv
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark an ASR HTTP endpoint across N repetitions."
    )
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("mode", nargs="?", default="json", choices=["json", "sse"],
                        help="Response mode (default: json)")
    parser.add_argument("--reps", type=int, default=1,
                        help="Number of repetitions (default: 1)")
    parser.add_argument(
        "--url",
        default=os.environ.get("URL", "http://localhost:8000/v1/audio/transcriptions"),
        help="Endpoint URL",
    )
    parser.add_argument(
        "--server-pid",
        type=int,
        default=int(os.environ["SERVER_PID"]) if os.environ.get("SERVER_PID") else None,
        help="Server process PID",
    )
    parser.add_argument(
        "--server-match",
        default=os.environ.get("SERVER_MATCH", "custom_server.py"),
        help="pgrep pattern to find server PID",
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get(
            "OUT_DIR",
            f"/tmp/asr-bench-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        ),
        help="Output directory",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=float(os.environ.get("SAMPLE_INTERVAL", "1")),
        help="Memory sampling interval in seconds (default: 1)",
    )
    return parser.parse_args()


def find_server_pid(server_pid: int | None, server_match: str) -> int:
    if server_pid is not None:
        try:
            os.kill(server_pid, 0)  # signal 0 = existence check
        except PermissionError:
            pass  # process exists but owned by another user
        except ProcessLookupError:
            print(f"Error: server PID {server_pid} is not running", file=sys.stderr)
            sys.exit(1)
        return server_pid

    result = subprocess.run(
        ["pgrep", "-f", server_match],
        capture_output=True,
        text=True,
    )
    pids = result.stdout.strip().splitlines()
    if not pids:
        print(
            f"Error: could not find server process matching '{server_match}'. "
            "Set --server-pid or --server-match.",
            file=sys.stderr,
        )
        sys.exit(1)
    if len(pids) > 1:
        print(
            f"Warning: found {len(pids)} processes matching '{server_match}', "
            f"using PID {pids[0]}. Use --server-pid to disambiguate.",
            file=sys.stderr,
        )
    pid = int(pids[0])
    try:
        os.kill(pid, 0)
    except PermissionError:
        pass  # process exists but owned by another user
    except ProcessLookupError:
        print(f"Error: found PID {pid} but it is not running", file=sys.stderr)
        sys.exit(1)
    return pid


def memory_monitor(
    pid: int,
    mem_csv: Path,
    stop_event: threading.Event,
    sample_interval: float,
) -> None:
    fieldnames = [
        "ts_epoch", "elapsed_s", "rss_kb", "vsz_kb",
        "server_vram_mib", "total_gpu_mem_mib", "total_gpu_used_mib", "gpu_util_pct",
    ]
    start = time.monotonic()

    has_nvidia_smi = subprocess.run(["which", "nvidia-smi"], capture_output=True).returncode == 0

    with mem_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        while not stop_event.is_set():
            try:
                ts = int(time.time())
                elapsed = round(time.monotonic() - start, 3)

                # RAM via ps
                ps = subprocess.run(
                    ["ps", "-o", "rss=,vsz=", "-p", str(pid)],
                    capture_output=True, text=True,
                )
                parts = ps.stdout.split()
                if not parts:
                    print(f"Warning: process {pid} not found in ps output (may have exited)", file=sys.stderr)
                    rss, vsz = 0, 0
                else:
                    rss = int(parts[0]) if len(parts) >= 1 else 0
                    vsz = int(parts[1]) if len(parts) >= 2 else 0

                server_vram = 0
                total_gpu_mem = 0
                total_gpu_used = 0
                gpu_util = 0.0

                # GPU via nvidia-smi (skip silently if not available)
                if has_nvidia_smi:
                    # per-process VRAM
                    vram_out = subprocess.run(
                        ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True,
                    ).stdout
                    for line in vram_out.splitlines():
                        cols = [c.strip() for c in line.split(",")]
                        if len(cols) == 2 and cols[0] == str(pid):
                            server_vram += int(cols[1])

                    # total GPU memory, used, and utilization in one call
                    gpu_out = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.total,memory.used,utilization.gpu",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True,
                    ).stdout
                    util_vals = []
                    for line in gpu_out.splitlines():
                        parts_gpu = [p.strip() for p in line.split(",")]
                        if len(parts_gpu) == 3:
                            try:
                                total_gpu_mem += int(parts_gpu[0])
                                total_gpu_used += int(parts_gpu[1])
                                util_vals.append(float(parts_gpu[2]))
                            except ValueError:
                                pass
                    gpu_util = sum(util_vals) / len(util_vals) if util_vals else 0.0

                writer.writerow({
                    "ts_epoch": ts,
                    "elapsed_s": elapsed,
                    "rss_kb": rss,
                    "vsz_kb": vsz,
                    "server_vram_mib": server_vram,
                    "total_gpu_mem_mib": total_gpu_mem,
                    "total_gpu_used_mib": total_gpu_used,
                    "gpu_util_pct": round(gpu_util, 1),
                })
                f.flush()
            except Exception as e:
                print(f"Warning: memory_monitor sampling error: {e}", file=sys.stderr)

            stop_event.wait(sample_interval)


def run_rep(
    rep: int,
    audio: Path,
    mode: str,
    url: str,
    pid: int,
    out_dir: Path,
    sample_interval: float,
) -> float:
    mem_csv = out_dir / f"memory_{rep}.csv"
    resp_out = out_dir / f"response_{rep}.{mode}"
    curl_metrics = out_dir / f"curl_metrics_{rep}.txt"
    time_metrics = out_dir / f"time_metrics_{rep}.txt"

    print(f"\n=== Repetition {rep} ===")

    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=memory_monitor,
        args=(pid, mem_csv, stop_event, sample_interval),
        daemon=True,
    )
    monitor_thread.start()

    curl_write_out = (
        "time_namelookup=%{time_namelookup}\\n"
        "time_connect=%{time_connect}\\n"
        "time_starttransfer=%{time_starttransfer}\\n"
        "time_total=%{time_total}\\n"
        "http_code=%{http_code}\\n"
    )

    if mode == "sse":
        curl_cmd = [
            "/usr/bin/time", "-v", "-o", str(time_metrics),
            "curl", "-N", "-sS",
            "-X", "POST", url,
            "-F", f"file=@{audio}",
            "-F", "model=whisper-1",
            "-F", "stream=true",
            "-w", curl_write_out,
            "-o", str(resp_out),
        ]
    else:
        curl_cmd = [
            "/usr/bin/time", "-v", "-o", str(time_metrics),
            "curl", "-sS",
            "-X", "POST", url,
            "-F", f"file=@{audio}",
            "-F", "model=whisper-1",
            "-w", curl_write_out,
            "-o", str(resp_out),
        ]

    t_start = time.perf_counter_ns()
    result = subprocess.run(curl_cmd, capture_output=False, text=True,
                            stdout=open(curl_metrics, "w"))
    t_end = time.perf_counter_ns()

    stop_event.set()
    monitor_thread.join()

    wall_seconds = (t_end - t_start) / 1_000_000_000
    print(f"wall_seconds_rep{rep}={wall_seconds:.3f}")

    if mode == "sse" and resp_out.exists():
        text = resp_out.read_text(errors="replace")
        print(f"[sse event counts rep {rep}]")
        print(f"progress_events={text.count('transcript.progress')}")
        print(f"delta_events={text.count('transcript.text.delta')}")
        print(f"done_events={text.count('transcript.text.done')}")
        print(f"done_markers={text.count('[DONE]')}")

    return wall_seconds


def main() -> None:
    args = parse_args()

    audio = Path(args.audio)
    if not audio.is_file():
        print(f"Error: audio file not found: {audio}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pid = find_server_pid(args.server_pid, args.server_match)

    print(f"server_pid={pid}")
    print(f"audio={audio}")
    print(f"mode={args.mode}")
    print(f"reps={args.reps}")
    print(f"url={args.url}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
