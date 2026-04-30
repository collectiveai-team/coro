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

    with mem_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        while not stop_event.is_set():
            ts = int(time.time())
            elapsed = int(time.monotonic() - start)

            # RAM via ps
            ps = subprocess.run(
                ["ps", "-o", "rss=,vsz=", "-p", str(pid)],
                capture_output=True, text=True,
            )
            parts = ps.stdout.split()
            rss = int(parts[0]) if len(parts) >= 1 else 0
            vsz = int(parts[1]) if len(parts) >= 2 else 0

            server_vram = 0
            total_gpu_mem = 0
            total_gpu_used = 0
            gpu_util = 0.0

            # GPU via nvidia-smi (skip silently if not available)
            nsmi = subprocess.run(
                ["which", "nvidia-smi"], capture_output=True
            )
            if nsmi.returncode == 0:
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

                # total GPU memory
                mem_out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True,
                ).stdout
                total_gpu_mem = sum(
                    int(v.strip()) for v in mem_out.splitlines() if v.strip().isdigit()
                )

                # total GPU used
                used_out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True,
                ).stdout
                total_gpu_used = sum(
                    int(v.strip()) for v in used_out.splitlines() if v.strip().isdigit()
                )

                # GPU utilization (average across GPUs)
                util_out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True,
                ).stdout
                util_vals = [
                    float(v.strip()) for v in util_out.splitlines() if v.strip().replace(".", "").isdigit()
                ]
                gpu_util = sum(util_vals) / len(util_vals) if util_vals else 0.0

            writer.writerow({
                "ts_epoch": ts,
                "elapsed_s": elapsed,
                "rss_kb": rss,
                "vsz_kb": vsz,
                "server_vram_mib": server_vram,
                "total_gpu_mem_mib": total_gpu_mem,
                "total_gpu_used_mib": total_gpu_used,
                "gpu_util_pct": f"{gpu_util:.1f}",
            })
            f.flush()

            stop_event.wait(sample_interval)


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
