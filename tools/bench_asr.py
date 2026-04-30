#!/usr/bin/env python3
"""ASR endpoint benchmark tool."""

import argparse
import os
import subprocess
import sys
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
    pid = int(pids[0])
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        print(f"Error: found PID {pid} but it is not running", file=sys.stderr)
        sys.exit(1)
    return pid


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
