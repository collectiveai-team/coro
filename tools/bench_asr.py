#!/usr/bin/env python3
"""ASR endpoint benchmark tool."""

import argparse
import os
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
    parser.add_argument("reps", nargs="?", type=int, default=1,
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


def main() -> None:
    args = parse_args()

    audio = Path(args.audio)
    if not audio.is_file():
        print(f"Error: audio file not found: {audio}", file=sys.stderr)
        sys.exit(1)

    print(f"audio={audio}")
    print(f"mode={args.mode}")
    print(f"reps={args.reps}")
    print(f"url={args.url}")
    print(f"out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
