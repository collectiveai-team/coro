"""asr-diar-bench CLI entry point.

The benchmark implementation is migrated from tools/bench_asr.py.
The entry point is registered as ``asr-diar-bench`` in pyproject.toml.

Usage:
    asr-diar-bench audio.wav [mode] [options]
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path


def parse_args(argv=None) -> argparse.Namespace:
    """Parse asr-diar-bench command-line arguments.

    Args:
        argv: Optional argument list (defaults to sys.argv).

    Returns:
        Parsed argparse Namespace.

    """
    parser = argparse.ArgumentParser(
        description="Benchmark an ASR HTTP endpoint across N repetitions.",
        prog="asr-diar-bench",
    )
    parser.add_argument("audio", type=Path, help="Path to audio file.")
    parser.add_argument(
        "mode",
        nargs="?",
        default="json",
        choices=["json", "sse"],
        help="Response mode (default: json).",
    )
    parser.add_argument("--reps", type=int, default=1, help="Number of repetitions.")
    parser.add_argument(
        "--url",
        default=os.environ.get("URL", "http://localhost:8000/v1/audio/transcriptions"),
        help="Endpoint URL.",
    )
    parser.add_argument(
        "--server-pid",
        type=int,
        default=int(os.environ["SERVER_PID"]) if os.environ.get("SERVER_PID") else None,
        help="Server process PID.",
    )
    parser.add_argument(
        "--server-match",
        default=os.environ.get("SERVER_MATCH", "asr-diar-server"),
        help="pgrep pattern to find server PID.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get(
            "OUT_DIR",
            f"/tmp/asr-bench-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        ),
        type=Path,
        help="Output directory.",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=float(os.environ.get("SAMPLE_INTERVAL", "0.25")),
        help="Resource sampling interval in seconds.",
    )
    parser.add_argument(
        "--reference-transcript",
        type=Path,
        help="Reference transcript text file for WER scoring.",
    )
    parser.add_argument(
        "--reference-diarization",
        type=Path,
        help="Reference RTTM file for DER scoring.",
    )
    parser.add_argument(
        "--der-collar",
        type=float,
        default=0.25,
        help="DER collar in seconds (default: 0.25).",
    )
    parser.add_argument(
        "--der-skip-overlap",
        action="store_true",
        help="Skip overlapped speech when scoring DER.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Entry point for the asr-diar-bench command."""
    # Import the full benchmark implementation (migrated from tools/bench_asr.py).
    # The heavy imports (psutil, requests, etc.) live here so importing the CLI
    # module for tests stays lightweight.
    from asr_diar_server.bench.run import run_benchmark

    args = parse_args()
    run_benchmark(args)
