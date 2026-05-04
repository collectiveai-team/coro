"""asr-diar-bench CLI entry point."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from asr_diar_server.bench.ami import (
    ensure_audio_and_annotations,
    materialize_reference_stms,
    resolve_workload_set,
)


def _add_shared_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--out-dir",
        default=os.environ.get(
            "OUT_DIR",
            f"/tmp/asr-bench-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        ),
        type=Path,
    )
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument(
        "--server-pid",
        type=int,
        default=int(os.environ["SERVER_PID"]) if os.environ.get("SERVER_PID") else None,
    )
    parser.add_argument(
        "--server-match",
        default=os.environ.get("SERVER_MATCH", "asr-diar-server"),
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=float(os.environ.get("SAMPLE_INTERVAL", "0.25")),
    )
    parser.add_argument("--ami-meetings", nargs="+", default=[])
    parser.add_argument(
        "--ami-groups", nargs="+", default=[], choices=["IB", "IN", "ES", "IS", "TS", "EN"],
    )
    parser.add_argument(
        "--ami-preset", choices=["sample", "eval", "full"], default=None,
    )
    parser.add_argument("--ami-root", type=Path, default=Path("./amicorpus/"))
    parser.add_argument("--no-download", action="store_true")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark an ASR HTTP endpoint.",
        prog="asr-diar-bench",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    for name in ("quality", "performance", "all"):
        sub = subparsers.add_parser(name)
        _add_shared_flags(sub)

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    meetings = resolve_workload_set(
        ami_meetings=args.ami_meetings,
        ami_groups=args.ami_groups,
        ami_preset=args.ami_preset,
    )
    ensure_audio_and_annotations(
        meetings, args.ami_root, no_download=args.no_download,
    )
    materialize_reference_stms(meetings, args.ami_root)
    print(f"{args.subcommand} not yet implemented")


if __name__ == "__main__":
    main()
