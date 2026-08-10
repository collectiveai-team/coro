"""Argument definition and validation for the coro-bench CLI.

Kept separate from the run orchestration in :mod:`coro.bench.cli` so the flag
surface can be read and validated on its own.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from coro.bench.process_lookup import DEFAULT_SERVER_MATCH

_MANAGED_FLAGS = {
    "server_asr_backend",
    "server_asr_model",
    "server_diar_backend",
    "server_diar_model",
    "server_pipeline",
    "server_port",
    "no_diarization",
}

_ATTACHED_FLAGS = ("server_pid", "server_match")

# Subcommands that sample the Server Process Tree and therefore need a real PID.
_SAMPLING_SUBCOMMANDS = frozenset({"performance", "all"})


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
        "--sample-interval",
        type=float,
        default=float(os.environ.get("SAMPLE_INTERVAL", "0.25")),
    )
    parser.add_argument("--ami-meetings", nargs="+", default=[])
    parser.add_argument(
        "--ami-groups",
        nargs="+",
        default=[],
        choices=["IB", "IN", "ES", "IS", "TS", "EN"],
    )
    parser.add_argument(
        "--ami-preset",
        choices=["sample", "eval", "full"],
        default=None,
    )
    parser.add_argument("--ami-root", type=Path, default=Path("./amicorpus/"))
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--reuse-reference-stms",
        action="store_true",
        help="Reuse Reference STM files already present under <ami-root>/stm instead "
        "of regenerating them. Opt-in: reused files are frozen against whatever the "
        "STM builder looked like when they were first written.",
    )

    managed = parser.add_argument_group("bench-managed server")
    managed.add_argument("--server-asr-backend", default=None)
    managed.add_argument("--server-asr-model", default=None)
    managed.add_argument("--server-diar-backend", default=None)
    managed.add_argument("--server-diar-model", default=None)
    managed.add_argument("--server-pipeline", default=None)
    managed.add_argument("--no-diarization", action="store_true", default=None)
    managed.add_argument("--server-port", type=int, default=None)

    attached = parser.add_argument_group("bench-attached server")
    attached.add_argument("--server-url", type=str, default=None)
    attached.add_argument(
        "--server-pid",
        type=int,
        default=int(os.environ["SERVER_PID"]) if os.environ.get("SERVER_PID") else None,
        help="Root PID of the Server Process Tree to sample. Bench-attached only; "
        "a Bench-Managed Server reports its own PID.",
    )
    attached.add_argument(
        "--server-match",
        default=os.environ.get("SERVER_MATCH") or None,
        help="Command-line substring identifying the Server Process Tree when "
        f"--server-pid is not given (default: {DEFAULT_SERVER_MATCH!r}). "
        "Bench-attached only.",
    )

    parser.add_argument("--warmup", action="store_true", default=False)
    parser.add_argument("--warmup-audio", type=Path, default=None)
    parser.add_argument("--audio", type=Path, default=None)
    parser.add_argument("--reference-stm", type=Path, default=None)
    parser.add_argument(
        "--clips-dir",
        type=Path,
        default=None,
        help="Directory of (<stem>.wav, <stem>.ref.stm) pairs to benchmark as a "
        "short-clip / curated workload (e.g. make_ami_clip output).",
    )
    parser.add_argument("--der-collar", type=float, default=0.0)
    parser.add_argument("--der-regions", choices=["all", "nooverlap", "single"], default="all")
    parser.add_argument("--stream", action="store_true", default=False)


def _apply_defaults(args: argparse.Namespace) -> None:
    defaults = {
        "server_asr_backend": "faster-whisper",
        "server_asr_model": "openai/whisper-medium",
        "server_diar_backend": "nemo",
        "server_diar_model": "nvidia/diar_streaming_sortformer_4spk-v2",
        "server_pipeline": "full-memory",
        "server_port": 0,
        "no_diarization": False,
    }
    for flag, default in defaults.items():
        if getattr(args, flag) is None:
            setattr(args, flag, default)


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if getattr(args, "stream", False) and args.subcommand == "quality":
        parser.error("--stream is not allowed for the 'quality' subcommand.")

    has_attached = args.server_url is not None
    has_managed_explicit = any(getattr(args, flag) is not None for flag in _MANAGED_FLAGS)

    if has_attached and has_managed_explicit:
        parser.error(
            "--server-url is mutually exclusive with bench-managed server flags "
            "(--server-asr-backend, --server-asr-model, --server-diar-backend, "
            "--server-diar-model, --server-pipeline, --server-port, --no-diarization)."
        )

    if not has_attached:
        for flag in _ATTACHED_FLAGS:
            if getattr(args, flag) is not None:
                parser.error(
                    f"--{flag.replace('_', '-')} only applies to a bench-attached server; "
                    "a bench-managed server reports its own PID. Pass --server-url to "
                    "attach to an already-running server."
                )

    _apply_defaults(args)

    if args.warmup_audio is not None:
        args.warmup = True

    if args.no_diarization:
        args.server_diar_backend = "none"

    if args.reference_stm is not None and args.audio is None:
        parser.error("--reference-stm requires --audio.")

    if args.audio is not None and args.reference_stm is None and args.subcommand == "quality":
        parser.error("--audio without --reference-stm is not allowed for the 'quality' subcommand.")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark an ASR HTTP endpoint.",
        prog="coro-bench",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    for name in ("quality", "performance", "all"):
        sub = subparsers.add_parser(name)
        _add_shared_flags(sub)

    args = parser.parse_args(argv)
    # Recorded verbatim in the run manifest so a result can be re-run as issued.
    args.cli_args = list(argv) if argv is not None else sys.argv[1:]
    _validate_args(args, parser)
    return args
