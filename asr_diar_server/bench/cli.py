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

_MANAGED_FLAGS = {
    "server_asr_backend",
    "server_asr_model",
    "server_diar_backend",
    "server_diar_model",
    "server_pipeline",
    "server_port",
    "no_diarization",
}


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

    parser.add_argument("--audio", type=Path, default=None)
    parser.add_argument("--reference-stm", type=Path, default=None)
    parser.add_argument("--der-collar", type=float, default=0.0)
    parser.add_argument(
        "--der-regions", choices=["all", "nooverlap", "single"], default="all"
    )


def _apply_defaults(args: argparse.Namespace) -> None:
    defaults = {
        "server_asr_backend": "whisperlivekit",
        "server_asr_model": "openai/whisper-medium",
        "server_diar_backend": "whisperlivekit",
        "server_diar_model": "nvidia/diar_sortformer_4spk-v1",
        "server_pipeline": "full-memory",
        "server_port": 0,
        "no_diarization": False,
    }
    for flag, default in defaults.items():
        if getattr(args, flag) is None:
            setattr(args, flag, default)


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    has_attached = args.server_url is not None
    has_managed_explicit = any(
        getattr(args, flag) is not None for flag in _MANAGED_FLAGS
    )

    if has_attached and has_managed_explicit:
        parser.error(
            "--server-url is mutually exclusive with bench-managed server flags "
            "(--server-asr-backend, --server-asr-model, --server-diar-backend, "
            "--server-diar-model, --server-pipeline, --server-port, --no-diarization)."
        )

    _apply_defaults(args)

    if args.no_diarization:
        args.server_diar_backend = "none"

    if args.reference_stm is not None and args.audio is None:
        parser.error("--reference-stm requires --audio.")

    if args.audio is not None and args.reference_stm is None and args.subcommand == "quality":
        parser.error(
            "--audio without --reference-stm is not allowed for the 'quality' subcommand."
        )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark an ASR HTTP endpoint.",
        prog="asr-diar-bench",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    for name in ("quality", "performance", "all"):
        sub = subparsers.add_parser(name)
        _add_shared_flags(sub)

    args = parser.parse_args(argv)
    _validate_args(args, parser)
    return args


def _run_performance(args: argparse.Namespace, meetings: list[str]) -> None:
    from asr_diar_server.bench.ami import get_audio_path
    from asr_diar_server.bench.orchestrate import run_performance_workload

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict] = []
    for meeting_id in meetings:
        audio_path = get_audio_path(args.ami_root, meeting_id)
        if audio_path.exists():
            items.append({
                "item_id": meeting_id,
                "audio_path": audio_path,
                "ref_stm_path": None,
            })

    if args.audio is not None:
        items.append({
            "item_id": args.audio.stem,
            "audio_path": args.audio,
            "ref_stm_path": None,
        })

    base_url = args.server_url or f"http://127.0.0.1:{args.server_port}"

    run_performance_workload(
        items=items,
        base_url=base_url,
        out_dir=out_dir,
        reps=args.reps,
        server_pid=args.server_pid or 1,
        sample_interval=args.sample_interval,
    )

    import json
    summary_path = out_dir / "performance" / "summary.json"
    if summary_path.exists():
        print(json.dumps(json.loads(summary_path.read_text()), indent=2))


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

    if args.subcommand == "performance":
        _run_performance(args, meetings)
    else:
        print(f"{args.subcommand} not yet implemented")


if __name__ == "__main__":
    main()
