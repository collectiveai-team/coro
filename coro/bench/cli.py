"""coro-bench CLI entry point."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from coro.bench.ami import (
    ensure_audio_and_annotations,
    materialize_reference_stms,
    resolve_workload_set,
)
from coro.bench.calibration import DEFAULT_CALIBRATION_MARGIN
from coro.bench.errors import ServerUnreachableError
from coro.bench.quarantine import CircularReferenceError, assert_scorable_reference
from coro.bench.spanish import SPANISH_PRESETS

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
        default=os.environ.get("SERVER_MATCH", "coro"),
    )
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

    spanish = parser.add_argument_group("Spanish workload set (public corpora)")
    spanish.add_argument(
        "--spanish-preset",
        choices=sorted(SPANISH_PRESETS),
        default=None,
        help="Materialise a Spanish Workload Set from freely-licensed public "
        "corpora and benchmark it as a --clips-dir workload. Single-speaker: "
        "WER only, no meaningful DER.",
    )
    spanish.add_argument("--spanish-root", type=Path, default=Path("./spanish-corpora/"))
    spanish.add_argument(
        "--spanish-limit",
        type=int,
        default=None,
        help="Override the preset's items-per-corpus count.",
    )
    spanish.add_argument(
        "--spanish-fetch-plan",
        action="store_true",
        default=False,
        help="Print the download footprint of --spanish-preset and exit.",
    )
    spanish.add_argument(
        "--calibration-margin",
        type=float,
        default=DEFAULT_CALIBRATION_MARGIN,
        help="Two-sided absolute WER band against published figures "
        f"(default {DEFAULT_CALIBRATION_MARGIN}). Exceeding it fails the run.",
    )
    spanish.add_argument(
        "--no-calibration",
        action="store_true",
        default=False,
        help="Report calibration without failing the run on a deviation.",
    )


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


def _validate_reference_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """Reject quarantined references and conflicting workload-set selectors."""
    if args.reference_stm is not None:
        try:
            assert_scorable_reference(args.reference_stm)
        except CircularReferenceError as exc:
            parser.error(str(exc))

    if args.spanish_preset is not None and args.clips_dir is not None:
        parser.error("--spanish-preset is mutually exclusive with --clips-dir.")

    if args.spanish_fetch_plan and args.spanish_preset is None:
        parser.error("--spanish-fetch-plan requires --spanish-preset.")


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

    _apply_defaults(args)

    if args.warmup_audio is not None:
        args.warmup = True

    if args.no_diarization:
        args.server_diar_backend = "none"

    if args.reference_stm is not None and args.audio is None:
        parser.error("--reference-stm requires --audio.")

    if args.audio is not None and args.reference_stm is None and args.subcommand == "quality":
        parser.error("--audio without --reference-stm is not allowed for the 'quality' subcommand.")

    _validate_reference_args(args, parser)


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
    _validate_args(args, parser)
    return args


def _run_performance(args: argparse.Namespace, meetings: list[str]) -> None:
    from coro.bench.ami import get_audio_path
    from coro.bench.data import WARMUP_AUDIO_PATH
    from coro.bench.orchestrate import run_performance_workload
    from coro.bench.report import build_report, render_markdown, render_stdout

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict] = []
    for meeting_id in meetings:
        audio_path = get_audio_path(args.ami_root, meeting_id)
        if audio_path.exists():
            items.append(
                {
                    "item_id": meeting_id,
                    "audio_path": audio_path,
                    "ref_stm_path": None,
                }
            )

    if args.audio is not None:
        items.append(
            {
                "item_id": args.audio.stem,
                "audio_path": args.audio,
                "ref_stm_path": None,
            }
        )

    if args.clips_dir is not None:
        from coro.bench.clips import resolve_clip_items

        items.extend(resolve_clip_items(args.clips_dir))

    base_url = args.server_url or f"http://127.0.0.1:{args.server_port}"

    warmup_audio = args.warmup_audio or WARMUP_AUDIO_PATH

    run_performance_workload(
        items=items,
        base_url=base_url,
        out_dir=out_dir,
        reps=args.reps,
        server_pid=args.server_pid or 1,
        sample_interval=args.sample_interval,
        stream=args.stream,
        warmup_audio=warmup_audio,
    )

    report = build_report(out_dir)
    render_stdout(report)
    (out_dir / "REPORT.md").write_text(render_markdown(report))


def _run_quality(args: argparse.Namespace, meetings: list[str]) -> None:
    from coro.bench.ami import get_audio_path
    from coro.bench.orchestrate import run_workload
    from coro.bench.report import build_report, render_markdown, render_stdout

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ami_root = args.ami_root
    items: list[dict] = []
    for meeting_id in meetings:
        audio_path = get_audio_path(ami_root, meeting_id)
        stm_path = ami_root / "stm" / f"{meeting_id}.ref.stm"
        ref_stm = stm_path if stm_path.exists() else None
        if audio_path.exists():
            items.append(
                {
                    "item_id": meeting_id,
                    "audio_path": audio_path,
                    "ref_stm_path": ref_stm,
                    "audio_seconds": 0.0,
                }
            )

    if args.audio is not None:
        items.append(
            {
                "item_id": args.audio.stem,
                "audio_path": args.audio,
                "ref_stm_path": args.reference_stm,
                "audio_seconds": 0.0,
            }
        )

    if args.clips_dir is not None:
        from coro.bench.clips import resolve_clip_items

        items.extend(resolve_clip_items(args.clips_dir))

    base_url = args.server_url or f"http://127.0.0.1:{args.server_port}"

    run_workload(
        items=items,
        base_url=base_url,
        out_dir=out_dir,
        reps=1,
        subcommand="quality",
        der_collar=args.der_collar,
        der_regions=args.der_regions,
    )

    report = build_report(out_dir)
    render_stdout(report)
    (out_dir / "REPORT.md").write_text(render_markdown(report))


def _run_all(args: argparse.Namespace, meetings: list[str]) -> None:
    from coro.bench.ami import get_audio_path
    from coro.bench.data import WARMUP_AUDIO_PATH
    from coro.bench.orchestrate import run_all_workload
    from coro.bench.report import build_report, render_markdown, render_stdout

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ami_root = args.ami_root
    items: list[dict] = []
    for meeting_id in meetings:
        audio_path = get_audio_path(ami_root, meeting_id)
        stm_path = ami_root / "stm" / f"{meeting_id}.ref.stm"
        ref_stm = stm_path if stm_path.exists() else None
        if audio_path.exists():
            items.append(
                {
                    "item_id": meeting_id,
                    "audio_path": audio_path,
                    "ref_stm_path": ref_stm,
                    "audio_seconds": 0.0,
                }
            )

    if args.audio is not None:
        items.append(
            {
                "item_id": args.audio.stem,
                "audio_path": args.audio,
                "ref_stm_path": args.reference_stm,
                "audio_seconds": 0.0,
            }
        )

    if args.clips_dir is not None:
        from coro.bench.clips import resolve_clip_items

        items.extend(resolve_clip_items(args.clips_dir))

    base_url = args.server_url or f"http://127.0.0.1:{args.server_port}"

    warmup_audio = args.warmup_audio or WARMUP_AUDIO_PATH

    run_all_workload(
        items=items,
        base_url=base_url,
        out_dir=out_dir,
        reps=args.reps,
        server_pid=args.server_pid or 1,
        sample_interval=args.sample_interval,
        der_collar=args.der_collar,
        der_regions=args.der_regions,
        warmup_audio=warmup_audio,
        stream=args.stream,
    )

    report = build_report(out_dir)
    render_stdout(report)
    (out_dir / "REPORT.md").write_text(render_markdown(report))


def _print_spanish_fetch_plan(args: argparse.Namespace) -> None:
    """Print the per-corpus download footprint of the selected Spanish preset."""
    from coro.bench.spanish import preset_fetch_plan

    plans = preset_fetch_plan(args.spanish_preset, items_per_corpus=args.spanish_limit)
    print(f"Spanish preset '{args.spanish_preset}' fetch plan:")
    total = 0
    for plan in plans:
        total += plan.download_bytes
        print(
            f"  {plan.corpus:<12} rows={plan.rows:<5} "
            f"row_groups={plan.row_groups:<3} download≈{plan.download_bytes / 1e6:,.1f} MB"
        )
    print(f"  {'total':<12} download≈{total / 1e6:,.1f} MB (one-time; cached on disk)")


def _prepare_spanish_workload(args: argparse.Namespace) -> None:
    """Materialise the selected Spanish preset into ``args.clips_dir``."""
    from coro.bench.spanish import materialize_spanish_workload_set

    args.clips_dir = materialize_spanish_workload_set(
        args.spanish_preset,
        args.spanish_root,
        items_per_corpus=args.spanish_limit,
        no_download=args.no_download,
    )
    print(f"Spanish workload set materialised at {args.clips_dir}")


def _run_calibration(args: argparse.Namespace) -> bool:
    """Score the run against published WER; return True when it deviates."""
    import json
    from dataclasses import asdict

    from coro.bench.calibration import calibrate_run, model_id_from_manifest, render_calibration

    report = calibrate_run(
        args.out_dir,
        model_id=model_id_from_manifest(args.out_dir),
        margin=args.calibration_margin,
    )
    if not report.outcomes:
        return False

    print(render_calibration(report))
    quality_dir = args.out_dir / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    (quality_dir / "calibration.json").write_text(json.dumps(asdict(report), indent=2))

    if report.failed and not args.no_calibration:
        print(
            "error: Spanish WER calibration deviated beyond the margin; "
            "treat this as a harness fault until proven otherwise.",
            file=sys.stderr,
        )
        return True
    return False


def main() -> None:
    args = parse_args()

    if args.spanish_fetch_plan:
        _print_spanish_fetch_plan(args)
        return

    if args.spanish_preset is not None:
        _prepare_spanish_workload(args)

    # A custom workload (--clips-dir / --audio) suppresses the implicit AMI
    # "sample" default so curated/short-clip runs don't pull AMI meetings.
    has_explicit_ami = bool(args.ami_meetings or args.ami_groups or args.ami_preset)
    has_custom_workload = args.clips_dir is not None or args.audio is not None
    if has_explicit_ami or not has_custom_workload:
        meetings = resolve_workload_set(
            ami_meetings=args.ami_meetings,
            ami_groups=args.ami_groups,
            ami_preset=args.ami_preset,
        )
    else:
        meetings = []

    if meetings:
        ensure_audio_and_annotations(
            meetings,
            args.ami_root,
            no_download=args.no_download,
        )
        materialize_reference_stms(meetings, args.ami_root)

    try:
        if args.subcommand == "performance":
            _run_performance(args, meetings)
        elif args.subcommand == "quality":
            _run_quality(args, meetings)
        else:
            _run_all(args, meetings)
    except ServerUnreachableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    except CircularReferenceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(4)

    if args.subcommand in ("quality", "all") and _run_calibration(args):
        sys.exit(3)


if __name__ == "__main__":
    main()
