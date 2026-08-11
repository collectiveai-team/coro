"""coro-bench CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from coro.bench.ami import (
    ensure_audio_and_annotations,
    materialize_reference_stms,
    resolve_workload_set,
)
from coro.bench.cli_args import _SAMPLING_SUBCOMMANDS, parse_args
from coro.bench.cli_spanish import (
    prepare_spanish_workload,
    print_spanish_fetch_plan,
    run_calibration,
)
from coro.bench.errors import ServerPidUnresolvedError, ServerUnreachableError
from coro.bench.process_lookup import DEFAULT_SERVER_MATCH
from coro.bench.quarantine import CircularReferenceError
from coro.bench.server_lifecycle import (
    BenchAttachedServer,
    BenchManagedServer,
    ServerHandle,
)

__all__ = ["build_server_handle", "main", "parse_args", "resolve_meetings"]


def _build_items(
    args: argparse.Namespace,
    meetings: list[str],
    *,
    with_references: bool,
) -> list[dict]:
    """Assemble the Workload Set from AMI meetings, --audio, and --clips-dir.

    ``with_references`` attaches a Reference STM to each item; the performance
    subcommand makes no quality claim and therefore carries none.
    """
    from coro.bench.ami import get_audio_path

    items: list[dict] = []
    for meeting_id in meetings:
        audio_path = get_audio_path(args.ami_root, meeting_id)
        if not audio_path.exists():
            continue
        ref_stm = None
        if with_references:
            stm_path = args.ami_root / "stm" / f"{meeting_id}.ref.stm"
            ref_stm = stm_path if stm_path.exists() else None
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
                "ref_stm_path": args.reference_stm if with_references else None,
                "audio_seconds": 0.0,
            }
        )

    if args.clips_dir is not None:
        from coro.bench.clips import resolve_clip_items

        items.extend(resolve_clip_items(args.clips_dir))

    return items


def _render_report(out_dir: Path) -> None:
    """Print the run report to stdout and write ``REPORT.md`` beside the artifacts."""
    from coro.bench.report import build_report, render_markdown, render_stdout

    report = build_report(out_dir)
    render_stdout(report)
    (out_dir / "REPORT.md").write_text(render_markdown(report))


def _require_server_pid(server: ServerHandle) -> int:
    """Return the Server Process Tree root PID, refusing to sample an unrelated one."""
    if server.server_pid is None:
        raise ServerPidUnresolvedError(DEFAULT_SERVER_MATCH)
    return server.server_pid


def _run_performance(
    args: argparse.Namespace,
    meetings: list[str],
    server: ServerHandle,
) -> None:
    from coro.bench.data import WARMUP_AUDIO_PATH
    from coro.bench.orchestrate import run_performance_workload

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    run_performance_workload(
        items=_build_items(args, meetings, with_references=False),
        base_url=server.base_url,
        out_dir=out_dir,
        reps=args.reps,
        server_pid=_require_server_pid(server),
        sample_interval=args.sample_interval,
        cli_args=args.cli_args,
        stream=args.stream,
        warmup_audio=args.warmup_audio or WARMUP_AUDIO_PATH,
    )

    _render_report(out_dir)


def _run_quality(
    args: argparse.Namespace,
    meetings: list[str],
    server: ServerHandle,
) -> None:
    from coro.bench.orchestrate import run_workload

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    run_workload(
        items=_build_items(args, meetings, with_references=True),
        base_url=server.base_url,
        out_dir=out_dir,
        reps=1,
        subcommand="quality",
        cli_args=args.cli_args,
        der_collar=args.der_collar,
        der_regions=args.der_regions,
    )

    _render_report(out_dir)


def _run_all(
    args: argparse.Namespace,
    meetings: list[str],
    server: ServerHandle,
) -> None:
    from coro.bench.data import WARMUP_AUDIO_PATH
    from coro.bench.orchestrate import run_all_workload

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    run_all_workload(
        items=_build_items(args, meetings, with_references=True),
        base_url=server.base_url,
        out_dir=out_dir,
        reps=args.reps,
        server_pid=_require_server_pid(server),
        sample_interval=args.sample_interval,
        cli_args=args.cli_args,
        der_collar=args.der_collar,
        der_regions=args.der_regions,
        warmup_audio=args.warmup_audio or WARMUP_AUDIO_PATH,
        stream=args.stream,
    )

    _render_report(out_dir)


def build_server_handle(args: argparse.Namespace) -> ServerHandle:
    """Build the Bench-Attached or Bench-Managed Server handle for this run.

    Bench-attached runs that sample resources resolve the Server Process Tree
    root from ``--server-pid`` or ``--server-match``; a Bench-Managed Server
    spawns the server itself and therefore already knows its PID.
    """
    from coro.bench.process_lookup import resolve_server_pid

    if args.server_url is not None:
        pid = args.server_pid
        if pid is None and args.subcommand in _SAMPLING_SUBCOMMANDS:
            pid = resolve_server_pid(args.server_match or DEFAULT_SERVER_MATCH)
        return BenchAttachedServer(args.server_url, pid=pid)

    return BenchManagedServer(
        asr_backend=args.server_asr_backend,
        asr_model=args.server_asr_model,
        diar_backend=args.server_diar_backend,
        diar_model=args.server_diar_model,
        pipeline=args.server_pipeline,
        port=args.server_port,
        log_path=args.out_dir / "server.log",
    )


def resolve_meetings(args: argparse.Namespace) -> list[str]:
    """Resolve the AMI Workload Set and materialize its audio and Reference STMs.

    A custom workload (``--clips-dir`` / ``--audio``) suppresses the implicit
    AMI "sample" default so curated/short-clip runs do not pull AMI meetings.
    """
    has_explicit_ami = bool(args.ami_meetings or args.ami_groups or args.ami_preset)
    has_custom_workload = args.clips_dir is not None or args.audio is not None
    if not has_explicit_ami and has_custom_workload:
        return []

    meetings = resolve_workload_set(
        ami_meetings=args.ami_meetings,
        ami_groups=args.ami_groups,
        ami_preset=args.ami_preset,
    )
    if meetings:
        ensure_audio_and_annotations(
            meetings,
            args.ami_root,
            no_download=args.no_download,
        )
        materialize_reference_stms(
            meetings,
            args.ami_root,
            reuse_existing=args.reuse_reference_stms,
        )
    return meetings


def main() -> None:
    args = parse_args()

    if args.spanish_fetch_plan:
        print_spanish_fetch_plan(args)
        return

    # Materialising the preset sets args.clips_dir, which is what suppresses the
    # implicit AMI default inside resolve_meetings — so it has to happen first.
    if args.spanish_preset is not None:
        prepare_spanish_workload(args)

    try:
        meetings = resolve_meetings(args)
        with build_server_handle(args) as server:
            if args.subcommand == "performance":
                _run_performance(args, meetings, server)
            elif args.subcommand == "quality":
                _run_quality(args, meetings, server)
            else:
                _run_all(args, meetings, server)
    except (ServerUnreachableError, ServerPidUnresolvedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    except CircularReferenceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(4)

    if args.subcommand in ("quality", "all") and run_calibration(args):
        sys.exit(3)


if __name__ == "__main__":
    main()
