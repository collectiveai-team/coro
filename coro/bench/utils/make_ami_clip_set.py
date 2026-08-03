#!/usr/bin/env python3

"""Materialize a whole AMI group as a clip Workload Set in one reproducible run.

Defaults to the measurement workload of ADR 0008: the 30 meetings of the AMI
**ES** group, each cut to a 10-minute clip with a rebased Reference STM.

ES is load-bearing, not arbitrary. Every ES meeting has exactly four
participants, matching the default Diarization Model Selection's hard
four-speaker cap. The non-scenario groups (IB, IN, EN) can exceed it, forcing
the diarizer to merge two real speakers and inflating the Speaker Attribution
Gap for a reason no segmentation change can recover. The built-in ``sample``
preset (IB4001 + IN1001) must not be used for that measurement.

    python -m coro.bench.utils.make_ami_clip_set --ami-root ./amicorpus
    coro-bench quality --clips-dir ./amicorpus/clips --server-url http://127.0.0.1:8123

Re-running is idempotent: meetings whose clip and Reference STM already exist
are skipped, and audio for them is never re-downloaded. That is what lets the
baseline (issue 03) and the post-dedup re-measurement (issue 05) run against
provably identical audio.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from coro.bench.ami import AMI_GROUPS, ensure_audio_and_annotations
from coro.bench.utils.make_ami_clip import clip_stem, materialize_clip

DEFAULT_GROUP = "ES"
DEFAULT_DURATION = 600.0
DEFAULT_START = 0.0


def clip_is_materialized(out_dir: Path, meeting_id: str, start: float, duration: float) -> bool:
    """Report whether both the clip audio and its Reference STM already exist."""
    stem = clip_stem(meeting_id, start, duration)
    return (out_dir / f"{stem}.wav").exists() and (out_dir / f"{stem}.ref.stm").exists()


def materialize_clip_set(
    ami_root: Path,
    meetings: list[str],
    *,
    out_dir: Path,
    start: float = DEFAULT_START,
    duration: float = DEFAULT_DURATION,
    no_download: bool = False,
) -> list[str]:
    """Materialize a clip Workload Set; return the meetings actually processed.

    Already-materialized meetings are filtered out *before* the download step,
    so a completed set can be re-run without the full-length source audio being
    present at all.
    """
    pending = [m for m in meetings if not clip_is_materialized(out_dir, m, start, duration)]
    if not pending:
        return []

    ensure_audio_and_annotations(pending, ami_root, no_download=no_download)

    out_dir.mkdir(parents=True, exist_ok=True)
    for meeting_id in pending:
        audio_dst, stm_dst = materialize_clip(ami_root, meeting_id, start, duration, out_dir)
        print(f"wrote {audio_dst} and {stm_dst}")
    return pending


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize an AMI group as a clip workload set (default: ES, 10 min).",
    )
    parser.add_argument("--ami-root", type=Path, default=Path("amicorpus"))
    parser.add_argument(
        "--group",
        choices=sorted(AMI_GROUPS),
        default=DEFAULT_GROUP,
        help="AMI group to materialize (default: %(default)s).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help="Clip length in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=DEFAULT_START,
        help="Clip start offset in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Clip output directory (default: <ami-root>/clips).",
    )
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir is not None else args.ami_root / "clips"
    meetings = AMI_GROUPS[args.group]

    processed = materialize_clip_set(
        args.ami_root,
        meetings,
        out_dir=out_dir,
        start=args.start,
        duration=args.duration,
        no_download=args.no_download,
    )

    skipped = len(meetings) - len(processed)
    print(
        f"{args.group}: {len(processed)} materialized, {skipped} already present "
        f"({len(meetings)} total) in {out_dir}"
    )


if __name__ == "__main__":
    main()
