#!/usr/bin/env python3

"""Cut a short AMI clip + rebased reference STM for reliable short-audio benchmarks.

Produces a 16 kHz mono WAV clip (via ffmpeg) and a time-rebased reference STM
windowed from the AMI manual annotations. Pair the outputs with the quality
subcommand's ``--audio`` / ``--reference-stm`` flags:

    coro-bench quality --audio IB4001_180_60.wav --reference-stm IB4001_180_60.ref.stm

All STM logic lives in the library (ami.clip_reference_stm / stm.slice_stm_window).
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from coro.bench.ami import clip_reference_stm, get_audio_path


def cut_audio_clip(src: Path, dst: Path, start: float, duration: float) -> None:
    """Cut a 16 kHz mono WAV clip from ``src`` using ffmpeg."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            str(start),
            "-t",
            str(duration),
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(dst),
        ],
        check=True,
    )


def clip_stem(meeting_id: str, start: float, duration: float) -> str:
    """Return the clip stem, which is also the benchmark Workload Item id."""
    return f"{meeting_id}_{int(start)}_{int(duration)}"


def write_clip_reference(
    ami_root: Path,
    meeting_id: str,
    start: float,
    duration: float,
    out_dir: Path,
) -> Path:
    """Write one clip's rebased Reference STM; return its path.

    Split from the audio cut because the two have opposite lifetimes: the clip
    audio is an immutable fixture that must stay byte-identical across runs,
    while the reference is derived from annotation-parsing code that changes and
    therefore has to be rebuildable in place.
    """
    stem = clip_stem(meeting_id, start, duration)
    stm_dst = out_dir / f"{stem}.ref.stm"
    # The clip stem is the benchmark item_id, so the reference session id must
    # match it (the hypothesis STM is keyed by item_id).
    stm_text = clip_reference_stm(
        ami_root,
        meeting_id,
        start,
        duration,
        recording_id=stem,
    )
    stm_dst.parent.mkdir(parents=True, exist_ok=True)
    stm_dst.write_text(stm_text, encoding="utf-8")
    return stm_dst


def materialize_clip(
    ami_root: Path,
    meeting_id: str,
    start: float,
    duration: float,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Cut one clip and write its rebased Reference STM; return both paths.

    The single unit of AMI clip materialization, shared by the single-meeting
    CLI and the clip Workload Set materializer so neither duplicates the audio
    cutting or the STM windowing.
    """
    stem = clip_stem(meeting_id, start, duration)
    audio_dst = out_dir / f"{stem}.wav"

    cut_audio_clip(get_audio_path(ami_root, meeting_id), audio_dst, start, duration)
    stm_dst = write_clip_reference(ami_root, meeting_id, start, duration, out_dir)
    return audio_dst, stm_dst


def main() -> None:
    parser = argparse.ArgumentParser(description="Cut a short AMI clip + reference STM.")
    parser.add_argument("meeting_id")
    parser.add_argument("--ami-root", type=Path, default=Path("amicorpus"))
    parser.add_argument("--start", type=float, required=True, help="Clip start (seconds).")
    parser.add_argument("--duration", type=float, default=60.0, help="Clip length (seconds).")
    parser.add_argument("--out-dir", type=Path, default=Path("ami-clips"))
    args = parser.parse_args()

    audio_dst, stm_dst = materialize_clip(
        args.ami_root,
        args.meeting_id,
        args.start,
        args.duration,
        args.out_dir,
    )

    print(f"wrote {audio_dst}")
    print(f"wrote {stm_dst} ({len(stm_dst.read_text().splitlines())} lines)")


if __name__ == "__main__":
    main()
