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
            "ffmpeg", "-v", "error", "-y",
            "-ss", str(start), "-t", str(duration),
            "-i", str(src),
            "-ac", "1", "-ar", "16000",
            str(dst),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cut a short AMI clip + reference STM.")
    parser.add_argument("meeting_id")
    parser.add_argument("--ami-root", type=Path, default=Path("amicorpus"))
    parser.add_argument("--start", type=float, required=True, help="Clip start (seconds).")
    parser.add_argument("--duration", type=float, default=60.0, help="Clip length (seconds).")
    parser.add_argument("--out-dir", type=Path, default=Path("ami-clips"))
    args = parser.parse_args()

    stem = f"{args.meeting_id}_{int(args.start)}_{int(args.duration)}"
    audio_dst = args.out_dir / f"{stem}.wav"
    stm_dst = args.out_dir / f"{stem}.ref.stm"

    cut_audio_clip(
        get_audio_path(args.ami_root, args.meeting_id),
        audio_dst,
        args.start,
        args.duration,
    )
    # The clip stem is the benchmark item_id, so the reference session id must
    # match it (the hypothesis STM is keyed by item_id).
    stm_text = clip_reference_stm(
        args.ami_root, args.meeting_id, args.start, args.duration, recording_id=stem,
    )
    stm_dst.write_text(stm_text, encoding="utf-8")

    print(f"wrote {audio_dst}")
    print(f"wrote {stm_dst} ({len(stm_text.splitlines())} lines)")


if __name__ == "__main__":
    main()
