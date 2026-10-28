#!/usr/bin/env python3

"""Cut a short AMI clip + rebased reference STM for reliable short-audio benchmarks.

Produces a 16 kHz mono WAV clip (via ffmpeg) and a time-rebased reference STM
windowed from the AMI manual annotations. Pair the outputs with the quality
subcommand's ``--audio`` / ``--reference-stm`` flags:

    coro-bench quality --audio IB4001_180_60.wav --reference-stm IB4001_180_60.ref.stm

All STM logic lives in the library (ami.clip_reference_stm / stm.slice_stm_window).
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

from coro.bench.ami import clip_reference_stm, get_audio_path
from coro.cli_support import invoke, new_app, out_console


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


app = new_app("make_ami_clip", help="Cut a short AMI clip + reference STM.")


@app.default
def _cut(
    meeting_id: str,
    /,
    *,
    start: float,
    ami_root: Path = Path("amicorpus"),
    duration: float = 60.0,
    out_dir: Path = Path("ami-clips"),
) -> None:
    """Cut a short AMI clip + reference STM.

    Args:
        meeting_id: AMI meeting to cut from.
        start: Clip start (seconds).
        ami_root: Root of the AMI corpus.
        duration: Clip length (seconds).
        out_dir: Directory the clip and its Reference STM are written into.

    """
    stem = f"{meeting_id}_{int(start)}_{int(duration)}"
    audio_dst = out_dir / f"{stem}.wav"
    stm_dst = out_dir / f"{stem}.ref.stm"

    cut_audio_clip(get_audio_path(ami_root, meeting_id), audio_dst, start, duration)
    # The clip stem is the benchmark item_id, so the reference session id must
    # match it (the hypothesis STM is keyed by item_id).
    stm_text = clip_reference_stm(ami_root, meeting_id, start, duration, recording_id=stem)
    stm_dst.write_text(stm_text, encoding="utf-8")

    out_console.out(f"wrote {audio_dst}")
    out_console.out(f"wrote {stm_dst} ({len(stm_text.splitlines())} lines)")


def main(argv: Sequence[str] | None = None) -> None:
    """Run the AMI clip cutter command line."""
    invoke(app, argv)


if __name__ == "__main__":
    main()
