#!/usr/bin/env python3

r"""Cut a short clip + diarization-only reference STM from an RTTM + audio pair.

For diarization corpora that ship speaker turns as RTTM but no transcript
(e.g. VoxConverse: https://github.com/joonson/voxconverse, CC-BY-4.0). Produces
a 16 kHz mono WAV clip (via ffmpeg) and a time-rebased diarization-only STM
(stm.DIARIZATION_ONLY_TEXT sentinel). Drop the pair into a ``--clips-dir`` or
pass via ``quality --audio``/``--reference-stm``: DER is scored, WER is omitted.

    python -m coro.bench.utils.make_rttm_clip \
        --audio voxconverse_dev_wav/audio/abjxc.wav \
        --rttm voxconverse/dev/abjxc.rttm \
        --start 0 --duration 120 --out-dir voxc-clips

All STM logic lives in the library (stm.rttm_to_stm / stm.slice_stm_window).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from coro.bench.stm import rttm_to_stm, slice_stm_window
from coro.bench.utils.make_ami_clip import cut_audio_clip


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cut a short clip + diarization-only reference STM from RTTM.",
    )
    parser.add_argument("--audio", type=Path, required=True, help="Source audio file.")
    parser.add_argument("--rttm", type=Path, required=True, help="RTTM speaker turns.")
    parser.add_argument("--start", type=float, default=0.0, help="Clip start (seconds).")
    parser.add_argument("--duration", type=float, required=True, help="Clip length (seconds).")
    parser.add_argument("--out-dir", type=Path, default=Path("rttm-clips"))
    parser.add_argument(
        "--recording-id",
        default=None,
        help="Session id base (default: RTTM filename stem).",
    )
    args = parser.parse_args()

    session = args.recording_id or args.rttm.stem
    # The clip stem is the benchmark item_id, so the reference session id must
    # match it (the hypothesis STM is keyed by item_id).
    stem = f"{session}_{int(args.start)}_{int(args.duration)}"
    audio_dst = args.out_dir / f"{stem}.wav"
    stm_dst = args.out_dir / f"{stem}.ref.stm"

    cut_audio_clip(args.audio, audio_dst, args.start, args.duration)

    stm_dst.parent.mkdir(parents=True, exist_ok=True)
    full_stm = rttm_to_stm(args.rttm.read_text(encoding="utf-8"), stem)
    stm_text = slice_stm_window(
        full_stm,
        args.start,
        args.start + args.duration,
        rebase=True,
        recording_id=stem,
    )
    stm_dst.write_text(stm_text, encoding="utf-8")

    print(f"wrote {audio_dst}")
    print(f"wrote {stm_dst} ({len(stm_text.splitlines())} lines)")


if __name__ == "__main__":
    main()
