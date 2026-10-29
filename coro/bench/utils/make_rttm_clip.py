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

from collections.abc import Sequence
from pathlib import Path

from coro.bench.stm import rttm_to_stm, slice_stm_window
from coro.bench.utils.make_ami_clip import cut_audio_clip
from coro.cli_support import invoke, new_app, out_console

app = new_app("make_rttm_clip", help="Cut a short clip + diarization-only reference STM from RTTM.")


@app.default
def _cut(
    *,
    audio: Path,
    rttm: Path,
    duration: float,
    start: float = 0.0,
    out_dir: Path = Path("rttm-clips"),
    recording_id: str | None = None,
) -> None:
    """Cut a short clip + diarization-only reference STM from RTTM.

    Args:
        audio: Source audio file.
        rttm: RTTM speaker turns.
        duration: Clip length (seconds).
        start: Clip start (seconds).
        out_dir: Directory the clip and its Reference STM are written into.
        recording_id: Session id base (default: RTTM filename stem).

    """
    # The clip stem is the benchmark item_id, so the reference session id must
    # match it (the hypothesis STM is keyed by item_id).
    stem = f"{recording_id or rttm.stem}_{int(start)}_{int(duration)}"
    audio_dst = out_dir / f"{stem}.wav"
    stm_dst = out_dir / f"{stem}.ref.stm"

    cut_audio_clip(audio, audio_dst, start, duration)
    line_count = write_clip_stm(rttm, stm_dst, stem, start, duration)

    out_console.out(f"wrote {audio_dst}")
    out_console.out(f"wrote {stm_dst} ({line_count} lines)")


def write_clip_stm(rttm: Path, stm_dst: Path, stem: str, start: float, duration: float) -> int:
    """Write the diarization-only Reference STM of one clip window.

    Args:
        rttm: RTTM speaker turns of the whole recording.
        stm_dst: Path of the STM to write.
        stem: Session id of the clip.
        start: Clip start (seconds).
        duration: Clip length (seconds).

    Returns:
        The number of STM lines written.

    """
    full_stm = rttm_to_stm(rttm.read_text(encoding="utf-8"), stem)
    stm_text = slice_stm_window(full_stm, start, start + duration, rebase=True, recording_id=stem)
    stm_dst.parent.mkdir(parents=True, exist_ok=True)
    stm_dst.write_text(stm_text, encoding="utf-8")
    return len(stm_text.splitlines())


def main(argv: Sequence[str] | None = None) -> None:
    """Run the RTTM clip cutter command line."""
    invoke(app, argv)


if __name__ == "__main__":
    main()
