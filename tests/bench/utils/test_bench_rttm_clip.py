"""Tests for the make_rttm_clip diarization-clip maker (VoxConverse-style).

The clip is cut by the real ffmpeg, which CI installs as a runtime dependency.
"""

from __future__ import annotations

import wave
from pathlib import Path

from coro.bench.utils.make_rttm_clip import main
from support.corpus import write_silent_wav

_RTTM = (
    "SPEAKER abjxc 1 0.50 2.00 <NA> <NA> spk0 <NA> <NA>\n"
    "SPEAKER abjxc 1 3.00 2.00 <NA> <NA> spk1 <NA> <NA>\n"
    "SPEAKER abjxc 1 70.00 5.00 <NA> <NA> spk0 <NA> <NA>\n"
)


def _recording(tmp_path: Path, stem: str) -> tuple[Path, Path]:
    rttm = tmp_path / f"{stem}.rttm"
    rttm.write_text(_RTTM)
    audio = tmp_path / f"{stem}.wav"
    write_silent_wav(audio, seconds=2.0)
    return audio, rttm


def _wav_format(path: Path) -> tuple[int, int]:
    """The channel count and sample rate of a WAV file."""
    with wave.open(str(path)) as clip:
        return clip.getnchannels(), clip.getframerate()


def _stm_lines(path: Path) -> list[str]:
    return path.read_text().strip().splitlines()


def test_make_rttm_clip_writes_windowed_diarization_only_stm(tmp_path: Path):
    audio, rttm = _recording(tmp_path, "abjxc")
    out_dir = tmp_path / "clips"

    main(
        [
            "--audio",
            str(audio),
            "--rttm",
            str(rttm),
            "--start",
            "0",
            "--duration",
            "60",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert _wav_format(out_dir / "abjxc_0_60.wav") == (1, 16000)
    stm = _stm_lines(out_dir / "abjxc_0_60.ref.stm")
    # The 70s turn is outside the [0,60) window and must be dropped; the two
    # early turns are kept, rebased, session id == clip stem, sentinel text.
    assert stm == [
        "abjxc_0_60 1 spk0 0.500 2.500 <sd>",
        "abjxc_0_60 1 spk1 3.000 5.000 <sd>",
    ]


def test_make_rttm_clip_uses_recording_id_override(tmp_path: Path):
    audio, rttm = _recording(tmp_path, "session")
    out_dir = tmp_path / "clips"

    main(
        [
            "--audio",
            str(audio),
            "--rttm",
            str(rttm),
            "--start",
            "0",
            "--duration",
            "10",
            "--out-dir",
            str(out_dir),
            "--recording-id",
            "vox01",
        ]
    )

    assert (out_dir / "vox01_0_10.wav").exists()
    assert (out_dir / "vox01_0_10.ref.stm").read_text().startswith("vox01_0_10 1 spk0 0.500")
