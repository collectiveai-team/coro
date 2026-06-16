"""Tests for the make_rttm_clip diarization-clip maker (VoxConverse-style)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

_RTTM = (
    "SPEAKER abjxc 1 0.50 2.00 <NA> <NA> spk0 <NA> <NA>\n"
    "SPEAKER abjxc 1 3.00 2.00 <NA> <NA> spk1 <NA> <NA>\n"
    "SPEAKER abjxc 1 70.00 5.00 <NA> <NA> spk0 <NA> <NA>\n"
)


def _run(argv: list[str]) -> None:
    from asr_diar_server.bench.utils.make_rttm_clip import main

    with patch.object(sys, "argv", argv):
        main()


def test_make_rttm_clip_writes_windowed_diarization_only_stm(tmp_path: Path):
    rttm = tmp_path / "abjxc.rttm"
    rttm.write_text(_RTTM)
    audio = tmp_path / "abjxc.wav"
    audio.write_bytes(b"RIFF")
    out_dir = tmp_path / "clips"

    # Patch the ffmpeg cut so no external binary / real audio is needed.
    with patch(
        "asr_diar_server.bench.utils.make_rttm_clip.cut_audio_clip"
    ) as cut:
        _run([
            "make_rttm_clip",
            "--audio", str(audio),
            "--rttm", str(rttm),
            "--start", "0",
            "--duration", "60",
            "--out-dir", str(out_dir),
        ])
        cut.assert_called_once()

    stm = (out_dir / "abjxc_0_60.ref.stm").read_text().strip().splitlines()
    # The 70s turn is outside the [0,60) window and must be dropped; the two
    # early turns are kept, rebased, session id == clip stem, sentinel text.
    assert stm == [
        "abjxc_0_60 1 spk0 0.500 2.500 <sd>",
        "abjxc_0_60 1 spk1 3.000 5.000 <sd>",
    ]


def test_make_rttm_clip_uses_recording_id_override(tmp_path: Path):
    rttm = tmp_path / "session.rttm"
    rttm.write_text(_RTTM)
    audio = tmp_path / "session.wav"
    audio.write_bytes(b"RIFF")
    out_dir = tmp_path / "clips"

    with patch("asr_diar_server.bench.utils.make_rttm_clip.cut_audio_clip"):
        _run([
            "make_rttm_clip",
            "--audio", str(audio),
            "--rttm", str(rttm),
            "--start", "0",
            "--duration", "10",
            "--out-dir", str(out_dir),
            "--recording-id", "vox01",
        ])

    assert (out_dir / "vox01_0_10.ref.stm").exists()
