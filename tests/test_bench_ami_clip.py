"""Tests for the single-meeting AMI clip materializer."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

# One line inside the [180, 240) window and one outside it.
_STM = "IB4001 1 A 185.000 190.000 hello world\nIB4001 1 B 900.000 902.000 out of window\n"


def _fake_cut(src: Path, dst: Path, start: float, duration: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(b"RIFF")


def test_cli_writes_clip_and_window_rebased_reference(tmp_path: Path):
    out_dir = tmp_path / "clips"
    argv = [
        "make_ami_clip",
        "IB4001",
        "--ami-root",
        str(tmp_path / "amicorpus"),
        "--start",
        "180",
        "--duration",
        "60",
        "--out-dir",
        str(out_dir),
    ]

    from coro.bench.utils.make_ami_clip import main

    with (
        patch("coro.bench.utils.make_ami_clip.cut_audio_clip", side_effect=_fake_cut),
        patch("coro.bench.ami.ami_meeting_to_stm", return_value=_STM),
        patch.object(sys, "argv", argv),
    ):
        main()

    assert (out_dir / "IB4001_180_60.wav").exists()

    lines = (out_dir / "IB4001_180_60.ref.stm").read_text().splitlines()
    assert len(lines) == 1
    session, _channel, speaker, start, end, text = lines[0].split(maxsplit=5)
    # The session id is the clip stem so it matches the Hypothesis STM's item_id,
    # and times are rebased onto the cut audio, which starts at 0.
    assert session == "IB4001_180_60"
    assert speaker == "A"
    assert (start, end) == ("5.000", "10.000")
    assert text == "hello world"
