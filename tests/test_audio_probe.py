"""ffprobe measures upload duration before any decoding (ADR 0024).

Real ffprobe on real files: the point is the container handling, which a mock
would only restate.
"""

from __future__ import annotations

import subprocess

import pytest

from coro.audio_probe import probe_duration_seconds
from support.factories import make_wav

pytestmark = pytest.mark.asyncio


async def test_a_wav_reports_its_declared_duration(tmp_path):
    path = tmp_path / "one_second.wav"
    path.write_bytes(make_wav(frames=16000))
    assert await probe_duration_seconds(str(path)) == pytest.approx(1.0, abs=1e-3)


async def test_a_webm_without_declared_duration_falls_back_to_its_packets(tmp_path):
    # Written to a pipe, the muxer cannot seek back to record the duration:
    # the shape of a browser MediaRecorder upload.
    webm = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
            "-c:a",
            "libopus",
            "-f",
            "webm",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    ).stdout
    path = tmp_path / "recording.webm"
    path.write_bytes(webm)
    declared = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert declared == "N/A"
    assert await probe_duration_seconds(str(path)) == pytest.approx(3.0, abs=0.05)


async def test_a_file_that_is_not_media_is_unknown(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not audio")
    assert await probe_duration_seconds(str(path)) is None
