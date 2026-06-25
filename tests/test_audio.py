"""Cycle 8: Audio module — aligned PCM chunking and edge cases.

Most tests use small in-memory byte streams. The video conversion regression
uses a generated MP4 fixture because the bug depends on ffmpeg container
probing against non-seekable input.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from coro.audio import (
    SAMPLE_RATE,
    AudioInput,
    _suffix_from_filename,
    convert_to_pcm_bytes,
    iter_aligned_pcm_chunks,
    stream_pcm_from_file,
)

# ---------------------------------------------------------------------------
# iter_aligned_pcm_chunks
# ---------------------------------------------------------------------------


def _pcm_bytes(n_samples: int) -> bytes:
    """Produce n_samples 16-bit little-endian samples (all zeros)."""
    return b"\x00\x00" * n_samples


def test_chunks_are_aligned_to_two_bytes():
    """Every yielded chunk has an even byte length (16-bit PCM alignment)."""
    pcm = _pcm_bytes(100)
    chunks = list(iter_aligned_pcm_chunks(iter([pcm]), target_bytes=30))
    for chunk in chunks:
        assert len(chunk) % 2 == 0, f"Unaligned chunk: {len(chunk)} bytes"


def test_total_bytes_preserved():
    """Total bytes across all chunks equals the input byte count."""
    pcm = _pcm_bytes(200)
    total_in = len(pcm)
    chunks = list(iter_aligned_pcm_chunks(iter([pcm]), target_bytes=64))
    total_out = sum(len(c) for c in chunks)
    assert total_out == total_in


def test_no_chunks_from_empty_input():
    """Empty input yields no chunks."""
    chunks = list(iter_aligned_pcm_chunks(iter([b""]), target_bytes=64))
    assert chunks == []


def test_single_sample_less_than_target_yielded_as_one_chunk():
    """Input smaller than target_bytes is yielded as a single chunk."""
    pcm = _pcm_bytes(4)  # 8 bytes < any reasonable target
    chunks = list(iter_aligned_pcm_chunks(iter([pcm]), target_bytes=64))
    assert len(chunks) == 1
    assert chunks[0] == pcm


def test_chunks_respect_target_bytes_upper_bound():
    """No chunk exceeds target_bytes."""
    pcm = _pcm_bytes(1000)
    target = 64
    chunks = list(iter_aligned_pcm_chunks(iter([pcm]), target_bytes=target))
    for chunk in chunks:
        assert len(chunk) <= target


def test_sample_rate_constant():
    """SAMPLE_RATE is 16000 Hz."""
    assert SAMPLE_RATE == 16000


class _FakeUpload:
    def __init__(self, chunks: list[bytes], filename: str | None = None) -> None:
        self._chunks = list(chunks)
        self.filename = filename

    async def read(self, _size: int = -1) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


@pytest.mark.asyncio
async def test_audio_input_reads_upload_bytes():
    audio = await AudioInput.from_upload(_FakeUpload([b"abc", b"def"]))

    assert await audio.read_bytes() == b"abcdef"


@pytest.mark.asyncio
async def test_audio_input_temp_path_is_removed_on_cleanup():
    audio = await AudioInput.from_upload(_FakeUpload([b"audio"]))
    path = await audio.temp_path()

    assert Path(path).exists()
    await audio.cleanup()

    assert not Path(path).exists()


@pytest.mark.asyncio
async def test_audio_input_temp_path_preserves_upload_suffix():
    audio = await AudioInput.from_upload(_FakeUpload([b"video"], filename="clip.mp4"))
    path = await audio.temp_path()

    try:
        assert Path(path).suffix == ".mp4"
    finally:
        await audio.cleanup()


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("clip.mp4", ".mp4"),
        ("CLIP.WEBM", ".WEBM"),
        ("archive.tar.gz", ".gz"),
        (None, ".media"),
        ("noextension", ".media"),
        ("weird name.m p4", ".media"),  # space -> not a real extension
        ("danger.mp4\n", ".media"),  # control char
        ("x." + "y" * 40, ".media"),  # absurdly long suffix
    ],
)
def test_suffix_from_filename_sanitizes_untrusted_input(filename, expected):
    """Only short, plain-alphanumeric extensions survive; the rest fall back."""
    assert _suffix_from_filename(filename) == expected


@pytest.mark.asyncio
async def test_convert_to_pcm_bytes_rejects_empty_ffmpeg_output(monkeypatch):
    """ffmpeg exiting 0 with no PCM is a failure, not silent empty success."""

    class _FakeProc:
        returncode = 0

        async def communicate(self, _input=None):
            return b"", b"silently produced nothing"

    async def _fake_exec(*_args, **_kwargs):
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    with pytest.raises(ValueError, match="no decodable audio stream"):
        await convert_to_pcm_bytes(b"whatever")


@pytest.mark.asyncio
async def test_convert_to_pcm_bytes_raises_on_nonzero_exit(monkeypatch):
    """A non-zero ffmpeg exit surfaces the stderr as a conversion failure."""

    class _FakeProc:
        returncode = 1

        async def communicate(self, _input=None):
            return b"", b"Invalid data found when processing input"

    async def _fake_exec(*_args, **_kwargs):
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    with pytest.raises(ValueError, match="Audio conversion failed: Invalid data"):
        await convert_to_pcm_bytes(b"whatever")


class _FakeStream:
    """Minimal async stream reader yielding queued chunks then EOF."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def read(self, _size: int = -1) -> bytes:
        return self._chunks.pop(0) if self._chunks else b""


@pytest.mark.asyncio
async def test_stream_pcm_from_file_rejects_empty_output(monkeypatch):
    """A clean ffmpeg exit that streamed no PCM raises instead of yielding nothing."""

    class _FakeProc:
        def __init__(self) -> None:
            self.stdout = _FakeStream([])  # no PCM ever produced
            self.stderr = _FakeStream([b"no decodable stream here"])
            self.returncode: int | None = None

        async def wait(self) -> int:
            self.returncode = 0
            return 0

        def kill(self) -> None:
            self.returncode = 0

    async def _fake_exec(*_args, **_kwargs):
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    with pytest.raises(ValueError, match="no decodable audio stream"):
        async for _chunk in stream_pcm_from_file("/nonexistent.wav"):
            pass


@pytest.mark.asyncio
async def test_convert_to_pcm_bytes_decodes_video_container(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required for video conversion regression coverage")

    video_path = tmp_path / "sample.mp4"
    subprocess.run(  # noqa: S603
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=30:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-shortest",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            str(video_path),
        ],
        check=True,
    )

    pcm = await convert_to_pcm_bytes(video_path.read_bytes())

    assert len(pcm) > SAMPLE_RATE * 2
