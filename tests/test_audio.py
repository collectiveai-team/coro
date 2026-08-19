"""Cycle 8: Audio module — aligned PCM chunking and edge cases.

Most tests use small in-memory byte streams. The video conversion regression
uses a generated MP4 fixture because the bug depends on ffmpeg container
probing against non-seekable input.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
import tracemalloc
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


class _ChunkedUpload:
    """Yield one shared chunk `count` times, never allocating the whole upload.

    Reusing a single buffer keeps the fixture itself out of the memory
    measurement, so the peak reflects only what `from_upload` retains.
    """

    def __init__(self, chunk: bytes, count: int, filename: str | None = None) -> None:
        self._chunk = chunk
        self._remaining = count
        self.filename = filename

    async def read(self, _size: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        self._remaining -= 1
        return self._chunk


@pytest.mark.asyncio
async def test_audio_input_from_upload_does_not_hold_the_upload_in_memory():
    """A large upload is spooled chunk-by-chunk rather than accumulated and joined.

    Collecting the chunks into a list and joining them cost roughly twice the
    upload size in resident RAM before any decoding began, so a multi-gigabyte
    upload could exhaust the host before a model ever ran. Peak traced memory
    must stay near one chunk, not near the upload size.
    """
    chunk = b"\x01" * (1024 * 1024)
    chunk_count = 16
    expected_size = len(chunk) * chunk_count

    tracemalloc.start()
    try:
        audio = await AudioInput.from_upload(_ChunkedUpload(chunk, count=chunk_count))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    try:
        assert audio._data is None, "file-backed input must not retain the encoded bytes"
        assert audio.size == expected_size
        assert Path(await audio.temp_path()).stat().st_size == expected_size
        assert peak < 4 * len(chunk), (
            f"peak traced memory {peak} bytes for a {expected_size}-byte upload "
            "suggests the upload was accumulated in RAM"
        )
    finally:
        await audio.cleanup()


@pytest.mark.asyncio
async def test_audio_input_size_is_available_without_reading_the_upload():
    """`size` is served from the spool counter, so callers need not materialise bytes."""
    audio = await AudioInput.from_upload(_FakeUpload([b"abc", b"defgh"]))

    try:
        assert audio.size == 8
    finally:
        await audio.cleanup()


@pytest.mark.asyncio
async def test_audio_input_read_bytes_reads_back_a_spooled_upload():
    """A file-backed input can still serve its bytes, by reading them from disk."""
    audio = await AudioInput.from_upload(_FakeUpload([b"abc", b"def"]))

    try:
        assert await audio.read_bytes() == b"abcdef"
    finally:
        await audio.cleanup()


@pytest.mark.asyncio
async def test_audio_input_rejects_use_after_cleanup():
    """A file-backed input has no bytes left after cleanup, and says so explicitly."""
    audio = await AudioInput.from_upload(_FakeUpload([b"abc"]))
    await audio.cleanup()

    with pytest.raises(RuntimeError, match="already been cleaned up"):
        await audio.read_bytes()
    with pytest.raises(RuntimeError, match="already been cleaned up"):
        await audio.temp_path()


@pytest.mark.asyncio
async def test_audio_input_from_upload_leaves_no_file_when_spooling_fails():
    """A read error mid-spool must not leave the partial temp file behind."""

    class _FailingUpload:
        filename = "clip.mp4"

        def __init__(self) -> None:
            self.calls = 0

        async def read(self, _size: int = -1) -> bytes:
            self.calls += 1
            if self.calls == 1:
                return b"partial"
            raise OSError("upload stream died")

    before = set(Path(tempfile.gettempdir()).glob("asr-upload-*"))
    with pytest.raises(OSError, match="upload stream died"):
        await AudioInput.from_upload(_FailingUpload())

    assert set(Path(tempfile.gettempdir()).glob("asr-upload-*")) == before


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
    # Not skipped when ffmpeg is missing: it is a runtime dependency, so its
    # absence is a broken environment, not an excuse to drop the only coverage
    # of video-container decoding.
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"

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


# ---------------------------------------------------------------------------
# Referencing a file without owning it
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_from_path_cleanup_leaves_the_referenced_file_alone(tmp_path):
    """The offline command depends on this: both pipelines clean up in a finally."""
    source = tmp_path / "input.wav"
    source.write_bytes(b"not really audio")

    audio = AudioInput.from_path(source)
    await audio.cleanup()

    assert source.exists()


@pytest.mark.asyncio
async def test_from_path_exposes_the_file_without_copying_it(tmp_path):
    source = tmp_path / "input.wav"
    source.write_bytes(b"payload")

    audio = AudioInput.from_path(source)

    assert await audio.temp_path() == str(source)
    assert await audio.read_bytes() == b"payload"
    assert audio.size == len(b"payload")


@pytest.mark.asyncio
async def test_cleanup_still_removes_a_temp_file_this_instance_created(tmp_path):
    audio = AudioInput(b"payload", filename="clip.wav")
    spooled = Path(await audio.temp_path())
    assert spooled.exists()

    await audio.cleanup()

    assert not spooled.exists()


@pytest.mark.asyncio
async def test_from_path_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        AudioInput.from_path(tmp_path / "absent.wav")
