"""AudioInput: upload spooling, temp-file ownership and cleanup."""

from __future__ import annotations

import tempfile
import tracemalloc
from pathlib import Path

import pytest

from coro.audio_input import AudioInput, _suffix_from_filename


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

    assert Path(path).exists()  # noqa: ASYNC240
    await audio.cleanup()

    assert not Path(path).exists()  # noqa: ASYNC240


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


async def _with_peak_traced_memory(awaitable):
    """Await ``awaitable`` under tracemalloc; return its result and the peak bytes traced."""
    tracemalloc.start()
    try:
        result = await awaitable
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


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

    audio, peak = await _with_peak_traced_memory(
        AudioInput.from_upload(_ChunkedUpload(chunk, count=chunk_count))
    )

    try:
        assert audio._data is None, "file-backed input must not retain the encoded bytes"
        assert audio.size == expected_size
        assert Path(await audio.temp_path()).stat().st_size == expected_size  # noqa: ASYNC240
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

    before = set(Path(tempfile.gettempdir()).glob("asr-upload-*"))  # noqa: ASYNC240
    with pytest.raises(OSError, match="upload stream died"):
        await AudioInput.from_upload(_FailingUpload())

    assert set(Path(tempfile.gettempdir()).glob("asr-upload-*")) == before  # noqa: ASYNC240


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
    assert spooled.exists()  # noqa: ASYNC240

    await audio.cleanup()

    assert not spooled.exists()  # noqa: ASYNC240


@pytest.mark.asyncio
async def test_from_path_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        AudioInput.from_path(tmp_path / "absent.wav")
