"""Audio Module — ffmpeg conversion, PCM streaming, and audio constants.

This module owns all audio or video IO concerns so that ffmpeg and byte-streaming
behaviour do not pollute API routers or core transformations. Upload spooling
lives in :mod:`coro.audio_input`; :class:`AudioInput` is re-exported here.

Public surface:
    SAMPLE_RATE          — canonical 16 kHz.
    BYTES_PER_SAMPLE     — 2 (16-bit little-endian).
    AudioInput           — an uploaded file and the temp file that backs it.
    iter_aligned_pcm_chunks — synchronous aligned-chunk iterator.
    convert_to_pcm_bytes — async full-memory ffmpeg conversion from bytes.
    convert_path_to_pcm_bytes — async full-memory ffmpeg conversion from a path.
    stream_pcm_from_file — async generator that streams PCM chunks from a path.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Iterator

from coro.audio_input import (
    DEFAULT_MEDIA_SUFFIX,
    AudioInput,
    _spool_to_temp,
    _unlink_quietly,
)

__all__ = [
    "BYTES_PER_SAMPLE",
    "DEFAULT_MEDIA_SUFFIX",
    "SAMPLE_RATE",
    "AudioConversionError",
    "AudioInput",
    "convert_path_to_pcm_bytes",
    "convert_to_pcm_bytes",
    "iter_aligned_pcm_chunks",
    "stream_pcm_from_file",
]

# MARK: Audio Constants
SAMPLE_RATE: int = 16000
"""Canonical audio sample rate in Hz."""

BYTES_PER_SAMPLE: int = 2
"""Bytes per PCM sample (16-bit little-endian mono)."""

_STDERR_READ_BYTES = 4096
_STDERR_CAPTURE_LIMIT = 65536


# MARK: Errors
class AudioConversionError(ValueError):
    """Raised when ffmpeg cannot decode the input into PCM.

    Signals a *client* problem (unsupported/corrupt/silent media) so the API
    boundary can map it to a 400 instead of a generic 500. Subclasses
    ``ValueError`` to stay backward-compatible with broad ``except`` callers.
    """


# MARK: FFmpeg Configuration
_FFMPEG_PCM_ARGS = (
    "-f",
    "s16le",
    "-acodec",
    "pcm_s16le",
    "-ar",
    str(SAMPLE_RATE),
    "-ac",
    "1",
    "-loglevel",
    "error",
)


def _conversion_error(stderr: bytes, *, empty_output: bool = False) -> AudioConversionError:
    """Build a uniform ffmpeg-failure error from captured stderr.

    ``empty_output`` distinguishes a clean exit that yielded no PCM (e.g. an
    input with no decodable audio stream) from a non-zero ffmpeg exit.
    """
    detail = stderr.decode().strip()
    if empty_output:
        return AudioConversionError(
            f"Audio conversion produced no PCM output (no decodable audio stream?): {detail}"
        )
    return AudioConversionError(f"Audio conversion failed: {detail}")


# MARK: PCM Chunking
def iter_aligned_pcm_chunks(
    byte_chunks: Iterator[bytes],
    target_bytes: int,
) -> Iterator[bytes]:
    """Yield PCM chunks aligned to 16-bit sample boundaries.

    Args:
        byte_chunks: An iterator of raw byte strings (arbitrary size).
        target_bytes: Target chunk size in bytes.  Rounded down to an even
            number internally to maintain 16-bit alignment.

    Yields:
        Non-empty byte strings, each with an even byte length, each <=
        the (even-adjusted) target.

    """
    target = max(2, target_bytes - (target_bytes % 2))
    pending = b""

    for chunk in byte_chunks:
        if not chunk:
            continue
        pending += chunk
        while len(pending) >= target:
            yield pending[:target]
            pending = pending[target:]

    # Flush remainder aligned to 2-byte boundary.
    if len(pending) >= 2:
        flush_len = len(pending) - (len(pending) % 2)
        if flush_len > 0:
            yield pending[:flush_len]


# MARK: FFmpeg Conversion
async def _spawn_ffmpeg(path: str) -> asyncio.subprocess.Process:
    """Start ffmpeg decoding ``path`` to canonical PCM on its stdout."""
    return await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-i",
        path,
        *_FFMPEG_PCM_ARGS,
        "pipe:1",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


async def convert_path_to_pcm_bytes(path: str) -> bytes:
    """Convert an audio or video file to PCM s16le mono 16 kHz using ffmpeg.

    Decodes from a seekable path (not ``pipe:0``) so ffmpeg can probe container
    formats whose index lives at the end of the stream (e.g. MP4 ``moov``
    atoms). ffmpeg auto-detects the container from the file content, so no
    extension hint is required.

    Args:
        path: Filesystem path to the audio or video file.

    Returns:
        Raw PCM bytes (s16le mono 16 kHz).

    Raises:
        AudioConversionError: If ffmpeg fails, or succeeds but decodes no audio
            (e.g. the input has no decodable audio stream).

    """
    proc = await _spawn_ffmpeg(path)
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise _conversion_error(stderr)
    if not stdout:
        raise _conversion_error(stderr, empty_output=True)
    return stdout


async def convert_to_pcm_bytes(audio_bytes: bytes) -> bytes:
    """Convert in-memory audio or video bytes to PCM s16le mono 16 kHz.

    Spools to a temp file and delegates to :func:`convert_path_to_pcm_bytes`.
    Callers that already hold a path should use that directly rather than
    reading the file in just to have it written back out.

    Args:
        audio_bytes: Raw audio or video data in any format supported by ffmpeg.

    Returns:
        Raw PCM bytes (s16le mono 16 kHz).

    Raises:
        AudioConversionError: If ffmpeg fails, or succeeds but decodes no audio
            (e.g. the input has no decodable audio stream).

    """
    tmp_path = await asyncio.to_thread(
        _spool_to_temp, audio_bytes, prefix="asr-conv-", suffix=DEFAULT_MEDIA_SUFFIX
    )
    try:
        return await convert_path_to_pcm_bytes(tmp_path)
    finally:
        await asyncio.to_thread(_unlink_quietly, tmp_path)


# MARK: FFmpeg Streaming
class _PcmAligner:
    """Re-cut arbitrary reads into fixed-size chunks on 16-bit sample boundaries."""

    def __init__(self, target_bytes: int) -> None:
        self._target = target_bytes
        self._pending = b""

    def feed(self, data: bytes) -> Iterator[bytes]:
        """Buffer ``data`` and yield every complete ``target_bytes`` chunk."""
        self._pending += data
        while len(self._pending) >= self._target:
            yield self._pending[: self._target]
            self._pending = self._pending[self._target :]

    def tail(self) -> bytes:
        """Return the buffered remainder trimmed to whole samples, possibly empty."""
        return self._pending[: len(self._pending) - (len(self._pending) % 2)]


async def _read_until_eof(stream: asyncio.StreamReader | None, size: int) -> AsyncIterator[bytes]:
    """Yield reads of up to ``size`` bytes until ``stream`` reaches EOF."""
    if stream is None:
        return
    while chunk := await stream.read(size):
        yield chunk


async def _capture_stderr(stream: asyncio.StreamReader | None, sink: list[bytes]) -> None:
    """Drain ffmpeg's stderr so it never blocks, keeping the first 64 KiB for errors."""
    total = 0
    async for chunk in _read_until_eof(stream, _STDERR_READ_BYTES):
        if total < _STDERR_CAPTURE_LIMIT:
            sink.append(chunk[: _STDERR_CAPTURE_LIMIT - total])
            total += len(sink[-1])


async def _reap(proc: asyncio.subprocess.Process, stderr_task: asyncio.Task[None]) -> None:
    """Kill ffmpeg if it is still running and stop draining its stderr."""
    if proc.returncode is None:
        proc.kill()
        await proc.wait()
    if not stderr_task.done():
        stderr_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stderr_task


async def stream_pcm_from_file(path: str, chunk_seconds: float = 1.0) -> AsyncIterator[bytes]:
    """Stream PCM chunks from an audio or video file path via ffmpeg.

    Args:
        path: Filesystem path to the audio or video file.
        chunk_seconds: Target chunk duration in seconds.

    Yields:
        PCM byte chunks (s16le mono 16 kHz), aligned to 2-byte boundaries.

    Raises:
        AudioConversionError: If ffmpeg fails, or succeeds but decodes no audio.

    """
    target_bytes = max(2, int(SAMPLE_RATE * BYTES_PER_SAMPLE * chunk_seconds))
    target_bytes -= target_bytes % 2

    proc = await _spawn_ffmpeg(path)
    stderr_chunks: list[bytes] = []
    stderr_task = asyncio.create_task(_capture_stderr(proc.stderr, stderr_chunks))
    aligner = _PcmAligner(target_bytes)
    produced = False

    try:
        async for data in _read_until_eof(proc.stdout, target_bytes):
            for chunk in aligner.feed(data):
                yield chunk
                produced = True

        returncode = await proc.wait()
        await stderr_task
        if returncode != 0:
            raise _conversion_error(b"".join(stderr_chunks))

        if tail := aligner.tail():
            yield tail
            produced = True

        if not produced:
            raise _conversion_error(b"".join(stderr_chunks), empty_output=True)

    finally:
        await _reap(proc, stderr_task)
