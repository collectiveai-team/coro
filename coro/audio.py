"""Audio Module — ffmpeg conversion, PCM streaming, upload spooling, and constants.

This module owns all audio or video IO concerns so that ffmpeg and byte-streaming
behaviour do not pollute API routers or core transformations.

Public surface:
    SAMPLE_RATE          — canonical 16 kHz.
    BYTES_PER_SAMPLE     — 2 (16-bit little-endian).
    iter_aligned_pcm_chunks — synchronous aligned-chunk iterator.
    convert_to_pcm_bytes — async full-memory ffmpeg conversion.
    stream_pcm_from_file — async generator that streams PCM chunks from a path.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import tempfile
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

# MARK: Audio Constants
SAMPLE_RATE: int = 16000
"""Canonical audio sample rate in Hz."""

BYTES_PER_SAMPLE: int = 2
"""Bytes per PCM sample (16-bit little-endian mono)."""

DEFAULT_MEDIA_SUFFIX: str = ".media"
"""Generic container-hint suffix used when an upload has no usable extension."""

_SAFE_SUFFIX = re.compile(r"\A\.[A-Za-z0-9][A-Za-z0-9]{0,11}\Z")
"""A short, alphanumeric file extension (e.g. ``.mp4``, ``.webm``, ``.m4a``)."""


# MARK: Errors
class AudioConversionError(ValueError):
    """Raised when ffmpeg cannot decode the input into PCM.

    Signals a *client* problem (unsupported/corrupt/silent media) so the API
    boundary can map it to a 400 instead of a generic 500. Subclasses
    ``ValueError`` to stay backward-compatible with broad ``except`` callers.
    """


# MARK: Temp File Spooling
def _suffix_from_filename(filename: str | None) -> str:
    """Derive a safe temp-file suffix from a (client-controlled) upload filename.

    Only short, plain-alphanumeric extensions are honoured; anything else
    (missing, extensionless, spaces, unicode, over-long, path-like) falls back
    to ``DEFAULT_MEDIA_SUFFIX`` so untrusted input never shapes the temp filename.
    """
    if not filename:
        return DEFAULT_MEDIA_SUFFIX
    suffix = Path(filename).suffix
    return suffix if _SAFE_SUFFIX.match(suffix) else DEFAULT_MEDIA_SUFFIX


def _spool_to_temp(data: bytes, *, prefix: str, suffix: str) -> str:
    """Write bytes to a uniquely-named temp file and return its path.

    The caller owns the returned path and is responsible for unlinking it.
    """
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    with os.fdopen(fd, "wb") as tmp:
        tmp.write(data)
    return path


# MARK: Audio Input
class AudioInput:
    """Package-owned uploaded audio or video representation with cleanup ownership."""

    def __init__(self, data: bytes, filename: str | None = None) -> None:
        self._data = data
        self._temp_path: str | None = None
        self._filename = filename

    @classmethod
    async def from_upload(cls, upload: Any) -> AudioInput:
        chunks: list[bytes] = []
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return cls(b"".join(chunks), filename=getattr(upload, "filename", None))

    async def read_bytes(self) -> bytes:
        return self._data

    async def temp_path(self) -> str:
        if self._temp_path is None:
            suffix = _suffix_from_filename(self._filename)
            self._temp_path = _spool_to_temp(self._data, prefix="asr-upload-", suffix=suffix)
        return self._temp_path

    async def cleanup(self) -> None:
        if self._temp_path is not None:
            with contextlib.suppress(FileNotFoundError):
                Path(self._temp_path).unlink()
            self._temp_path = None


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
async def convert_to_pcm_bytes(audio_bytes: bytes) -> bytes:
    """Convert any audio or video format to PCM s16le mono 16 kHz using ffmpeg.

    Decodes through a seekable temporary file (not ``pipe:0``) so ffmpeg can
    probe container formats whose index lives at the end of the stream (e.g.
    MP4 ``moov`` atoms). ffmpeg auto-detects the container from the file
    content, so no extension hint is required.

    Args:
        audio_bytes: Raw audio or video data in any format supported by ffmpeg.

    Returns:
        Raw PCM bytes (s16le mono 16 kHz).

    Raises:
        AudioConversionError: If ffmpeg fails, or succeeds but decodes no audio
            (e.g. the input has no decodable audio stream).

    """
    tmp_path = _spool_to_temp(audio_bytes, prefix="asr-conv-", suffix=DEFAULT_MEDIA_SUFFIX)
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-i",
            tmp_path,
            *_FFMPEG_PCM_ARGS,
            "pipe:1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise _conversion_error(stderr)
        if not stdout:
            raise _conversion_error(stderr, empty_output=True)
        return stdout
    finally:
        with contextlib.suppress(FileNotFoundError):
            Path(tmp_path).unlink()


# MARK: FFmpeg Streaming
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

    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-i",
        path,
        *_FFMPEG_PCM_ARGS,
        "pipe:1",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stderr_chunks: list[bytes] = []

    async def _drain_stderr():
        if not proc.stderr:
            return
        total = 0
        while True:
            chunk = await proc.stderr.read(4096)
            if not chunk:
                break
            if total < 65536:
                stderr_chunks.append(chunk[: 65536 - total])
                total += len(stderr_chunks[-1])

    stderr_task = asyncio.create_task(_drain_stderr())
    pending = b""
    produced = False

    try:
        while True:
            chunk = await proc.stdout.read(target_bytes) if proc.stdout else b""
            if not chunk:
                break
            pending += chunk
            aligned_len = len(pending) - (len(pending) % 2)
            while aligned_len >= target_bytes:
                yield pending[:target_bytes]
                produced = True
                pending = pending[target_bytes:]
                aligned_len = len(pending) - (len(pending) % 2)

        returncode = await proc.wait()
        await stderr_task
        if returncode != 0:
            raise _conversion_error(b"".join(stderr_chunks))

        if len(pending) >= 2:
            yield pending[: len(pending) - (len(pending) % 2)]
            produced = True

        if not produced:
            raise _conversion_error(b"".join(stderr_chunks), empty_output=True)

    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        if not stderr_task.done():
            stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stderr_task
