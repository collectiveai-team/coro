"""Audio Module — ffmpeg conversion, PCM streaming, upload spooling, and constants.

This module owns all audio IO concerns so that ffmpeg and byte-streaming
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
import tempfile
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

SAMPLE_RATE: int = 16000
"""Canonical audio sample rate in Hz."""

BYTES_PER_SAMPLE: int = 2
"""Bytes per PCM sample (16-bit little-endian mono)."""


class AudioInput:
    """Package-owned uploaded audio representation with cleanup ownership."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._temp_path: str | None = None

    @classmethod
    async def from_upload(cls, upload: Any) -> AudioInput:
        chunks: list[bytes] = []
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return cls(b"".join(chunks))

    async def read_bytes(self) -> bytes:
        return self._data

    async def temp_path(self) -> str:
        if self._temp_path is None:
            fd, path = tempfile.mkstemp(prefix="asr-upload-", suffix=".audio")
            with os.fdopen(fd, "wb") as tmp:
                tmp.write(self._data)
            self._temp_path = path
        return self._temp_path

    async def cleanup(self) -> None:
        if self._temp_path is not None:
            with contextlib.suppress(FileNotFoundError):
                Path(self._temp_path).unlink()
            self._temp_path = None

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


async def convert_to_pcm_bytes(audio_bytes: bytes) -> bytes:
    """Convert any audio format to PCM s16le mono 16 kHz using ffmpeg (full-memory).

    Args:
        audio_bytes: Raw audio data in any format supported by ffmpeg.

    Returns:
        Raw PCM bytes (s16le mono 16 kHz).

    Raises:
        ValueError: If ffmpeg returns a non-zero exit code.

    """
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-i",
        "pipe:0",
        *_FFMPEG_PCM_ARGS,
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(input=audio_bytes)
    if proc.returncode != 0:
        raise ValueError(f"Audio conversion failed: {stderr.decode().strip()}")
    return stdout


async def stream_pcm_from_file(path: str, chunk_seconds: float = 1.0) -> AsyncIterator[bytes]:
    """Stream PCM chunks from a file path via ffmpeg.

    Args:
        path: Filesystem path to the audio file.
        chunk_seconds: Target chunk duration in seconds.

    Yields:
        PCM byte chunks (s16le mono 16 kHz), aligned to 2-byte boundaries.

    Raises:
        ValueError: If ffmpeg returns a non-zero exit code.

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

    try:
        while True:
            chunk = await proc.stdout.read(target_bytes) if proc.stdout else b""
            if not chunk:
                break
            pending += chunk
            aligned_len = len(pending) - (len(pending) % 2)
            while aligned_len >= target_bytes:
                yield pending[:target_bytes]
                pending = pending[target_bytes:]
                aligned_len = len(pending) - (len(pending) % 2)

        returncode = await proc.wait()
        await stderr_task
        if returncode != 0:
            stderr = b"".join(stderr_chunks)
            raise ValueError(f"Audio conversion failed: {stderr.decode().strip()}")

        if len(pending) >= 2:
            yield pending[: len(pending) - (len(pending) % 2)]

    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        if not stderr_task.done():
            stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stderr_task
