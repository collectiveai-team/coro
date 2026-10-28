"""Audio Input — an uploaded audio or video file and the temp file that backs it.

Split from :mod:`coro.audio`, which re-exports :class:`AudioInput`: this module
owns upload spooling and temp-file cleanup, that one owns ffmpeg decoding.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import tempfile
from pathlib import Path
from typing import Protocol

DEFAULT_MEDIA_SUFFIX: str = ".media"
"""Generic container-hint suffix used when an upload has no usable extension."""

_SAFE_SUFFIX = re.compile(r"\A\.[A-Za-z0-9][A-Za-z0-9]{0,11}\Z")
"""A short, alphanumeric file extension (e.g. ``.mp4``, ``.webm``, ``.m4a``)."""

_UPLOAD_CHUNK_BYTES: int = 1024 * 1024
"""Read granularity when spooling an upload to disk; also the peak RAM it costs."""


class UploadReader(Protocol):
    """An upload that can be read in chunks, such as a Starlette ``UploadFile``.

    A ``filename`` attribute is optional and read with ``getattr``.
    """

    async def read(self, size: int = -1, /) -> bytes:
        """Return up to ``size`` bytes, or ``b""`` at the end of the upload."""
        ...


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


def _unlink_quietly(path: str) -> None:
    """Delete ``path``, ignoring a file that is already gone."""
    with contextlib.suppress(FileNotFoundError):
        Path(path).unlink()


# MARK: Audio Input
class AudioInput:
    """Package-owned uploaded audio or video representation with cleanup ownership.

    Two backings exist:

    * **In-memory** — constructed directly from ``bytes`` (raw-body routes,
      warmup, tests). ``temp_path`` spools a copy on demand.
    * **File-backed** — constructed by :meth:`from_upload`, which streams the
      upload straight to a temp file so the encoded bytes are never fully
      resident.

    A third, **referenced** backing is produced by :meth:`from_path`, which
    points at a file the caller already owns.

    Cleanup deletes only files this instance created. That distinction is
    load-bearing: both pipelines call ``cleanup()`` in a ``finally``, so without
    it, transcribing a local file in place would delete the user's input.
    """

    def __init__(self, data: bytes, filename: str | None = None) -> None:
        self._data: bytes | None = data
        self._temp_path: str | None = None
        self._owns_temp_path = False
        self._filename = filename
        self._size = len(data)

    @classmethod
    def _file_backed(cls, path: str, *, filename: str | None, size: int) -> AudioInput:
        """Wrap an already-spooled upload without holding its bytes in memory."""
        audio = cls(b"", filename=filename)
        audio._data = None
        audio._temp_path = path
        audio._owns_temp_path = True
        audio._size = size
        return audio

    @classmethod
    def from_path(cls, path: str | os.PathLike[str]) -> AudioInput:
        """Wrap an existing file by reference, without taking ownership of it.

        The file is neither copied nor read: pipelines decode straight from the
        path. :meth:`cleanup` will not unlink it, so running against a local
        file is never destructive.

        Args:
            path: Filesystem path to an existing audio or video file.

        Returns:
            An instance backed by, but not owning, ``path``.

        Raises:
            FileNotFoundError: If ``path`` does not exist.

        """
        resolved = Path(path)
        size = resolved.stat().st_size
        audio = cls(b"", filename=resolved.name)
        audio._data = None
        audio._temp_path = str(resolved)
        audio._owns_temp_path = False
        audio._size = size
        return audio

    @classmethod
    async def from_upload(cls, upload: UploadReader) -> AudioInput:
        """Spool an upload to a temp file one chunk at a time.

        Peak memory is one ``_UPLOAD_CHUNK_BYTES`` chunk regardless of upload
        size. Accumulating the chunks and joining them instead cost roughly
        twice the upload size in resident RAM before any decoding started,
        which made multi-gigabyte uploads fail on hosts that could otherwise
        have transcribed them.

        Args:
            upload: Any object exposing ``async read(size)`` and, optionally, a
                ``filename`` attribute (e.g. a Starlette ``UploadFile``).

        Returns:
            A file-backed instance owning the spooled temp file.

        """
        filename: str | None = getattr(upload, "filename", None)
        fd, path = tempfile.mkstemp(prefix="asr-upload-", suffix=_suffix_from_filename(filename))
        size = 0
        try:
            with os.fdopen(fd, "wb") as tmp:
                while True:
                    chunk = await upload.read(_UPLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    tmp.write(chunk)
                    size += len(chunk)
        except BaseException:
            # Nothing else holds the path yet, so a failed spool must not leak it.
            # The thread runs to completion even if this await is cancelled.
            await asyncio.to_thread(_unlink_quietly, path)
            raise
        return cls._file_backed(path, filename=filename, size=size)

    @property
    def size(self) -> int:
        """Encoded upload size in bytes, without materialising the upload."""
        return self._size

    async def read_bytes(self) -> bytes:
        """Return the whole encoded upload, materialising it when file-backed.

        Prefer :attr:`size` or :meth:`temp_path`: this is O(upload) in resident
        memory by definition.

        Raises:
            RuntimeError: If the instance has already been cleaned up.

        """
        if self._data is not None:
            return self._data
        if self._temp_path is None:
            raise RuntimeError("AudioInput has already been cleaned up.")
        return await asyncio.to_thread(Path(self._temp_path).read_bytes)

    async def temp_path(self) -> str:
        """Return a filesystem path to the encoded upload, spooling if needed.

        Raises:
            RuntimeError: If the instance has already been cleaned up.

        """
        if self._temp_path is None:
            if self._data is None:
                raise RuntimeError("AudioInput has already been cleaned up.")
            suffix = _suffix_from_filename(self._filename)
            self._temp_path = _spool_to_temp(self._data, prefix="asr-upload-", suffix=suffix)
            self._owns_temp_path = True
        return self._temp_path

    async def cleanup(self) -> None:
        """Unlink the temp file this instance created, if any. Idempotent.

        A referenced file (see :meth:`from_path`) belongs to the caller and is
        left in place; only the reference is dropped.
        """
        path, owned = self._temp_path, self._owns_temp_path
        self._temp_path = None
        self._owns_temp_path = False
        if path is not None and owned:
            await asyncio.to_thread(_unlink_quietly, path)
