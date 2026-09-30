"""Live PCM source: an unbounded chunk queue whose backlog spills to disk.

The socket handler never waits on the pipeline (ADR 0023): a handler that
stops reading while it is behind loses the socket to a keepalive timeout. So
the backlog is unbounded, and it is only bounded in *memory*: past
``memory_limit_bytes`` new chunks are appended to an anonymous temp file and
read back in order, keeping host RAM flat however far behind a stream is, like
the Streaming Pipeline's transcript spill store.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import BinaryIO

_SENTINEL = object()

# About 33 s of canonical 16 kHz s16 PCM: more than one ASR window's worth of
# audio in flight, so a stream that keeps up never touches the disk.
DEFAULT_MEMORY_LIMIT_BYTES = 1 << 20


class LiveAudioSource:
    """An async PCM chunk iterator fed by a producer that is still running.

    The socket handler pushes frames in as they arrive and calls
    :meth:`close` when the client signals end of stream; the windowing layer
    pulls from the other end and cannot tell the difference from a file.
    :meth:`release` must be called once the source is no longer needed; it
    closes the spill file, which is unlinked from creation, so nothing is left
    on disk even if the process dies.
    """

    def __init__(
        self,
        *,
        spill_dir: str | None = None,
        memory_limit_bytes: int = DEFAULT_MEMORY_LIMIT_BYTES,
    ) -> None:
        # Items are bytes held in memory, or (offset, length) into the spill file.
        self._queue: asyncio.Queue = asyncio.Queue()
        self._closed = False
        self._spill_dir = spill_dir
        self._memory_limit = memory_limit_bytes
        self._memory_bytes = 0
        self._spilled_bytes = 0
        self._spill: BinaryIO | None = None
        self._write_offset = 0

    @property
    def closed(self) -> bool:
        """True once the producer has signalled end of stream."""
        return self._closed

    @property
    def pending_bytes(self) -> int:
        """PCM bytes received but not yet pulled by the consumer."""
        return self._memory_bytes + self._spilled_bytes

    @property
    def memory_bytes(self) -> int:
        """Pending PCM bytes held in memory."""
        return self._memory_bytes

    @property
    def spilled_bytes(self) -> int:
        """Pending PCM bytes held in the spill file."""
        return self._spilled_bytes

    @property
    def spill_file_size(self) -> int:
        """Current size of the spill file; 0 when none is open."""
        return self._write_offset

    async def push(self, chunk: bytes) -> None:
        """Hand one PCM chunk to the consumer; never waits."""
        if self._closed or not chunk:
            return
        if self._memory_bytes + len(chunk) <= self._memory_limit:
            self._memory_bytes += len(chunk)
            self._queue.put_nowait(chunk)
            return
        spill = self._spill_file()
        os.pwrite(spill.fileno(), chunk, self._write_offset)
        self._queue.put_nowait((self._write_offset, len(chunk)))
        self._write_offset += len(chunk)
        self._spilled_bytes += len(chunk)

    async def close(self) -> None:
        """Signal end of stream; the consumer finishes its current work."""
        if self._closed:
            return
        self._closed = True
        self._queue.put_nowait(_SENTINEL)

    def release(self) -> None:
        """Close the spill file; pending audio is discarded."""
        if self._spill is not None:
            self._spill.close()
            self._spill = None
        self._write_offset = 0

    async def chunks(self) -> AsyncIterator[bytes]:
        """Yield PCM chunks until the producer closes the stream."""
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                return
            if isinstance(item, bytes):
                self._memory_bytes -= len(item)
                yield item
                continue
            yield self._read_spilled(*item)

    def _spill_file(self) -> BinaryIO:
        if self._spill is None:
            if self._spill_dir is not None:
                Path(self._spill_dir).mkdir(parents=True, exist_ok=True)
            # Held open across calls and closed by release(); unlinked on creation.
            self._spill = tempfile.TemporaryFile(dir=self._spill_dir, prefix="coro-live-")  # noqa: SIM115
        return self._spill

    def _read_spilled(self, offset: int, length: int) -> bytes:
        spill = self._spill_file()
        chunk = os.pread(spill.fileno(), length, offset)
        self._spilled_bytes -= length
        if self._spilled_bytes == 0:
            # No spilled item is outstanding, so the file can be reused from 0.
            spill.truncate(0)
            self._write_offset = 0
        return chunk
