"""LiveAudioSource keeps a bounded amount of audio in memory and spills the rest.

The socket handler never waits on the pipeline (ADR 0023), so a client that
sends a long recording at once leaves the whole backlog pending. It must land
on disk, not in RAM, and come back out byte-for-byte in order.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from coro.pipelines.live import LiveAudioSource

pytestmark = pytest.mark.asyncio

MEMORY_LIMIT = 1000


def _chunks(count: int, size: int = 300) -> list[bytes]:
    """Distinct, non-uniform chunks, so a reordered or corrupted read shows."""
    return [(hashlib.sha256(i.to_bytes(2, "big")).digest() * size)[:size] for i in range(count)]


async def _drain(source: LiveAudioSource) -> list[bytes]:
    return [chunk async for chunk in source.chunks()]


def _open_fds() -> int:
    return len(list(Path("/proc/self/fd").iterdir()))


async def test_backlog_beyond_the_memory_limit_is_spilled_to_disk(tmp_path):
    source = LiveAudioSource(spill_dir=str(tmp_path), memory_limit_bytes=MEMORY_LIMIT)
    for chunk in _chunks(100):
        await source.push(chunk)
    assert source.memory_bytes <= MEMORY_LIMIT
    assert source.spilled_bytes == 100 * 300 - source.memory_bytes
    assert source.pending_bytes == 100 * 300
    await source.close()
    source.release()


async def test_spilled_chunks_come_back_in_order_and_intact(tmp_path):
    sent = _chunks(100)
    source = LiveAudioSource(spill_dir=str(tmp_path), memory_limit_bytes=MEMORY_LIMIT)
    for chunk in sent:
        await source.push(chunk)
    await source.close()
    assert await _drain(source) == sent
    assert source.pending_bytes == 0
    source.release()


async def test_interleaved_push_and_pull_preserve_order(tmp_path):
    sent = _chunks(60)
    source = LiveAudioSource(spill_dir=str(tmp_path), memory_limit_bytes=MEMORY_LIMIT)
    received: list[bytes] = []
    pull = source.chunks()
    for i, chunk in enumerate(sent):
        await source.push(chunk)
        if i % 3 == 0:
            received.append(await anext(pull))
    await source.close()
    received += [chunk async for chunk in pull]
    assert received == sent
    source.release()


async def test_a_drained_spill_file_is_truncated(tmp_path):
    source = LiveAudioSource(spill_dir=str(tmp_path), memory_limit_bytes=MEMORY_LIMIT)
    pull = source.chunks()
    for chunk in _chunks(20):
        await source.push(chunk)
    for _ in range(20):
        await anext(pull)
    assert source.spilled_bytes == 0
    assert source.spill_file_size == 0
    source.release()


async def test_release_closes_the_spill_file_and_leaves_nothing_on_disk(tmp_path):
    before = _open_fds()
    source = LiveAudioSource(spill_dir=str(tmp_path), memory_limit_bytes=MEMORY_LIMIT)
    for chunk in _chunks(50):
        await source.push(chunk)
    source.release()
    assert _open_fds() == before
    assert list(tmp_path.iterdir()) == []


async def test_below_the_memory_limit_nothing_touches_disk(tmp_path):
    before = _open_fds()
    source = LiveAudioSource(spill_dir=str(tmp_path), memory_limit_bytes=MEMORY_LIMIT)
    for chunk in _chunks(3):
        await source.push(chunk)
    assert source.spilled_bytes == 0
    assert _open_fds() == before
    source.release()
