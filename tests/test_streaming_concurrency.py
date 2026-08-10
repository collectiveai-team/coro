"""The Streaming Diarization Feed must not run model work on the event loop.

Inserting each PCM chunk into the diarizer synchronously ran a mel preprocessor
and a model forward step on the event loop thread, so one request stalled every
other request and transcript delta events could not flush while the diarizer
ran.

Both tests are deterministic rather than timing-based: they block inside the
diarizer on a synchronisation primitive that can only be released by other work
progressing. If that work is on the event loop and the diarizer is too, nothing
progresses and the wait raises instead of hanging the suite.
"""

from __future__ import annotations

import asyncio
import struct
import threading
from unittest.mock import patch

import pytest

from coro.audio import SAMPLE_RATE, AudioInput
from coro.core.models import SpeakerSegment, TranscriptToken
from coro.pipelines.streaming import StreamingPipeline
from coro.pipelines.windowing import ASRWindowing

_ONE_SECOND_PCM = struct.pack(f"<{SAMPLE_RATE}h", *([0] * SAMPLE_RATE))
_CHUNKS_PER_REQUEST = 3
_CONCURRENT_REQUESTS = 2
# Generous: a passing run never waits, and a regression fails rather than hangs.
_RENDEZVOUS_TIMEOUT_S = 10.0
_TIMELINE = [SpeakerSegment(start=0.0, end=float(_CHUNKS_PER_REQUEST), speaker=1)]


class _PunctuatingASR:
    """Emit one punctuation-terminated token per window so segments finalize."""

    def __init__(self) -> None:
        self.calls = 0

    async def transcribe_pcm(self, pcm, *, language=None, prompt=None):
        self.calls += 1
        return [TranscriptToken(start=0.0, end=0.5, text=f" w{self.calls}.", probability=1.0)]


class _RendezvousDiarizer:
    """Block in ingest and finalize until every concurrent request arrives.

    Clearing a barrier requires all requests to be inside it at once, which is
    only possible when their model work runs on separate threads.
    """

    def __init__(self, ingest: threading.Barrier, finalize: threading.Barrier) -> None:
        self._ingest = ingest
        self._finalize = finalize
        self._met_ingest = False

    def ingest_pcm_chunk(self, pcm: bytes) -> None:
        if not self._met_ingest:
            self._met_ingest = True
            self._ingest.wait()

    def finalize(self) -> list[SpeakerSegment]:
        self._finalize.wait()
        return list(_TIMELINE)


class _LoopReleasedDiarizer:
    """Block in ingest until a coroutine on the event loop releases it.

    The releasing coroutine can only run if the event loop is free, so a
    successful ingest proves the diarizer did not occupy the loop thread.
    """

    def __init__(self, released: threading.Event) -> None:
        self._released = released

    def ingest_pcm_chunk(self, pcm: bytes) -> None:
        if not self._released.wait(timeout=_RENDEZVOUS_TIMEOUT_S):
            raise AssertionError(
                "the event loop never ran while the diarizer ingested a chunk, "
                "so ingest is blocking the loop thread"
            )

    def finalize(self) -> list[SpeakerSegment]:
        return list(_TIMELINE)


async def _fixed_chunks(path: str, chunk_seconds: float = 1.0):
    for _ in range(_CHUNKS_PER_REQUEST):
        yield _ONE_SECOND_PCM


def _mock_chunks():
    return patch("coro.pipelines.streaming.stream_pcm_from_file", new=_fixed_chunks)


async def _transcribe(factory, spill_dir: str):
    pipeline = StreamingPipeline(
        asr=_PunctuatingASR(),
        windowing=ASRWindowing(window_seconds=1.0, overlap_seconds=0.0),
        streaming_diarizer_factory=factory,
        spill_dir=spill_dir,
    )
    return await pipeline.transcribe(AudioInput(b"audio"))


@pytest.mark.asyncio
async def test_concurrent_streaming_requests_do_not_stall_each_other(tmp_path):
    ingest = threading.Barrier(_CONCURRENT_REQUESTS, timeout=_RENDEZVOUS_TIMEOUT_S)
    finalize = threading.Barrier(_CONCURRENT_REQUESTS, timeout=_RENDEZVOUS_TIMEOUT_S)

    def factory() -> _RendezvousDiarizer:
        return _RendezvousDiarizer(ingest, finalize)

    with _mock_chunks():
        results = await asyncio.gather(
            *(
                _transcribe(factory, str(tmp_path / f"request-{i}"))
                for i in range(_CONCURRENT_REQUESTS)
            )
        )

    assert len(results) == _CONCURRENT_REQUESTS
    assert all(len(result.segments) > 0 for result in results), (
        "each concurrent request must produce a transcript"
    )
    assert {segment.speaker for result in results for segment in result.segments} == {"1"}


@pytest.mark.asyncio
async def test_event_loop_keeps_running_while_the_diarizer_ingests(tmp_path):
    released = threading.Event()

    async def _release_from_the_loop() -> None:
        await asyncio.sleep(0)
        released.set()

    releaser = asyncio.create_task(_release_from_the_loop())
    with _mock_chunks():
        result = await _transcribe(lambda: _LoopReleasedDiarizer(released), str(tmp_path))
    await releaser

    # This test asserts the event loop stayed responsive; that a transcript came
    # back at all is the liveness signal, not a count to pin down.
    assert len(result.segments) > 0  # falsegreen: ignore
