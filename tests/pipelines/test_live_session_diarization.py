"""LiveTranscriptionSession must run diarization off the event loop.

``ingest_pcm_chunk`` runs a mel preprocessor and a Sortformer forward step, and
``finalize`` runs post-processing over the whole prediction tensor. Called on
the event loop they freeze every other connection and request on the server
for the duration of each chunk — the Deepgram WebSocket path did exactly that
while the HTTP Streaming Pipeline already off-loaded both.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from coro.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
from coro.core.models import SpeakerSegment
from coro.pipelines.live import LiveAudioSource, LiveTranscriptionSession
from coro.pipelines.windowing import ASRWindowing

_CHUNK = b"\x00\x00" * (SAMPLE_RATE // 2)
_N_CHUNKS = 4
_TIMELINE = [SpeakerSegment(start=0.0, end=2.0, speaker=1)]


class _SilentASR:
    async def transcribe_pcm(self, pcm: bytes, *, language=None, prompt=None):
        return []


class _LoopProbingDiarizer:
    """Blocks each call until the event loop proves it is still running.

    If the call runs on the loop itself, the probe coroutine can never be
    scheduled, the wait times out, and the call is recorded as blocking.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self.ingested: list[int] = []
        self.loop_stayed_free: list[bool] = []
        self._in_call = threading.Lock()

    def _probe_loop(self) -> None:
        answered = threading.Event()
        self._loop.call_soon_threadsafe(answered.set)
        self.loop_stayed_free.append(answered.wait(timeout=2))

    def ingest_pcm_chunk(self, pcm: bytes) -> None:
        # Chunks of one stream must still arrive one at a time, in order.
        assert self._in_call.acquire(blocking=False), "overlapping ingest on one diarizer"
        try:
            self._probe_loop()
            self.ingested.append(len(self.ingested))
        finally:
            self._in_call.release()

    def finalize(self) -> list[SpeakerSegment]:
        self._probe_loop()
        return list(_TIMELINE)


@pytest.mark.asyncio
async def test_diarization_never_blocks_the_event_loop():
    diarizer = _LoopProbingDiarizer(asyncio.get_running_loop())
    session = LiveTranscriptionSession(
        asr=_SilentASR(),
        windowing=ASRWindowing(window_seconds=1.0, overlap_seconds=0.0),
        streaming_diarizer_factory=lambda: diarizer,
    )
    source = LiveAudioSource()
    for _ in range(_N_CHUNKS):
        await source.push(_CHUNK)
    await source.close()

    async for _ in session.run(source):
        pass
    timeline = await session.finalize()

    assert diarizer.ingested == list(range(_N_CHUNKS))
    assert diarizer.loop_stayed_free == [True] * (_N_CHUNKS + 1)
    assert timeline == _TIMELINE
    assert session.audio_seconds == _N_CHUNKS * len(_CHUNK) / (SAMPLE_RATE * BYTES_PER_SAMPLE)


@pytest.mark.asyncio
async def test_finalize_without_a_diarizer_is_empty():
    session = LiveTranscriptionSession(asr=_SilentASR())
    assert await session.finalize() == []
