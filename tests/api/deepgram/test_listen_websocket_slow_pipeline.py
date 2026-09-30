"""WebSocket /v1/listen when the pipeline runs slower than real time.

Against real uvicorn, whose WebSocket protocol pauses reading the socket after
every message until the app calls ``receive()``. Ping and pong frames share the
TCP stream with audio, so a handler that stops reading while its pipeline is
behind leaves the peer's pong stuck behind unread audio, and uvicorn closes the
socket with ``1011 keepalive ping timeout``. Deepgram absorbs the lag instead.

Keepalive intervals are shortened so the backlog outlives several ping rounds
in a fraction of a second of test time.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time

import pytest
import websockets

from coro.core.models import SpeakerSegment, TranscriptToken
from coro.settings import ServerSettings
from support.factories import make_app
from support.live_server import LiveServer, keep_injected_runtime

pytestmark = pytest.mark.asyncio

FRAME_SECONDS = 0.1
FRAME = b"\x00\x00" * int(16000 * FRAME_SECONDS)
PING_SECONDS = 0.2
DIARIZE = "?encoding=linear16&sample_rate=16000&diarize=true"


class _FakeASR:
    async def transcribe_pcm(self, pcm, *, language=None, prompt=None):
        return [TranscriptToken(start=2.0, end=2.6, text=" hola", probability=0.9)]


class _SlowDiarizer:
    """Ingests each 0.1 s frame in 20 ms: far slower than the client sends."""

    def ingest_pcm_chunk(self, chunk):
        time.sleep(0.02)

    def finalize(self):
        return [SpeakerSegment(start=0.0, end=600.0, speaker=1)]


class _GatedDiarizer:
    """Blocks the first ingest until the test opens the gate."""

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.ingested = 0
        self.finalized = 0

    def ingest_pcm_chunk(self, chunk):
        self.ingested += 1
        self.gate.wait(timeout=30)

    def finalize(self):
        self.finalized += 1
        return []


def _app(diarizer_factory):
    settings = ServerSettings(_env_file=None)
    app = make_app(pipeline=object(), settings=settings)
    app.state.runtime.asr_adapter = _FakeASR()
    app.state.runtime.streaming_diarizer_factory = diarizer_factory
    return keep_injected_runtime(app, settings)


def _server(app) -> LiveServer:
    return LiveServer(app, ws_ping_interval=PING_SECONDS, ws_ping_timeout=PING_SECONDS)


class TestSlowerThanRealTime:
    async def test_keepalive_survives_a_backlog_and_the_stream_completes(self):
        frames: list[dict] = []
        async with (
            _server(_app(_SlowDiarizer)) as server,
            websockets.connect(
                server.ws_url + DIARIZE, ping_interval=PING_SECONDS, ping_timeout=PING_SECONDS
            ) as ws,
        ):
            # 20 s of audio at once: 4 s of processing, twenty ping rounds.
            for _ in range(200):
                await ws.send(FRAME)
            await ws.send(json.dumps({"type": "CloseStream"}))
            async for raw in ws:
                frames.append(json.loads(raw))
        assert frames[-1]["type"] == "Metadata"
        assert frames[-1]["duration"] == pytest.approx(20.0, abs=1e-6)

    async def test_keepalive_messages_during_a_backlog_do_not_stall_the_socket(self):
        frames: list[dict] = []
        async with (
            _server(_app(_SlowDiarizer)) as server,
            websockets.connect(
                server.ws_url + DIARIZE, ping_interval=PING_SECONDS, ping_timeout=PING_SECONDS
            ) as ws,
        ):
            for _ in range(200):
                await ws.send(FRAME)
                await ws.send(json.dumps({"type": "KeepAlive"}))
            await ws.send(json.dumps({"type": "CloseStream"}))
            async for raw in ws:
                frames.append(json.loads(raw))
        assert [f["type"] for f in frames].count("Metadata") == 1
        assert frames[-1]["duration"] == pytest.approx(20.0, abs=1e-6)


class _HashingDiarizer:
    """Hashes exactly the audio the pipeline hands it, in order."""

    def __init__(self) -> None:
        self.digest = hashlib.sha256()

    def ingest_pcm_chunk(self, chunk):
        self.digest.update(chunk)
        time.sleep(0.005)

    def finalize(self):
        return []


class TestSpilledBacklog:
    async def test_a_backlog_larger_than_memory_reaches_the_pipeline_intact(self):
        # 60 s at once is ~1.9 MB against a 1 MiB in-memory limit, so the
        # tail of the backlog goes through the spill file.
        diarizer = _HashingDiarizer()
        audio = [
            hashlib.sha256(i.to_bytes(2, "big")).digest() * (len(FRAME) // 32) for i in range(600)
        ]
        frames: list[dict] = []
        async with (
            _server(_app(lambda: diarizer)) as server,
            websockets.connect(server.ws_url + DIARIZE) as ws,
        ):
            for chunk in audio:
                await ws.send(chunk)
            await ws.send(json.dumps({"type": "CloseStream"}))
            async for raw in ws:
                frames.append(json.loads(raw))
        assert frames[-1]["type"] == "Metadata"
        assert frames[-1]["duration"] == pytest.approx(60.0, abs=1e-6)
        assert diarizer.digest.hexdigest() == hashlib.sha256(b"".join(audio)).hexdigest()


class TestClientGone:
    async def test_disconnect_abandons_the_backlog_instead_of_processing_it(self):
        diarizer = _GatedDiarizer()
        try:
            async with _server(_app(lambda: diarizer)) as server:
                async with websockets.connect(server.ws_url + DIARIZE) as ws:
                    for _ in range(50):
                        await ws.send(FRAME)
                # The client left without CloseStream while the first chunk
                # is still in the diarizer: nobody can receive the rest.
                await server.wait_for_handlers()
                diarizer.gate.set()
        finally:
            diarizer.gate.set()
        assert diarizer.ingested == 1
        assert diarizer.finalized == 0
