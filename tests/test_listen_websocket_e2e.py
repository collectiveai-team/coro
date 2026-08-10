"""End-to-end WebSocket: a real uvicorn server, a real client, real audio.

``tests/test_listen_websocket.py`` drives the ASGI app through Starlette's
in-process transport, which never exercises uvicorn's WebSocket implementation
— so the ``websockets`` runtime dependency could be missing and every one of
those tests would still pass while production rejected the upgrade.

These tests bind a real socket, connect with the ``websockets`` client, stream
PCM decoded from a real recording, and validate every frame against Deepgram's
own published live types. Frame conformance was broken and green before this
file existed: ``Results.metadata`` and ``Metadata.{transaction_key,sha256}``
are required by the vendor schema and coro emitted none of them.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import threading

import pytest
import uvicorn
import websockets
from deepgram.listen.v1.types.listen_v1metadata import ListenV1Metadata
from deepgram.listen.v1.types.listen_v1results import ListenV1Results

from coro.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
from coro.bench.data import WARMUP_AUDIO_PATH
from coro.core.models import SpeakerSegment, TranscriptToken
from coro.settings import ServerSettings
from conftest import make_app

pytestmark = pytest.mark.asyncio

CHUNK_SECONDS = 1.0
CHUNK_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * CHUNK_SECONDS)


class _FakeASR:
    """Deterministic tokens; the point here is transport, not model quality."""

    async def transcribe_pcm(self, pcm, *, language=None, prompt=None):
        return [
            TranscriptToken(start=0.0, end=0.6, text=" ask", probability=0.94),
            TranscriptToken(start=0.6, end=1.2, text=" not.", probability=0.88),
        ]


class _FakeDiarizer:
    def __init__(self):
        self.processed_chunks = 0

    def ingest_pcm_chunk(self, chunk):
        self.processed_chunks += 1

    def finalize(self):
        return [SpeakerSegment(start=0.0, end=600.0, speaker=1)]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _real_pcm(seconds: int) -> list[bytes]:
    """Decode the bundled recording to canonical PCM and slice it into frames."""
    import subprocess  # noqa: S404 -- ffmpeg is already the project's decoder

    raw = subprocess.run(  # noqa: S603
        [
            "ffmpeg", "-nostdin", "-loglevel", "quiet", "-i", str(WARMUP_AUDIO_PATH),
            "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    frames = [raw[i : i + CHUNK_BYTES] for i in range(0, len(raw), CHUNK_BYTES)]
    # Loop the clip if it is shorter than the window we need to force a result.
    while len(frames) < seconds:
        frames += frames
    return frames[:seconds]


class _LiveServer:
    """A real uvicorn instance on a real port."""

    def __init__(self, app):
        self.port = _free_port()
        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error")
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    async def __aenter__(self):
        self._thread.start()
        for _ in range(200):
            if self._server.started:
                return self
            await asyncio.sleep(0.05)
        raise RuntimeError("uvicorn did not start")

    async def __aexit__(self, *exc):
        self._server.should_exit = True
        self._thread.join(timeout=10)

    @property
    def ws_url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/v1/listen"


def _app(*, diarize: bool = False):
    """Build the real app, but keep the lifespan from loading real models.

    ``create_app``'s lifespan builds an ASR adapter and runs warmup, which would
    download whisper-medium. These tests are about transport and frame
    conformance, so the injected runtime is preserved instead.
    """
    settings = ServerSettings(_env_file=None)
    app = make_app(pipeline=object(), settings=settings)
    app.state.runtime.asr_adapter = _FakeASR()
    if diarize:
        app.state.runtime.streaming_diarizer_factory = _FakeDiarizer
    runtime = app.state.runtime

    @contextlib.asynccontextmanager
    async def _keep_injected_runtime(application):
        application.state.settings = settings
        application.state.runtime = runtime
        yield

    app.router.lifespan_context = _keep_injected_runtime
    return app


async def _stream(server: _LiveServer, query: str = "", *, seconds: int = 31) -> list[dict]:
    frames: list[dict] = []
    async with websockets.connect(server.ws_url + query) as ws:
        for chunk in _real_pcm(seconds):
            await ws.send(chunk)
        await ws.send(json.dumps({"type": "CloseStream"}))
        async for raw in ws:
            frame = json.loads(raw)
            frames.append(frame)
            if frame.get("type") == "Metadata":
                break
    return frames


class TestRealServerTransport:
    async def test_uvicorn_accepts_the_websocket_upgrade(self):
        # Fails outright if the `websockets` runtime dependency is missing,
        # which the in-process TestClient can never detect.
        async with _LiveServer(_app()) as server:
            async with websockets.connect(server.ws_url) as ws:
                assert ws.state is websockets.protocol.State.OPEN

    async def test_real_audio_produces_results_and_metadata(self):
        async with _LiveServer(_app()) as server:
            frames = await _stream(server)
        assert any(f["type"] == "Results" for f in frames)
        assert frames[-1]["type"] == "Metadata"

    async def test_results_arrive_before_the_client_closes(self):
        async with _LiveServer(_app()) as server, websockets.connect(server.ws_url) as ws:
            for chunk in _real_pcm(31):
                await ws.send(chunk)
            frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            assert frame["type"] == "Results"
            await ws.send(json.dumps({"type": "CloseStream"}))


class TestDeepgramLiveSdkConformance:
    async def test_every_results_frame_validates_against_the_sdk(self):
        async with _LiveServer(_app()) as server:
            frames = await _stream(server)
        results = [f for f in frames if f["type"] == "Results"]
        assert results
        for frame in results:
            ListenV1Results.model_validate(frame)

    async def test_metadata_frame_validates_against_the_sdk(self):
        async with _LiveServer(_app()) as server:
            frames = await _stream(server)
        parsed = ListenV1Metadata.model_validate(frames[-1])
        assert parsed.request_id
        assert len(parsed.sha256) == 64
        assert parsed.channels == 1

    async def test_metadata_digest_matches_the_audio_actually_sent(self):
        import hashlib

        chunks = _real_pcm(31)
        expected = hashlib.sha256(b"".join(chunks)).hexdigest()
        async with _LiveServer(_app()) as server:
            frames = await _stream(server)
        assert frames[-1]["sha256"] == expected

    async def test_diarized_words_validate_and_carry_speakers(self):
        async with _LiveServer(_app(diarize=True)) as server:
            frames = await _stream(server, "?diarize=true")
        results = [ListenV1Results.model_validate(f) for f in frames if f["type"] == "Results"]
        speakers = [w.speaker for w in results[-1].channel.alternatives[0].words]
        assert speakers and set(speakers) == {1}


class TestRealClientRejection:
    async def test_unsupported_encoding_is_refused_over_a_real_socket(self):
        async with _LiveServer(_app()) as server:
            async with websockets.connect(server.ws_url + "?encoding=opus") as ws:
                frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                assert frame["type"] == "Error"
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(ws.recv(), timeout=5)
