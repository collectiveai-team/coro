"""Supported Endpoint Set behavior."""

from __future__ import annotations

import io
import struct
import wave

import pytest
from httpx import ASGITransport, AsyncClient

from coro.app import create_app
from coro.settings import ServerSettings


def _minimal_wav_bytes() -> bytes:
    buf = io.BytesIO()
    n_frames = 1600
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<" + "h" * n_frames, *([0] * n_frames)))
    return buf.getvalue()


@pytest.mark.asyncio
async def test_behavior_specific_transcription_endpoint_is_not_supported():
    app = create_app(ServerSettings(_env_file=None))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v2/audio/transcriptions",
            files={"file": ("test.wav", _minimal_wav_bytes(), "audio/wav")},
        )
    assert response.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/", "/asr", "/v1/models", "/v2/transcript"])
async def test_excluded_routes_stay_excluded(path: str):
    # ADR 0001's exclusions still hold; ADR 0015 amended the set only to add
    # a deliberately implemented /v1/listen.
    app = create_app(ServerSettings(_env_file=None))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.post(path, content=b"")).status_code == 404


@pytest.mark.asyncio
async def test_deepgram_endpoint_is_in_the_supported_endpoint_set():
    app = create_app(ServerSettings(_env_file=None))
    routes = {getattr(route, "path", None) for route in app.routes}
    assert "/v1/listen" in routes
    assert "/v1/audio/transcriptions" in routes
    assert "/health" in routes
