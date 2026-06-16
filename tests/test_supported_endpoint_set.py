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
