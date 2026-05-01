"""Cycle 5: unsupported response_format returns OpenAI-style error.

Confirms that both v1 and v2 routes reject unsupported formats (text, srt,
vtt) and accept JSON-like formats.
"""

from __future__ import annotations

import io
import struct
import wave

import pytest
from httpx import ASGITransport, AsyncClient

from asr_diar_server.app import create_app
from asr_diar_server.runtime import RuntimeState
from asr_diar_server.settings import ServerSettings


def _minimal_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<1600h", *([0] * 1600)))
    return buf.getvalue()


class _FakePipeline:
    async def run(self, audio_bytes, *, language=None, prompt=None):
        return {
            "segments": [],
            "word_segments": [],
            "transcript": [],
            "diarization": [],
            "raw_words": [],
        }

    async def run_from_path(self, path, *, language=None, prompt=None):
        return {
            "segments": [],
            "word_segments": [],
            "transcript": [],
            "diarization": [],
            "raw_words": [],
        }


def _app():
    from fastapi import FastAPI

    application: FastAPI = create_app(ServerSettings())
    runtime = RuntimeState(asr_adapter=object())
    runtime.v1_pipeline = _FakePipeline()
    runtime.v2_pipeline = _FakePipeline()
    application.state.runtime = runtime
    return application


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["/v1/audio/transcriptions", "/v2/audio/transcriptions"])
@pytest.mark.parametrize("fmt", ["text", "srt", "vtt", "tsv"])
async def test_unsupported_format_returns_openai_error(route, fmt):
    """Unsupported response_format yields 400 with OpenAI-style error body."""
    app = _app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            route,
            files={"file": ("test.wav", _minimal_wav(), "audio/wav")},
            data={"response_format": fmt},
        )
    assert response.status_code == 400
    body = response.json()
    assert "error" in body
    assert "message" in body["error"]
    assert body["error"].get("param") == "response_format"


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["/v1/audio/transcriptions", "/v2/audio/transcriptions"])
@pytest.mark.parametrize("fmt", ["json", "verbose_json", "diarized_json", "", None])
async def test_json_like_formats_accepted(route, fmt):
    """JSON-like response formats (including empty/None) are accepted."""
    app = _app()
    data: dict = {}
    if fmt is not None:
        data["response_format"] = fmt
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            route,
            files={"file": ("test.wav", _minimal_wav(), "audio/wav")},
            data=data,
        )
    assert response.status_code == 200
