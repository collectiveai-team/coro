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

from asr_diar_server.api.v1.transcriptions import ResponseFormat
from asr_diar_server.app import create_app
from asr_diar_server.runtime import RuntimeState
from asr_diar_server.settings import ServerSettings


def test_response_format_enum_has_json_members():
    """ResponseFormat Enum exposes expected JSON-like and unsupported members."""
    assert ResponseFormat.JSON.value == "json"
    assert ResponseFormat.VERBOSE_JSON.value == "verbose_json"
    assert ResponseFormat.DIARIZED_JSON.value == "diarized_json"


def test_response_format_enum_has_unsupported_members():
    """ResponseFormat Enum includes unsupported format names for validation."""
    assert ResponseFormat.TEXT.value == "text"
    assert ResponseFormat.SRT.value == "srt"
    assert ResponseFormat.VTT.value == "vtt"
    assert ResponseFormat.TSV.value == "tsv"


def test_response_format_enum_json_like_is_iterable():
    """JSON-like formats can be determined from the Enum without a hard-coded set."""
    from asr_diar_server.api.v1.transcriptions import _JSON_LIKE_FORMATS
    assert ResponseFormat.JSON in _JSON_LIKE_FORMATS
    assert ResponseFormat.VERBOSE_JSON in _JSON_LIKE_FORMATS
    assert ResponseFormat.DIARIZED_JSON in _JSON_LIKE_FORMATS


def _minimal_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<1600h", *([0] * 1600)))
    return buf.getvalue()


class _FakePipeline:
    async def transcribe(self, audio, *, language=None, prompt=None):
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
    runtime.pipeline = _FakePipeline()
    application.state.runtime = runtime
    return application


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["/v1/audio/transcriptions"])
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
@pytest.mark.parametrize("route", ["/v1/audio/transcriptions"])
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
