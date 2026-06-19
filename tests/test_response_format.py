"""Cycle 5: unsupported response_format returns OpenAI-style error.

Confirms that the v1 route rejects unsupported text formats and accepts the
implemented JSON response formats.
"""

from __future__ import annotations

import io
import struct
import wave

import pytest
from httpx import ASGITransport, AsyncClient

from coro.api.v1.transcriptions import ResponseFormat
from coro.app import create_app
from coro.runtime import RuntimeState
from coro.settings import ServerSettings


def test_response_format_enum_has_json_members():
    """ResponseFormat Enum exposes expected JSON-like and unsupported members."""
    assert ResponseFormat.JSON.value == "json"
    assert ResponseFormat.VERBOSE_JSON.value == "verbose_json"
    assert ResponseFormat.JSON_VERBOSE.value == "json_verbose"
    assert ResponseFormat.DIARIZED_JSON.value == "diarized_json"
    assert ResponseFormat.DIRIZED_JSON.value == "dirized_json"


def test_response_format_enum_has_unsupported_members():
    """ResponseFormat Enum includes unsupported format names for validation."""
    assert ResponseFormat.TEXT.value == "text"
    assert ResponseFormat.SRT.value == "srt"
    assert ResponseFormat.VTT.value == "vtt"
    assert ResponseFormat.TSV.value == "tsv"


def test_response_format_enum_json_like_is_iterable():
    """JSON-like formats can be determined from the Enum without a hard-coded set."""
    from coro.api.v1.transcriptions import _JSON_LIKE_FORMATS

    assert ResponseFormat.JSON in _JSON_LIKE_FORMATS
    assert ResponseFormat.VERBOSE_JSON in _JSON_LIKE_FORMATS
    assert ResponseFormat.JSON_VERBOSE in _JSON_LIKE_FORMATS
    assert ResponseFormat.DIARIZED_JSON in _JSON_LIKE_FORMATS
    assert ResponseFormat.DIRIZED_JSON in _JSON_LIKE_FORMATS


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
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello",
                    "speaker": "agent",
                    "words": [],
                }
            ],
            "word_segments": [
                {"word": "hello", "start": 0.0, "end": 1.0, "score": 1.0, "speaker": "agent"}
            ],
            "transcript": [{"start": 0.0, "end": 1.0, "text": "hello"}],
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
            data={"model": "whisper-1", "response_format": fmt},
        )
    assert response.status_code == 400
    body = response.json()
    assert "error" in body
    assert "message" in body["error"]
    assert body["error"].get("param") == "response_format"


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["/v1/audio/transcriptions"])
@pytest.mark.parametrize(
    ("model", "fmt"),
    [
        ("whisper-1", "json"),
        ("whisper-1", "verbose_json"),
        ("anything", "json_verbose"),
        ("gpt-4o-transcribe-diarize", "diarized_json"),
        ("ignored-model", "dirized_json"),
        ("whisper-1", ""),
        ("whisper-1", None),
    ],
)
async def test_json_formats_accepted(route, model, fmt):
    """Implemented JSON response formats (including empty/None) are accepted."""
    app = _app()
    data: dict = {"model": model}
    if fmt is not None:
        data["response_format"] = fmt
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            route,
            files={"file": ("test.wav", _minimal_wav(), "audio/wav")},
            data=data,
        )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_model_is_ignored_for_response_format_support():
    app = _app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", _minimal_wav(), "audio/wav")},
            data={"model": "gpt-4o-transcribe", "response_format": "verbose_json"},
        )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_extra_openai_parameters_are_ignored():
    app = _app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", _minimal_wav(), "audio/wav")},
            data={
                "model": "whisper-1",
                "timestamp_granularities[]": "word",
                "temperature": "1",
                "include[]": "logprobs",
                "known_speaker_names[]": "agent",
                "chunking_strategy": "auto",
            },
        )
    assert response.status_code == 200
