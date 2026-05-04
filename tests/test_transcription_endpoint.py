"""Public Transcription Endpoint behavior.

Tests use a fake pipeline injected into RuntimeState so no real ASR model
is loaded.  All assertions target the public Transcription API Contract.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PIPELINE_RESULT = {
    "segments": [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "hello world",
            "speaker": "agent",
            "words": [],
        }
    ],
    "word_segments": [
        {"word": "hello", "start": 0.0, "end": 0.4, "score": 1.0, "speaker": "agent"},
        {"word": "world", "start": 0.5, "end": 1.0, "score": 1.0, "speaker": "agent"},
    ],
    "transcript": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
    "diarization": [{"start": 0.0, "end": 1.0, "speaker": "agent"}],
    "raw_words": [],
}


def _minimal_wav_bytes() -> bytes:
    """Produce a tiny valid WAV file (100 ms silence, 16 kHz mono 16-bit)."""
    buf = io.BytesIO()
    n_frames = 1600  # 100 ms at 16 kHz
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<" + "h" * n_frames, *([0] * n_frames)))
    return buf.getvalue()


class _FakePipeline:
    """Fake configured pipeline that returns a fixed transcription response."""

    async def transcribe(self, audio, *, language=None, prompt=None):
        return dict(_PIPELINE_RESULT)


def _app_with_fake_pipeline():
    """Build app with a fake configured pipeline injected into RuntimeState."""
    from fastapi import FastAPI

    application: FastAPI = create_app(ServerSettings())
    runtime = RuntimeState(asr_adapter=object())  # non-None → ready=True
    runtime.pipeline = _FakePipeline()
    application.state.runtime = runtime
    return application


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transcription_endpoint_returns_default_openai_json():
    """POST /v1/audio/transcriptions returns default OpenAI JSON."""
    app = _app_with_fake_pipeline()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", _minimal_wav_bytes(), "audio/wav")},
            data={"model": "whisper-1"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "hello world"
    assert body["usage"] == {"type": "duration", "seconds": 1}


@pytest.mark.asyncio
async def test_transcription_endpoint_accepts_openai_compatible_form_params():
    """POST /v1/audio/transcriptions accepts OpenAI-compatible form fields."""
    app = _app_with_fake_pipeline()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", _minimal_wav_bytes(), "audio/wav")},
            data={
                "model": "whisper-1",
                "language": "es",
                "prompt": "some hint",
                "response_format": "verbose_json",
                "temperature": "0",
                "timestamp_granularities[]": ["word", "segment"],
                "stream": "false",
            },
        )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_transcription_endpoint_empty_upload_returns_openai_style_error():
    """POST /v1/audio/transcriptions with empty body returns OpenAI-style error."""
    app = _app_with_fake_pipeline()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("empty.wav", b"", "audio/wav")},
            data={"model": "whisper-1"},
        )
    assert response.status_code == 400
    body = response.json()
    # OpenAI-style error: {"error": {"message": "...", "type": "...", ...}}
    assert "error" in body
    assert "message" in body["error"]


@pytest.mark.asyncio
async def test_transcription_endpoint_returns_diarized_json():
    """diarized_json returns speaker-annotated OpenAI segments."""
    app = _app_with_fake_pipeline()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", _minimal_wav_bytes(), "audio/wav")},
            data={"model": "gpt-4o-transcribe-diarize", "response_format": "diarized_json"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "transcribe"
    assert body["segments"][0]["type"] == "transcript.text.segment"
    assert body["segments"][0]["speaker"] == "agent"
