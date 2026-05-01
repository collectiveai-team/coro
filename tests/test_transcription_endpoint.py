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

WHISPERX_KEYS = {"segments", "word_segments", "transcript", "diarization", "raw_words"}

_EMPTY_WHISPERX = {
    "segments": [],
    "word_segments": [],
    "transcript": [],
    "diarization": [],
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
    """Fake configured pipeline that returns a fixed WhisperX response."""

    async def transcribe(self, audio, *, language=None, prompt=None):
        return dict(_EMPTY_WHISPERX)


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
async def test_transcription_endpoint_returns_whisperx_keys():
    """POST /v1/audio/transcriptions returns all WhisperX-style response keys."""
    app = _app_with_fake_pipeline()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", _minimal_wav_bytes(), "audio/wav")},
        )
    assert response.status_code == 200
    body = response.json()
    assert WHISPERX_KEYS.issubset(body.keys())


@pytest.mark.asyncio
async def test_transcription_endpoint_accepts_openai_compatible_form_params():
    """POST /v1/audio/transcriptions accepts model, language, prompt, stream fields."""
    app = _app_with_fake_pipeline()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", _minimal_wav_bytes(), "audio/wav")},
            data={
                "model": "whisper-1",
                "language": "es",
                "prompt": "some hint",
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
        )
    assert response.status_code == 400
    body = response.json()
    # OpenAI-style error: {"error": {"message": "...", "type": "...", ...}}
    assert "error" in body
    assert "message" in body["error"]
