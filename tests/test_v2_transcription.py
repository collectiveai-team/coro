"""Cycle 4: v2 transcription route — WhisperX-style response shape.

The v2 route uses the disk-backed pipeline.  Tests inject a fake v2 pipeline
so no real model is loaded.  Assertions target the public Transcription API
Contract.
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

WHISPERX_KEYS = {"segments", "word_segments", "transcript", "diarization", "raw_words"}

_EMPTY_WHISPERX = {
    "segments": [],
    "word_segments": [],
    "transcript": [],
    "diarization": [],
    "raw_words": [],
}


def _minimal_wav_bytes() -> bytes:
    buf = io.BytesIO()
    n_frames = 1600
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<" + "h" * n_frames, *([0] * n_frames)))
    return buf.getvalue()


class _FakeV2Pipeline:
    """Fake disk-backed v2 pipeline that returns a fixed WhisperX response."""

    async def run_from_path(self, path: str, *, language=None, prompt=None):
        return dict(_EMPTY_WHISPERX)


def _app_with_fake_v2_pipeline():
    from fastapi import FastAPI

    application: FastAPI = create_app(ServerSettings())
    runtime = RuntimeState(asr_adapter=object())
    runtime.v2_pipeline = _FakeV2Pipeline()
    application.state.runtime = runtime
    return application


@pytest.mark.asyncio
async def test_v2_returns_whisperx_keys():
    """POST /v2/audio/transcriptions returns all WhisperX-style response keys."""
    app = _app_with_fake_v2_pipeline()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v2/audio/transcriptions",
            files={"file": ("test.wav", _minimal_wav_bytes(), "audio/wav")},
        )
    assert response.status_code == 200
    body = response.json()
    assert WHISPERX_KEYS.issubset(body.keys())


@pytest.mark.asyncio
async def test_v2_accepts_same_openai_form_params_as_v1():
    """POST /v2/audio/transcriptions accepts model, language, prompt, stream."""
    app = _app_with_fake_v2_pipeline()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v2/audio/transcriptions",
            files={"file": ("test.wav", _minimal_wav_bytes(), "audio/wav")},
            data={"model": "whisper-1", "language": "es", "prompt": "hint", "stream": "false"},
        )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_v2_empty_upload_returns_openai_style_error():
    """POST /v2/audio/transcriptions with empty body returns OpenAI-style error."""
    app = _app_with_fake_v2_pipeline()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v2/audio/transcriptions",
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
    assert response.status_code == 400
    body = response.json()
    assert "error" in body
    assert "message" in body["error"]
