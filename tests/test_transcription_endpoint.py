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

from coro.api.exceptions import UNDECODABLE_MEDIA_MESSAGE
from coro.app import create_app
from coro.audio import AudioConversionError
from coro.core.models import (
    DiarizationItem,
    ResponseSegment,
    TranscriptionResult,
    TranscriptItem,
    TranscriptWord,
)
from coro.runtime import RuntimeState
from coro.settings import ServerSettings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PIPELINE_RESULT = TranscriptionResult(
    segments=[
        ResponseSegment(start=0.0, end=1.0, text="hello world", speaker="agent", words=[]),
    ],
    word_segments=[
        TranscriptWord(word="hello", start=0.0, end=0.4, score=1.0, speaker="agent"),
        TranscriptWord(word="world", start=0.5, end=1.0, score=1.0, speaker="agent"),
    ],
    transcript=[TranscriptItem(start=0.0, end=1.0, text="hello world")],
    diarization=[DiarizationItem(start=0.0, end=1.0, speaker="agent")],
    raw_words=[],
)


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
        return _PIPELINE_RESULT


def _app_with_pipeline(pipeline):
    """Build app with the given pipeline injected into RuntimeState."""
    from fastapi import FastAPI

    application: FastAPI = create_app(ServerSettings())
    runtime = RuntimeState(asr_adapter=object())  # non-None → ready=True
    runtime.pipeline = pipeline
    application.state.runtime = runtime
    return application


def _app_with_fake_pipeline():
    """Build app with a fake configured pipeline injected into RuntimeState."""
    return _app_with_pipeline(_FakePipeline())


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
async def test_transcription_endpoint_rejects_invalid_language_tag():
    """A non-BCP-47 language hint returns a 400, not a generic 500."""
    app = _app_with_fake_pipeline()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", _minimal_wav_bytes(), "audio/wav")},
            data={"model": "whisper-1", "language": "string"},
        )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["param"] == "language"


@pytest.mark.asyncio
async def test_transcription_endpoint_treats_blank_language_as_unset():
    """An empty / whitespace-only language hint is coerced to None and succeeds."""
    app = _app_with_fake_pipeline()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for blank in ("", "   "):
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", _minimal_wav_bytes(), "audio/wav")},
                data={"model": "whisper-1", "language": blank},
            )
            assert response.status_code == 200, blank


@pytest.mark.asyncio
async def test_transcription_endpoint_tolerates_swagger_blank_fields():
    """Swagger's 'Try it out' sends every optional field as an empty string.

    The empty ``known_speaker_references[]`` value must not fail UploadFile
    parsing with a 422; blanks are treated as unset.
    """
    app = _app_with_fake_pipeline()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", _minimal_wav_bytes(), "audio/wav")},
            data={
                "stream": "",
                "prompt": "",
                "timestamp_granularities[]": "",
                "known_speaker_references[]": "",
                "model": "",
                "include[]": "",
                "known_speaker_names[]": "",
                "temperature": "",
                "response_format": "diarized_json",
                "language": "",
                "chunking_strategy": "",
            },
        )
    assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_transcription_endpoint_undecodable_media_returns_400():
    """An undecodable upload is a client error: 400 invalid_request_error, no ffmpeg leak."""

    class _BadMediaPipeline:
        async def transcribe(self, audio, *, language=None, prompt=None):
            raise AudioConversionError("Audio conversion failed: moov atom not found")

    app = _app_with_pipeline(_BadMediaPipeline())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("clip.mp4", b"not really an mp4", "video/mp4")},
            data={"model": "whisper-1"},
        )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["param"] == "file"
    assert body["error"]["message"] == UNDECODABLE_MEDIA_MESSAGE
    assert "moov atom" not in body["error"]["message"]  # ffmpeg detail not leaked


@pytest.mark.asyncio
async def test_transcription_endpoint_internal_valueerror_stays_500():
    """A non-conversion ValueError (e.g. config) is NOT misclassified as a 400."""

    class _ConfigErrorPipeline:
        async def transcribe(self, audio, *, language=None, prompt=None):
            raise ValueError("overlap_seconds must be less than window_seconds")

    app = _app_with_pipeline(_ConfigErrorPipeline())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", _minimal_wav_bytes(), "audio/wav")},
            data={"model": "whisper-1"},
        )
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["type"] == "server_error"
    assert body["error"]["message"] == "Transcription processing failed."


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
