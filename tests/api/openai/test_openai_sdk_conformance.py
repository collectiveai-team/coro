"""OpenAI SDK conformance: server responses validate against `openai.types.audio`.

These tests guarantee that a consumer can parse this server's responses with the
official ``openai`` SDK types (no custom schema package required). If a field the
SDK marks required (e.g. ``seek`` on verbose segments) is dropped, these fail.
"""

from __future__ import annotations

import io
import struct
import wave
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from openai.types.audio import (
    Transcription,
    TranscriptionDiarized,
    TranscriptionVerbose,
)

from coro.app import create_app
from coro.core.models import (
    ResponseSegment,
    TranscriptionResult,
    TranscriptItem,
    TranscriptWord,
)
from coro.runtime import RuntimeState
from coro.settings import ServerSettings


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
        return TranscriptionResult(
            segments=[
                ResponseSegment(start=0.0, end=1.0, text="hello", speaker="agent", words=[]),
            ],
            word_segments=[
                TranscriptWord(word="hello", start=0.0, end=1.0, score=1.0, speaker="agent"),
            ],
            transcript=[TranscriptItem(start=0.0, end=1.0, text="hello")],
            diarization=[],
            raw_words=[],
        )


def _app():
    application = create_app(ServerSettings())
    runtime = RuntimeState(asr_adapter=object())
    runtime.pipeline = _FakePipeline()
    application.state.runtime = runtime
    return application


async def _transcribe(fmt: str) -> Any:
    app = _app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", _minimal_wav(), "audio/wav")},
            data={"model": "whisper-1", "response_format": fmt},
        )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.asyncio
async def test_json_response_validates_against_openai_transcription():
    body = await _transcribe("json")
    parsed = Transcription.model_validate(body)
    assert parsed.text == "hello"


@pytest.mark.asyncio
async def test_verbose_json_validates_against_openai_transcription_verbose():
    body = await _transcribe("verbose_json")
    parsed = TranscriptionVerbose.model_validate(body)
    assert parsed.segments is not None
    # `seek` is required by the SDK segment type; assert it survived the boundary.
    assert [(s.seek, s.start, s.end, s.text) for s in parsed.segments] == [(0, 0.0, 1.0, "hello")]


@pytest.mark.asyncio
async def test_diarized_json_validates_against_openai_transcription_diarized():
    body = await _transcribe("diarized_json")
    parsed = TranscriptionDiarized.model_validate(body)
    assert [(s.speaker, s.start, s.end, s.text) for s in parsed.segments] == [
        ("agent", 0.0, 1.0, "hello")
    ]
