"""Vendor SDK conformance: responses validate against the vendors' own types.

A vendor-named format that the vendor's SDK cannot parse is a liability, since
clients will hand it to that SDK (ADR 0010). These tests are the mechanical
check on that policy, mirroring ``test_openai_sdk_conformance`` for the OpenAI
formats: they parse the server's output with the real published SDK types
rather than with a local copy of the schema.
"""

from __future__ import annotations

from typing import Any

import pytest
from assemblyai.types import TranscriptResponse
from deepgram.types.listen_v1response import ListenV1Response
from httpx import ASGITransport, AsyncClient

from coro.core.models import (
    ResponseSegment,
    TranscriptionResult,
    TranscriptItem,
    TranscriptWord,
)
from conftest import make_app, make_wav


class _FakePipeline:
    async def transcribe(self, audio, *, language=None, prompt=None):
        words = [
            TranscriptWord(word="hola", start=0.0, end=0.5, score=0.91, speaker="1"),
            TranscriptWord(word="mundo", start=0.5, end=1.0, score=0.83, speaker="2"),
            TranscriptWord(word="claro", start=1.0, end=1.5, score=0.66, speaker="-1"),
        ]
        return TranscriptionResult(
            segments=[
                ResponseSegment(start=0.0, end=0.5, text="hola", speaker="1", words=words[:1]),
                ResponseSegment(start=0.5, end=1.0, text="mundo", speaker="2", words=words[1:2]),
                ResponseSegment(start=1.0, end=1.5, text="claro", speaker="-1", words=words[2:]),
            ],
            word_segments=words,
            transcript=[TranscriptItem(start=0.0, end=1.5, text="hola mundo claro")],
            diarization=[],
            raw_words=[],
        )


async def _transcribe(fmt: str) -> Any:
    app = make_app(_FakePipeline())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", make_wav(), "audio/wav")},
            data={"model": "whisper-1", "response_format": fmt},
        )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.asyncio
async def test_assemblyai_json_validates_against_assemblyai_sdk():
    body = await _transcribe("assemblyai_json")
    parsed = TranscriptResponse.model_validate(body)
    assert parsed.utterances
    # The per-word speaker slot is native to this shape; assert it survived.
    assert parsed.utterances[0].words[0].speaker == "1"
    assert parsed.words is not None
    assert [word.speaker for word in parsed.words] == ["1", "2", None]


@pytest.mark.asyncio
async def test_deepgram_json_validates_against_deepgram_sdk():
    body = await _transcribe("deepgram_json")
    parsed = ListenV1Response.model_validate(body)
    assert parsed.results.utterances
    assert [u.speaker for u in parsed.results.utterances] == [1, 2, None]
    utterance_words = parsed.results.utterances[0].words
    assert utterance_words is not None
    assert utterance_words[0].speaker == 1
    # speaker_confidence is typed by the SDK but deliberately not emitted.
    assert utterance_words[0].speaker_confidence is None


@pytest.mark.asyncio
async def test_deepgram_metadata_required_fields_are_populated():
    parsed = ListenV1Response.model_validate(await _transcribe("deepgram_json"))
    assert parsed.metadata.request_id
    assert len(parsed.metadata.sha256) == 64
    assert parsed.metadata.channels == 1
    assert parsed.metadata.duration > 0
