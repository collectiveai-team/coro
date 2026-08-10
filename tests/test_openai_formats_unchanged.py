"""The OpenAI-shaped formats are byte-frozen.

``diarized_json`` is a byte-exact clone of OpenAI's ``TranscriptionDiarized``,
and that exactness is a compatibility asset: an OpenAI SDK client can point at
this server and parse the typed object. Adding vendor-shaped formats must not
spend it (ADR 0010).

These are golden-byte assertions rather than shape assertions on purpose. A
field added to the internal ``TranscriptionResult`` — ``overlap`` was the most
recent — must not leak into an OpenAI projection, and only comparing the
serialized bytes catches that.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from coro.core.models import (
    ResponseSegment,
    TranscriptionResult,
    TranscriptItem,
    TranscriptWord,
)
from conftest import make_app, make_wav

_WORDS = [
    TranscriptWord(word="hola", start=0.0, end=0.5, score=0.91, speaker="1"),
    TranscriptWord(word="mundo", start=0.5, end=1.0, score=0.83, speaker="2"),
]

# Frozen wire bytes for the fixture below. Regenerate only alongside a recorded
# decision to change the OpenAI-compatible contract.
_GOLDEN = {
    "json": '{"text":"hola mundo","usage":{"type":"duration","seconds":1}}',
    "verbose_json": (
        '{"duration":1.0,"language":"es","text":"hola mundo","segments":['
        '{"id":0,"seek":0,"start":0.0,"end":0.5,"text":"hola","tokens":[],'
        '"temperature":0.0,"avg_logprob":0.0,"compression_ratio":0.0,"no_speech_prob":0.0},'
        '{"id":1,"seek":50,"start":0.5,"end":1.0,"text":"mundo","tokens":[],'
        '"temperature":0.0,"avg_logprob":0.0,"compression_ratio":0.0,"no_speech_prob":0.0}],'
        '"words":[{"word":"hola","start":0.0,"end":0.5},'
        '{"word":"mundo","start":0.5,"end":1.0}],'
        '"usage":{"type":"duration","seconds":1}}'
    ),
    "diarized_json": (
        '{"task":"transcribe","duration":1.0,"text":"hola mundo","segments":['
        '{"type":"transcript.text.segment","id":"seg_001","start":0.0,"end":0.5,'
        '"text":"hola","speaker":"1"},'
        '{"type":"transcript.text.segment","id":"seg_002","start":0.5,"end":1.0,'
        '"text":"mundo","speaker":"2"}],'
        '"usage":{"type":"duration","seconds":1}}'
    ),
}


class _FakePipeline:
    async def transcribe(self, audio, *, language=None, prompt=None):
        return TranscriptionResult(
            segments=[
                ResponseSegment(start=0.0, end=0.5, text="hola", speaker="1", words=_WORDS[:1]),
                ResponseSegment(start=0.5, end=1.0, text="mundo", speaker="2", words=_WORDS[1:]),
            ],
            word_segments=_WORDS,
            transcript=[
                TranscriptItem(start=0.0, end=0.5, text="hola"),
                TranscriptItem(start=0.5, end=1.0, text="mundo"),
            ],
            diarization=[],
            raw_words=[],
        )


async def _raw_body(fmt: str) -> str:
    app = make_app(_FakePipeline())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("t.wav", make_wav(), "audio/wav")},
            data={"response_format": fmt, "language": "es"},
        )
    assert response.status_code == 200, response.text
    return response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", sorted(_GOLDEN))
async def test_openai_format_bytes_are_unchanged(fmt: str):
    assert await _raw_body(fmt) == _GOLDEN[fmt]


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", sorted(_GOLDEN))
async def test_openai_formats_carry_no_per_word_speaker(fmt: str):
    # The vendor formats exist precisely because these have no slot for one.
    body = await _raw_body(fmt)
    assert "speaker_confidence" not in body
    assert '"overlap"' not in body


@pytest.mark.asyncio
@pytest.mark.parametrize("alias,canonical", [("json_verbose", "verbose_json"), ("dirized_json", "diarized_json")])
async def test_typo_aliases_stay_byte_identical_to_their_canonical_format(alias, canonical):
    assert await _raw_body(alias) == _GOLDEN[canonical]
