"""Both vendor routes still declare a Content-Length.

Response bodies are rendered incrementally and streamed from a spool, which
would ordinarily mean chunked transfer-encoding and no length: you cannot state
a size you have not finished computing. Spooling to disk first is what buys the
length back, and it was a deliberate choice — both vendors' own servers declare
one on non-streaming JSON, and the difference would be invisible to the oasdiff
gate, which sees no success schema for either route at all (ADR 0013, ADR 0018).

Nothing else pins that, so this does. The assertion is on the *declared* length
matching the delivered body, not merely on the header's presence, because a
wrong length is worse than an absent one.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from coro.core.models import (
    RawWord,
    ResponseSegment,
    TranscriptionResult,
    TranscriptItem,
    TranscriptWord,
)
from support.factories import make_app, make_wav

_WORDS = [
    TranscriptWord(word="hola", start=0.0, end=0.5, score=0.91, speaker="1"),
    TranscriptWord(word="mundo", start=0.5, end=1.0, score=0.83, speaker="2"),
]


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
            raw_words=[RawWord(word="hola", start=0.0, end=0.5, score=0.91)],
        )


async def _post(path: str, **kwargs):
    app = make_app(_FakePipeline())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.post(path, **kwargs)


@pytest.mark.parametrize("fmt", ["json", "verbose_json", "diarized_json"])
@pytest.mark.asyncio
async def test_transcription_declares_the_body_length(fmt: str):
    response = await _post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", make_wav(), "audio/wav")},
        data={"response_format": fmt},
    )

    assert response.status_code == 200
    assert response.headers["content-length"] == str(len(response.content))
    assert "transfer-encoding" not in response.headers


@pytest.mark.parametrize("query", ["", "?diarize=true&utterances=true"])
@pytest.mark.asyncio
async def test_listen_declares_the_body_length(query: str):
    response = await _post(
        f"/v1/listen{query}",
        content=make_wav(),
        headers={"content-type": "audio/wav"},
    )

    assert response.status_code == 200
    assert response.headers["content-length"] == str(len(response.content))
    assert "transfer-encoding" not in response.headers
