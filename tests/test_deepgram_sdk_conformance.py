"""Deepgram SDK conformance: /v1/listen validates against Deepgram's own types.

A vendor-named endpoint whose responses that vendor's SDK cannot parse is a
liability, since clients will point their SDK at it (ADR 0010). These tests are
the mechanical check, mirroring ``test_openai_sdk_conformance`` for the OpenAI
endpoint: they parse with the real published SDK types, not a local copy.
"""

from __future__ import annotations

from typing import Any

import pytest
from deepgram.types.listen_v1response import ListenV1Response
from httpx import ASGITransport, AsyncClient

from conftest import FakePipeline, make_app, make_wav


async def _listen(query: str = "") -> Any:
    app = make_app(FakePipeline())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            f"/v1/listen{query}", content=make_wav(), headers={"Content-Type": "audio/wav"}
        )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.asyncio
async def test_default_response_validates_against_deepgram_sdk():
    ListenV1Response.model_validate(await _listen())


@pytest.mark.asyncio
async def test_diarized_response_validates_against_deepgram_sdk():
    parsed = ListenV1Response.model_validate(await _listen("?diarize=true&utterances=true"))
    assert parsed.results.utterances
    assert [u.speaker for u in parsed.results.utterances] == [1, 2, None]
    words = parsed.results.utterances[0].words
    assert words is not None
    assert words[0].speaker == 1
    # Typed by the SDK, deliberately not emitted by coro.
    assert words[0].speaker_confidence is None


@pytest.mark.asyncio
async def test_metadata_required_fields_are_populated():
    parsed = ListenV1Response.model_validate(await _listen())
    assert parsed.metadata.request_id
    assert len(parsed.metadata.sha256) == 64
    assert parsed.metadata.channels == 1
    assert parsed.metadata.duration > 0
    assert parsed.metadata.models


@pytest.mark.asyncio
async def test_metadata_digest_matches_the_submitted_audio():
    import hashlib

    body = await _listen()
    assert body["metadata"]["sha256"] == hashlib.sha256(make_wav()).hexdigest()
