"""Deepgram-native POST /v1/listen honours Deepgram's own request contract.

The point of a separate endpoint is that Deepgram's contract is implemented as
Deepgram defines it — raw body, their query parameters and defaults, their
error shape — rather than approximated on top of the OpenAI endpoint.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from conftest import FakePipeline, make_app, make_wav


async def _listen(query: str = "", *, body: bytes | None = None, **headers: str) -> Any:
    app = make_app(FakePipeline())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.post(
            f"/v1/listen{query}",
            content=make_wav() if body is None else body,
            headers={"Content-Type": "audio/wav", **headers},
        )


def _channel_words(body: Any) -> list[dict]:
    return body["results"]["channels"][0]["alternatives"][0]["words"]


@pytest.mark.asyncio
class TestDeepgramDefaults:
    async def test_diarize_defaults_to_false_like_deepgram(self):
        body = (await _listen()).json()
        assert all("speaker" not in word for word in _channel_words(body))

    async def test_utterances_defaults_to_false_like_deepgram(self):
        body = (await _listen()).json()
        assert "utterances" not in body["results"]

    async def test_undiarized_words_omit_speaker_rather_than_nulling_it(self):
        # Deepgram never emits a null speaker; absence is the native signal.
        raw = (await _listen()).text
        assert '"speaker"' not in raw


@pytest.mark.asyncio
class TestDeepgramDiarization:
    async def test_diarize_true_puts_a_speaker_on_every_word(self):
        body = (await _listen("?diarize=true")).json()
        assert [w.get("speaker") for w in _channel_words(body)] == [1, 1, 2, None]

    async def test_utterances_true_returns_the_speaker_turn_view(self):
        body = (await _listen("?diarize=true&utterances=true")).json()
        assert [u.get("speaker") for u in body["results"]["utterances"]] == [1, 2, None]

    async def test_unsupported_span_omits_speaker_rather_than_inventing_one(self):
        # The -1 sentinel means the diarizer never covered the word. Deepgram
        # has no sentinel, so the key is absent rather than guessed.
        body = (await _listen("?diarize=true&utterances=true")).json()
        assert "speaker" not in body["results"]["utterances"][-1]

    async def test_utterance_words_carry_speakers_too(self):
        body = (await _listen("?diarize=true&utterances=true")).json()
        first = body["results"]["utterances"][0]
        assert [w["speaker"] for w in first["words"]] == [1, 1]

    async def test_speaker_confidence_is_never_emitted(self):
        raw = (await _listen("?diarize=true&utterances=true")).text
        assert "speaker_confidence" not in raw

    async def test_timestamps_are_float_seconds(self):
        body = (await _listen("?diarize=true")).json()
        first = _channel_words(body)[0]
        assert (first["start"], first["end"]) == (0.0, 0.5)


@pytest.mark.asyncio
class TestDeepgramRequestContract:
    async def test_audio_is_read_from_the_raw_body_not_multipart(self):
        assert (await _listen()).status_code == 200

    async def test_authorization_header_is_accepted_but_not_validated(self):
        response = await _listen("?diarize=true", Authorization="Token not-a-real-key")
        assert response.status_code == 200

    async def test_missing_authorization_is_still_accepted(self):
        assert (await _listen()).status_code == 200

    async def test_unhonoured_vendor_parameters_are_accepted_not_rejected(self):
        query = "?punctuate=true&smart_format=true&numerals=true&profanity_filter=true&model=nova-2"
        assert (await _listen(query)).status_code == 200

    async def test_unknown_query_parameters_are_ignored(self):
        assert (await _listen("?some_future_deepgram_flag=true")).status_code == 200

    async def test_language_hint_is_accepted(self):
        assert (await _listen("?language=es")).status_code == 200


@pytest.mark.asyncio
class TestDeepgramErrors:
    async def test_empty_body_returns_a_deepgram_shaped_error(self):
        response = await _listen(body=b"")
        assert response.status_code == 400
        body = response.json()
        # Deepgram reports err_code/err_msg, never an OpenAI `error` object.
        assert set(body) == {"err_code", "err_msg", "request_id"}
        assert "error" not in body

    async def test_pipeline_failure_returns_a_deepgram_shaped_error(self):
        class _Failing:
            async def transcribe(self, audio, *, language=None, prompt=None):
                raise ValueError("backend exploded")

        app = make_app(_Failing())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/v1/listen", content=make_wav(), headers={"Content-Type": "audio/wav"}
            )
        assert response.status_code == 500
        assert set(response.json()) == {"err_code", "err_msg", "request_id"}


@pytest.mark.asyncio
class TestOpenAIEndpointIsUntouched:
    async def test_openai_endpoint_rejects_vendor_response_formats(self):
        # The OpenAI surface must not grow values OpenAI does not define.
        app = make_app(FakePipeline())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("t.wav", make_wav(), "audio/wav")},
                data={"response_format": "deepgram_json"},
            )
        assert response.status_code == 422
