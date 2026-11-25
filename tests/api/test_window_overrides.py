"""Per-request ASR Windowing overrides on both vendor endpoints.

Requests run through the real pipelines with an in-memory ASR Adapter that
hears one word 0.1 s into every window, so the word start times in the
response show where each window began.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from coro.core.models import TranscriptToken
from coro.pipelines.full_memory import FullMemoryPipeline
from coro.pipelines.streaming import StreamingPipeline
from support.adapters import RecordingASR
from support.factories import make_app, make_wav

# 2.5 s: one window at the 30 s default, three 1 s windows with no overlap.
UPLOAD = make_wav(frames=40000)
ONE_SECOND_WINDOWS = {"coro_window_seconds": "1.0", "coro_overlap_seconds": "0"}


def _app(pipeline=None):
    asr = RecordingASR([TranscriptToken(start=0.1, end=0.4, text=" hola", probability=1.0)])
    return make_app(pipeline or FullMemoryPipeline(asr=asr))


async def _openai(app, data: dict):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("call.wav", UPLOAD, "audio/wav")},
            data={"response_format": "verbose_json", **data},
        )


async def _deepgram(app, query: str):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.post(
            f"/v1/listen{query}", content=UPLOAD, headers={"Content-Type": "audio/wav"}
        )


def _openai_starts(response) -> list[float]:
    return [word["start"] for word in response.json()["words"]]


def _deepgram_starts(response) -> list[float]:
    words = response.json()["results"]["channels"][0]["alternatives"][0]["words"]
    return [word["start"] for word in words]


@pytest.mark.asyncio
class TestOpenAIEndpoint:
    async def test_without_overrides_the_server_window_applies(self):
        response = await _openai(_app(), {})

        assert response.status_code == 200
        assert _openai_starts(response) == [0.1]

    async def test_window_and_overlap_override_the_server_window(self):
        response = await _openai(_app(), ONE_SECOND_WINDOWS)

        assert response.status_code == 200
        assert _openai_starts(response) == [0.1, 1.1, 2.1]

    async def test_an_override_does_not_change_later_requests(self):
        app = _app()
        await _openai(app, ONE_SECOND_WINDOWS)

        response = await _openai(app, {})

        assert _openai_starts(response) == [0.1]

    async def test_overlap_not_shorter_than_window_is_a_400(self):
        response = await _openai(_app(), {"coro_window_seconds": "2", "coro_overlap_seconds": "2"})

        assert response.status_code == 400
        assert response.json()["error"]["param"] == "coro_overlap_seconds"

    async def test_non_positive_window_is_a_400(self):
        response = await _openai(_app(), {"coro_window_seconds": "0"})

        assert response.status_code == 400
        assert response.json()["error"]["param"] == "coro_window_seconds"


@pytest.mark.asyncio
class TestDeepgramEndpoint:
    async def test_without_overrides_the_server_window_applies(self):
        response = await _deepgram(_app(), "")

        assert response.status_code == 200
        assert _deepgram_starts(response) == [0.1]

    async def test_window_and_overlap_override_the_server_window(self):
        response = await _deepgram(_app(), "?coro_window_seconds=1.0&coro_overlap_seconds=0")

        assert response.status_code == 200
        assert _deepgram_starts(response) == [0.1, 1.1, 2.1]

    async def test_streaming_pipeline_honours_the_override(self, tmp_path):
        asr = RecordingASR([TranscriptToken(start=0.1, end=0.4, text=" hola", probability=1.0)])
        app = _app(StreamingPipeline(asr=asr, spill_dir=str(tmp_path)))

        response = await _deepgram(app, "?coro_window_seconds=1.0&coro_overlap_seconds=0")

        assert response.status_code == 200
        assert _deepgram_starts(response) == [0.1, 1.1, 2.1]

    async def test_overlap_alone_is_checked_against_the_server_window(self):
        response = await _deepgram(_app(), "?coro_overlap_seconds=30")

        assert response.status_code == 400
        assert "coro_overlap_seconds" in response.json()["err_msg"]

    async def test_negative_overlap_is_a_400(self):
        response = await _deepgram(_app(), "?coro_overlap_seconds=-1")

        assert response.status_code == 400
        assert response.json()["err_code"] == "Bad Request"
