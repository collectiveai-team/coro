"""Per-request ASR Windowing overrides on both vendor endpoints.

Requests run through the real Full-Memory Pipeline with an in-memory ASR
Adapter that records the size of every window it is asked to transcribe, so
the assertions observe the window geometry the request asked for.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from coro.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
from coro.core.models import TranscriptToken
from coro.pipelines.full_memory import FullMemoryPipeline
from coro.pipelines.streaming import StreamingPipeline
from support.adapters import RecordingASR
from support.factories import make_app, make_wav

BYTES_PER_SECOND = SAMPLE_RATE * BYTES_PER_SAMPLE
# 2.5 s: one window at the 30 s default, three 1 s windows with no overlap.
UPLOAD = make_wav(frames=40000)


def _asr() -> RecordingASR:
    return RecordingASR([TranscriptToken(start=0.0, end=0.4, text=" hola", probability=1.0)])


async def _openai(app, data: dict):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("call.wav", UPLOAD, "audio/wav")},
            data=data,
        )


async def _deepgram(app, query: str):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.post(
            f"/v1/listen{query}", content=UPLOAD, headers={"Content-Type": "audio/wav"}
        )


@pytest.mark.asyncio
class TestOpenAIEndpoint:
    async def test_without_overrides_the_server_window_applies(self):
        asr = _asr()
        response = await _openai(make_app(FullMemoryPipeline(asr=asr)), {})

        assert response.status_code == 200
        assert len(asr.window_sizes) == 1

    async def test_window_and_overlap_override_the_server_window(self):
        asr = _asr()
        response = await _openai(
            make_app(FullMemoryPipeline(asr=asr)),
            {"coro_window_seconds": "1.0", "coro_overlap_seconds": "0"},
        )

        assert response.status_code == 200
        assert len(asr.window_sizes) == 3
        assert max(asr.window_sizes) == BYTES_PER_SECOND

    async def test_an_override_does_not_change_later_requests(self):
        asr = _asr()
        app = make_app(FullMemoryPipeline(asr=asr))
        await _openai(app, {"coro_window_seconds": "1.0", "coro_overlap_seconds": "0"})
        asr.window_sizes.clear()

        await _openai(app, {})

        assert len(asr.window_sizes) == 1

    async def test_overlap_not_shorter_than_window_is_a_400(self):
        response = await _openai(
            make_app(FullMemoryPipeline(asr=_asr())),
            {"coro_window_seconds": "2", "coro_overlap_seconds": "2"},
        )

        assert response.status_code == 400
        assert response.json()["error"]["param"] == "coro_overlap_seconds"

    async def test_non_positive_window_is_a_400(self):
        response = await _openai(
            make_app(FullMemoryPipeline(asr=_asr())), {"coro_window_seconds": "0"}
        )

        assert response.status_code == 400
        assert response.json()["error"]["param"] == "coro_window_seconds"


@pytest.mark.asyncio
class TestDeepgramEndpoint:
    async def test_without_overrides_the_server_window_applies(self):
        asr = _asr()
        response = await _deepgram(make_app(FullMemoryPipeline(asr=asr)), "")

        assert response.status_code == 200
        assert len(asr.window_sizes) == 1

    async def test_window_and_overlap_override_the_server_window(self):
        asr = _asr()
        response = await _deepgram(
            make_app(FullMemoryPipeline(asr=asr)),
            "?coro_window_seconds=1.0&coro_overlap_seconds=0",
        )

        assert response.status_code == 200
        assert len(asr.window_sizes) == 3
        assert max(asr.window_sizes) == BYTES_PER_SECOND

    async def test_streaming_pipeline_honours_the_override(self, tmp_path):
        asr = _asr()
        pipeline = StreamingPipeline(asr=asr, spill_dir=str(tmp_path))
        response = await _deepgram(
            make_app(pipeline), "?coro_window_seconds=1.0&coro_overlap_seconds=0"
        )

        assert response.status_code == 200
        assert len(asr.window_sizes) == 3

    async def test_overlap_alone_is_checked_against_the_server_window(self):
        response = await _deepgram(
            make_app(FullMemoryPipeline(asr=_asr())), "?coro_overlap_seconds=30"
        )

        assert response.status_code == 400
        assert "coro_overlap_seconds" in response.json()["err_msg"]

    async def test_negative_overlap_is_a_400(self):
        response = await _deepgram(
            make_app(FullMemoryPipeline(asr=_asr())), "?coro_overlap_seconds=-1"
        )

        assert response.status_code == 400
        assert response.json()["err_code"] == "Bad Request"
