"""Diarization runs only when the response renders speakers.

Requests run through the real pipelines with in-memory ASR and diarization
adapters that record each call, so the assertions observe whether the
diarizer was used for a request and that the body does not change.
"""

# botica: ignore-file[test-through-interface] -- same body by design; the fakes record calls

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from coro.core.models import SpeakerSegment, TranscriptToken
from coro.pipelines.full_memory import FullMemoryPipeline
from coro.pipelines.streaming import StreamingPipeline
from support.adapters import FixedDiarization, RecordingASR, RecordingStreamingDiarizerFactory
from support.factories import make_app, make_wav

UPLOAD = make_wav(frames=16000)
TIMELINE = [SpeakerSegment(start=0.0, end=1.0, speaker=0)]


def _asr() -> RecordingASR:
    return RecordingASR(
        [
            TranscriptToken(start=0.0, end=0.4, text=" hola", probability=0.9),
            TranscriptToken(start=0.5, end=1.0, text=" mundo.", probability=0.8),
        ]
    )


def _full_memory() -> tuple[FullMemoryPipeline, FixedDiarization]:
    diarization = FixedDiarization(TIMELINE)
    return FullMemoryPipeline(asr=_asr(), diarization=diarization), diarization


async def _openai(app, data: dict):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("call.wav", UPLOAD, "audio/wav")},
            data=data,
        )


async def _deepgram(app, query: str = ""):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.post(
            f"/v1/listen{query}", content=UPLOAD, headers={"Content-Type": "audio/wav"}
        )


@pytest.mark.asyncio
class TestOpenAIEndpoint:
    @pytest.mark.parametrize("response_format", ["json", "verbose_json"])
    async def test_formats_without_speakers_skip_diarization(self, response_format):
        pipeline, diarization = _full_memory()

        response = await _openai(make_app(pipeline), {"response_format": response_format})

        assert response.status_code == 200
        assert diarization.diarized_bytes == []

    async def test_diarized_json_runs_diarization(self):
        pipeline, diarization = _full_memory()

        response = await _openai(make_app(pipeline), {"response_format": "diarized_json"})

        assert len(diarization.diarized_bytes) == 1
        assert response.json()["segments"][0]["speaker"] == "0"

    async def test_streaming_runs_diarization_for_the_done_frame(self):
        pipeline, diarization = _full_memory()

        response = await _openai(make_app(pipeline), {"stream": "true"})

        assert response.status_code == 200
        assert len(diarization.diarized_bytes) == 1

    @pytest.mark.parametrize("response_format", ["json", "verbose_json"])
    async def test_skipping_diarization_keeps_the_body(self, response_format):
        diarized, _ = _full_memory()
        undiarized = FullMemoryPipeline(asr=_asr())

        with_diarizer = await _openai(make_app(diarized), {"response_format": response_format})
        without = await _openai(make_app(undiarized), {"response_format": response_format})

        assert with_diarizer.json() == without.json()


@pytest.mark.asyncio
class TestDeepgramEndpoint:
    async def test_plain_request_skips_diarization(self):
        pipeline, diarization = _full_memory()

        response = await _deepgram(make_app(pipeline))

        assert response.status_code == 200
        assert diarization.diarized_bytes == []

    @pytest.mark.parametrize("query", ["?diarize=true", "?utterances=true"])
    async def test_speaker_views_run_diarization(self, query):
        pipeline, diarization = _full_memory()

        response = await _deepgram(make_app(pipeline), query)

        assert response.status_code == 200
        assert len(diarization.diarized_bytes) == 1

    async def test_streaming_pipeline_creates_no_diarizer_for_a_plain_request(self, tmp_path):
        factory = RecordingStreamingDiarizerFactory(TIMELINE)
        pipeline = StreamingPipeline(
            asr=_asr(), streaming_diarizer_factory=factory, spill_dir=str(tmp_path)
        )

        await _deepgram(make_app(pipeline))
        await _deepgram(make_app(pipeline), "?diarize=true")

        assert len(factory.diarizers) == 1
