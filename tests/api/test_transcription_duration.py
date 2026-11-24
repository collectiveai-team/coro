"""File transcription reports decoded audio duration independently of word timings."""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from coro.core.models import TranscriptToken
from coro.pipelines.full_memory import FullMemoryPipeline
from coro.pipelines.streaming import StreamingPipeline
from support.adapters import RecordingASR
from support.factories import make_app, make_wav

_AUDIO_DURATION = 2.3125


@pytest.fixture
def upload(recognized_word) -> bytes:
    """Create 8 kHz audio with silence after a short signal, or only silence."""
    # A 400 Hz square wave from 0.1 to 0.5 seconds stands in for the early word.
    signal = bytes(1_600) + (b"\x00\x20" * 10 + b"\x00\xe0" * 10) * 160
    return make_wav(frames=18_500, rate=8_000, pcm_prefix=signal if recognized_word else b"")


@pytest.fixture(params=[False, True], ids=["no-words", "trailing-silence"])
def recognized_word(request) -> bool:
    """Choose whether the ASR adapter recognizes an early word or nothing."""
    return request.param


@pytest.fixture(params=["full-memory", "streaming"])
def app(request, recognized_word, tmp_path):
    """Exercise the actual FFmpeg decoding paths with deterministic ASR output."""
    tokens = [TranscriptToken(start=0.1, end=0.5, text="hello")] if recognized_word else []
    asr = RecordingASR(tokens)
    if request.param == "full-memory":
        return make_app(FullMemoryPipeline(asr=asr))
    return make_app(StreamingPipeline(asr=asr, spill_dir=str(tmp_path)))


@pytest.mark.asyncio
@pytest.mark.parametrize("response_format", ["json", "verbose_json", "diarized_json"])
async def test_openai_duration_and_usage_report_decoded_audio(
    app, upload, recognized_word, response_format
):
    """Keep word and segment boundaries separate from duration and rounded usage."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", upload, "audio/wav")},
            data={"response_format": response_format},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["text"] == ("hello" if recognized_word else "")
    assert body["usage"] == {"type": "duration", "seconds": 3}
    expected_duration = (
        None if response_format == "json" else pytest.approx(_AUDIO_DURATION, abs=1e-9)
    )
    assert body.get("duration") == expected_duration
    has_segments = recognized_word and response_format != "json"
    assert [segment["end"] for segment in body.get("segments", [])] == (
        [0.5] if has_segments else []
    )
    has_words = recognized_word and response_format == "verbose_json"
    assert body.get("words", []) == (
        [{"word": "hello", "start": 0.1, "end": 0.5}] if has_words else []
    )


@pytest.mark.asyncio
async def test_openai_done_frame_reports_decoded_audio_duration(app, upload, recognized_word):
    """Report the same duration when the file transcription is served via SSE."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", upload, "audio/wav")},
            data={"stream": "true"},
        )
    assert response.status_code == 200, response.text
    *_, done, terminator = response.text.strip().split("\n\n")
    assert terminator == "data: [DONE]"
    event = json.loads(done.removeprefix("data: "))
    assert event["type"] == "transcript.text.done"
    body = json.loads(event["text"])
    assert body["duration"] == pytest.approx(_AUDIO_DURATION, abs=1e-9)
    assert [word["end"] for word in body["word_segments"]] == ([0.5] if recognized_word else [])


@pytest.mark.asyncio
async def test_deepgram_metadata_reports_decoded_audio_duration(app, upload, recognized_word):
    """Report the full audio length without extending Deepgram word timings."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/listen", content=upload, headers={"Content-Type": "audio/wav"}
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["metadata"]["duration"] == pytest.approx(_AUDIO_DURATION, abs=1e-9)
    words = body["results"]["channels"][0]["alternatives"][0]["words"]
    assert [word["end"] for word in words] == ([0.5] if recognized_word else [])
