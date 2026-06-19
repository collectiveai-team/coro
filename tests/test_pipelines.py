"""Behavior-named Transcription Pipeline tests."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from coro.audio import AudioInput
from coro.core.models import TranscriptDeltaEvent, TranscriptDoneEvent, TranscriptToken
from coro.pipelines.full_memory import FullMemoryPipeline


class _FakeASR:
    async def transcribe_pcm(self, pcm, *, language=None, prompt=None):
        return [TranscriptToken(start=0.0, end=0.5, text=" hello.", probability=1.0)]


@pytest.mark.asyncio
async def test_full_memory_pipeline_uses_audio_input_bytes_and_windowing():
    pipeline = FullMemoryPipeline(asr=_FakeASR())
    audio = AudioInput(b"encoded")

    with patch(
        "coro.pipelines.full_memory.convert_to_pcm_bytes",
        new=AsyncMock(return_value=b"\x00\x00" * 16000),
    ) as convert:
        result = await pipeline.transcribe(audio, prompt="hint")

    convert.assert_awaited_once_with(b"encoded")
    assert result.segments[0].text == "hello."


@pytest.mark.asyncio
async def test_pipeline_stream_emits_delta_and_done():
    pipeline = FullMemoryPipeline(asr=_FakeASR())
    audio = AudioInput(b"encoded")

    with patch(
        "coro.pipelines.full_memory.convert_to_pcm_bytes",
        new=AsyncMock(return_value=b"\x00\x00" * 16000),
    ):
        events = [event async for event in pipeline.stream(audio)]

    assert isinstance(events[0], TranscriptDeltaEvent)
    assert events[0].delta == "hello."
    assert isinstance(events[-1], TranscriptDoneEvent)
    assert json.loads(events[-1].text)["segments"][0]["text"] == "hello."
