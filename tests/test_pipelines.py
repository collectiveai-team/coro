"""Behavior-named Transcription Pipeline tests."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from asr_diar_server.audio import AudioInput
from asr_diar_server.core.types import TranscriptDeltaEvent, TranscriptDoneEvent, TranscriptToken
from asr_diar_server.pipelines.chunked_file import ChunkedFilePipeline
from asr_diar_server.pipelines.full_memory import FullMemoryPipeline


class _FakeASR:
    async def transcribe_pcm(self, pcm, *, language=None, prompt=None):
        return [TranscriptToken(start=0.0, end=0.5, text=" hello.", probability=1.0)]


@pytest.mark.asyncio
async def test_full_memory_pipeline_uses_audio_input_bytes_and_windowing():
    pipeline = FullMemoryPipeline(asr=_FakeASR())
    audio = AudioInput(b"encoded")

    with patch(
        "asr_diar_server.pipelines.full_memory.convert_to_pcm_bytes",
        new=AsyncMock(return_value=b"\x00\x00" * 16000),
    ) as convert:
        result = await pipeline.transcribe(audio, prompt="hint")

    convert.assert_awaited_once_with(b"encoded")
    assert result["segments"][0]["text"] == "hello."


@pytest.mark.asyncio
async def test_chunked_file_pipeline_uses_audio_input_temp_path_and_cleans_up():
    pipeline = ChunkedFilePipeline(asr=_FakeASR())
    audio = AudioInput(b"encoded")

    async def fake_stream(_path: str, chunk_seconds: float = 1.0):
        yield b"\x00\x00" * 16000

    with patch("asr_diar_server.pipelines.chunked_file.stream_pcm_from_file", new=fake_stream):
        result = await pipeline.transcribe(audio)

    assert result["segments"][0]["text"] == "hello."


@pytest.mark.asyncio
async def test_pipeline_stream_emits_delta_and_done():
    pipeline = FullMemoryPipeline(asr=_FakeASR())
    audio = AudioInput(b"encoded")

    with patch(
        "asr_diar_server.pipelines.full_memory.convert_to_pcm_bytes",
        new=AsyncMock(return_value=b"\x00\x00" * 16000),
    ):
        events = [event async for event in pipeline.stream(audio)]

    assert isinstance(events[0], TranscriptDeltaEvent)
    assert events[0].delta == "hello."
    assert isinstance(events[-1], TranscriptDoneEvent)
    assert json.loads(events[-1].text)["segments"][0]["text"] == "hello."
