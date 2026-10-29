"""Behavior-named Transcription Pipeline tests.

The upload is a real WAV file decoded by ffmpeg, as in production; the ASR
Adapter is the in-memory ``RecordingASR``.
"""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

import pytest

from coro.audio import BYTES_PER_SAMPLE, SAMPLE_RATE, AudioInput
from coro.core.models import TranscriptDeltaEvent, TranscriptDoneEvent, TranscriptToken
from coro.pipelines.full_memory import FullMemoryPipeline
from support.adapters import RecordingASR
from support.factories import make_wav

ONE_SECOND_PCM_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE
ONE_SECOND_WAV = make_wav(frames=SAMPLE_RATE)


class _Upload:
    """An ``UploadReader`` over a named in-memory file, as Starlette's ``UploadFile``."""

    def __init__(self, data: bytes, filename: str) -> None:
        self._body = io.BytesIO(data)
        self.filename = filename

    async def read(self, size: int = -1, /) -> bytes:
        return self._body.read(size)


def _asr() -> RecordingASR:
    return RecordingASR([TranscriptToken(start=0.0, end=0.5, text=" hello.", probability=1.0)])


async def _spooled_upload() -> AudioInput:
    return await AudioInput.from_upload(_Upload(ONE_SECOND_WAV, "clip.wav"))


@pytest.mark.asyncio
async def test_full_memory_pipeline_decodes_the_spooled_upload_and_windows_it():
    asr = _asr()
    audio = await _spooled_upload()

    result = await FullMemoryPipeline(asr=asr).transcribe(audio, prompt="hint")

    assert sum(asr.window_sizes) == ONE_SECOND_PCM_BYTES
    assert asr.prompts == ["hint"] * len(asr.window_sizes)
    assert result.segments[0].text == "hello."


@pytest.mark.asyncio
async def test_full_memory_pipeline_removes_the_spooled_upload():
    audio = await _spooled_upload()
    spooled = Path(await audio.temp_path())

    await FullMemoryPipeline(asr=_asr()).transcribe(audio)

    assert spooled.name.startswith("asr-upload-")
    assert not await asyncio.to_thread(spooled.exists)


@pytest.mark.asyncio
async def test_pipeline_stream_emits_delta_and_done():
    pipeline = FullMemoryPipeline(asr=_asr())
    audio = AudioInput(ONE_SECOND_WAV)

    events = [event async for event in pipeline.stream(audio)]

    assert isinstance(events[0], TranscriptDeltaEvent)
    assert events[0].delta == "hello."
    assert isinstance(events[-1], TranscriptDoneEvent)
    assert json.loads(events[-1].text)["segments"][0]["text"] == "hello."
