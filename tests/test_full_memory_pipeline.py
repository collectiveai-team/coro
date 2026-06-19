"""Full-Memory Pipeline orchestration with fake ASR and diarization adapters.

Tests verify that the FullMemoryPipeline:
- Calls the ASR adapter with PCM bytes converted from the audio input.
- Calls the optional diarization adapter.
- Passes resulting tokens and timeline to the core response builder.
- Returns a valid transcription response dict.
- Propagates prompt to ASR adapter.
- Handles empty token output without crashing.

The audio conversion step (ffmpeg) is mocked so tests run without
subprocess or a real audio container.
"""

from __future__ import annotations

import struct
from dataclasses import asdict
from unittest.mock import AsyncMock, patch

import pytest

from coro.audio import AudioInput
from coro.core.types import SpeakerSegment, TranscriptToken
from coro.pipelines.full_memory import FullMemoryPipeline

RESPONSE_KEYS = {"segments", "word_segments", "transcript", "diarization", "raw_words"}

_FAKE_PCM = struct.pack("<1600h", *([0] * 1600))


def _mock_convert(return_value: bytes):
    """Return a context manager that patches convert_to_pcm_bytes."""
    return patch(
        "coro.pipelines.full_memory.convert_to_pcm_bytes",
        new=AsyncMock(return_value=return_value),
    )


class _FakeASRAdapter:
    """Return a fixed token list from transcribe_pcm."""

    def __init__(self, tokens=None):
        self._tokens = tokens or []
        self.last_prompt = None
        self.last_language = None

    async def transcribe_pcm(
        self, pcm: bytes, *, language: str | None = None, prompt: str | None = None
    ) -> list[TranscriptToken]:
        self.last_prompt = prompt
        self.last_language = language
        return list(self._tokens)


class _FakeDiarizationAdapter:
    def __init__(self, timeline=None):
        self._timeline = timeline or []

    async def diarize_pcm(self, pcm: bytes) -> list[SpeakerSegment]:
        return list(self._timeline)


@pytest.mark.asyncio
async def test_full_memory_pipeline_returns_response_shape():
    pipeline = FullMemoryPipeline(asr=_FakeASRAdapter(), diarization=None)
    with _mock_convert(_FAKE_PCM):
        result = await pipeline.transcribe(AudioInput(b"audio"))
    assert set(asdict(result)) == RESPONSE_KEYS


@pytest.mark.asyncio
async def test_full_memory_pipeline_passes_prompt_to_asr():
    asr = _FakeASRAdapter()
    pipeline = FullMemoryPipeline(asr=asr, diarization=None)
    with _mock_convert(_FAKE_PCM):
        await pipeline.transcribe(AudioInput(b"audio"), prompt="test prompt")
    assert asr.last_prompt == "test prompt"


@pytest.mark.asyncio
async def test_full_memory_pipeline_passes_language_to_asr():
    asr = _FakeASRAdapter()
    pipeline = FullMemoryPipeline(asr=asr, diarization=None)
    with _mock_convert(_FAKE_PCM):
        await pipeline.transcribe(AudioInput(b"audio"), language="es")
    assert asr.last_language == "es"


@pytest.mark.asyncio
async def test_full_memory_pipeline_uses_diarization_when_provided():
    tokens = [TranscriptToken(start=0.0, end=1.0, text=" hola.", probability=0.9)]
    timeline = [SpeakerSegment(start=0.0, end=2.0, speaker=1)]
    asr = _FakeASRAdapter(tokens=tokens)
    diar = _FakeDiarizationAdapter(timeline=timeline)
    pipeline = FullMemoryPipeline(asr=asr, diarization=diar)
    with _mock_convert(_FAKE_PCM):
        result = await pipeline.transcribe(AudioInput(b"audio"))
    seg = result.segments[0]
    assert seg.speaker == "1"


@pytest.mark.asyncio
async def test_full_memory_pipeline_empty_tokens_no_crash():
    pipeline = FullMemoryPipeline(asr=_FakeASRAdapter(tokens=[]), diarization=None)
    with _mock_convert(_FAKE_PCM):
        result = await pipeline.transcribe(AudioInput(b"audio"))
    assert result.segments == []
    assert result.raw_words == []
