"""Cycle 9: v1 pipeline orchestration with fake ASR and diarization adapters.

Tests verify that the V1Pipeline:
- Calls the ASR adapter with PCM bytes converted from the audio input.
- Calls the optional diarization adapter.
- Passes resulting tokens and timeline to the core response builder.
- Returns a valid WhisperX-style response dict.
- Propagates prompt to ASR adapter.
- Handles empty token output without crashing.

The audio conversion step (ffmpeg) is mocked so tests run without
subprocess or a real audio container.
"""

from __future__ import annotations

import struct
from unittest.mock import AsyncMock, patch

import pytest

from asr_diar_server.core.types import SpeakerSegment, TranscriptToken
from asr_diar_server.pipelines.v1 import V1Pipeline

WHISPERX_KEYS = {"segments", "word_segments", "transcript", "diarization", "raw_words"}

_FAKE_PCM = struct.pack("<1600h", *([0] * 1600))  # 100 ms silence, already PCM


def _mock_convert(return_value: bytes):
    """Return a context manager that patches convert_to_pcm_bytes."""
    return patch(
        "asr_diar_server.pipelines.v1.convert_to_pcm_bytes",
        new=AsyncMock(return_value=return_value),
    )


class _FakeASRAdapter:
    """Returns a fixed token list from transcribe_pcm."""

    def __init__(self, tokens=None):
        self._tokens = tokens or []
        self.last_prompt = None
        self.last_language = None

    async def transcribe_pcm(self, pcm_bytes: bytes, *, language=None, prompt=None):
        self.last_prompt = prompt
        self.last_language = language
        return list(self._tokens)


class _FakeDiarizationAdapter:
    def __init__(self, timeline=None):
        self._timeline = timeline or []

    async def diarize_pcm(self, pcm_bytes: bytes):
        return list(self._timeline)


@pytest.mark.asyncio
async def test_v1_pipeline_returns_whisperx_shape():
    """V1Pipeline.run returns all WhisperX-style response keys."""
    pipeline = V1Pipeline(asr=_FakeASRAdapter(), diarization=None)
    with _mock_convert(_FAKE_PCM):
        result = await pipeline.run(b"audio")
    assert WHISPERX_KEYS.issubset(result.keys())


@pytest.mark.asyncio
async def test_v1_pipeline_passes_prompt_to_asr():
    """run() forwards prompt parameter to the ASR adapter."""
    asr = _FakeASRAdapter()
    pipeline = V1Pipeline(asr=asr, diarization=None)
    with _mock_convert(_FAKE_PCM):
        await pipeline.run(b"audio", prompt="test prompt")
    assert asr.last_prompt == "test prompt"


@pytest.mark.asyncio
async def test_v1_pipeline_passes_language_to_asr():
    """run() forwards language parameter to the ASR adapter."""
    asr = _FakeASRAdapter()
    pipeline = V1Pipeline(asr=asr, diarization=None)
    with _mock_convert(_FAKE_PCM):
        await pipeline.run(b"audio", language="es")
    assert asr.last_language == "es"


@pytest.mark.asyncio
async def test_v1_pipeline_uses_diarization_when_provided():
    """V1Pipeline calls diarization adapter and includes speaker info in response."""
    tokens = [TranscriptToken(start=0.0, end=1.0, text=" hola.", probability=0.9)]
    timeline = [SpeakerSegment(start=0.0, end=2.0, speaker=1)]
    asr = _FakeASRAdapter(tokens=tokens)
    diar = _FakeDiarizationAdapter(timeline=timeline)
    pipeline = V1Pipeline(asr=asr, diarization=diar)
    with _mock_convert(_FAKE_PCM):
        result = await pipeline.run(b"audio")
    seg = result["segments"][0]
    assert seg["speaker"] == "1"


@pytest.mark.asyncio
async def test_v1_pipeline_empty_tokens_no_crash():
    """V1Pipeline returns valid empty response when ASR returns no tokens."""
    pipeline = V1Pipeline(asr=_FakeASRAdapter(tokens=[]), diarization=None)
    with _mock_convert(_FAKE_PCM):
        result = await pipeline.run(b"audio")
    assert result["segments"] == []
    assert result["raw_words"] == []
