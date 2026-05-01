"""Cycle 10: v2 disk-backed pipeline orchestration with fake adapters.

Tests verify that V2Pipeline:
- Accepts a file path and uses stream_pcm_from_file to read it.
- Calls ASR adapter with each PCM window chunk.
- Calls diarization adapter per chunk.
- Returns a valid WhisperX-style response dict.
- Propagates prompt across ASR chunks.
- Handles empty token output without crashing.

The ffmpeg streaming step is mocked so tests run without subprocess.
"""

from __future__ import annotations

import struct
from unittest.mock import patch

import pytest

from asr_diar_server.core.types import SpeakerSegment, TranscriptToken
from asr_diar_server.pipelines.v2 import V2Pipeline

WHISPERX_KEYS = {"segments", "word_segments", "transcript", "diarization", "raw_words"}

_FAKE_PCM = struct.pack("<1600h", *([0] * 1600))  # 100 ms 16-bit silence


class _FakeASRAdapter:
    def __init__(self, tokens=None):
        self._tokens = tokens or []
        self.call_count = 0
        self.last_prompt = None

    async def transcribe_pcm(self, pcm_bytes, *, language=None, prompt=None):
        self.call_count += 1
        self.last_prompt = prompt
        return list(self._tokens)


class _FakeDiarizationAdapter:
    def __init__(self, timeline=None):
        self._timeline = timeline or []

    async def diarize_pcm(self, pcm_bytes):
        return list(self._timeline)


async def _single_chunk_stream(path: str, chunk_seconds: float = 1.0):
    """Fake stream_pcm_from_file that yields one chunk and stops."""
    yield _FAKE_PCM


def _mock_stream():
    return patch(
        "asr_diar_server.pipelines.v2.stream_pcm_from_file",
        new=_single_chunk_stream,
    )


@pytest.mark.asyncio
async def test_v2_pipeline_returns_whisperx_shape():
    """V2Pipeline.run_from_path returns all WhisperX-style response keys."""
    pipeline = V2Pipeline(asr=_FakeASRAdapter(), diarization=None)
    with _mock_stream():
        result = await pipeline.run_from_path("/fake/path.wav")
    assert WHISPERX_KEYS.issubset(result.keys())


@pytest.mark.asyncio
async def test_v2_pipeline_passes_prompt_to_asr():
    """run_from_path() forwards prompt to the ASR adapter."""
    asr = _FakeASRAdapter()
    pipeline = V2Pipeline(asr=asr, diarization=None)
    with _mock_stream():
        await pipeline.run_from_path("/fake/path.wav", prompt="mi prompt")
    assert asr.last_prompt == "mi prompt"


@pytest.mark.asyncio
async def test_v2_pipeline_uses_diarization_when_provided():
    """V2Pipeline calls diarization adapter when it is set."""
    tokens = [TranscriptToken(start=0.0, end=1.0, text=" hola.", probability=0.9)]
    timeline = [SpeakerSegment(start=0.0, end=2.0, speaker=2)]
    asr = _FakeASRAdapter(tokens=tokens)
    diar = _FakeDiarizationAdapter(timeline=timeline)
    pipeline = V2Pipeline(asr=asr, diarization=diar)
    with _mock_stream():
        result = await pipeline.run_from_path("/fake/path.wav")
    seg = result["segments"][0]
    assert seg["speaker"] == "2"


@pytest.mark.asyncio
async def test_v2_pipeline_empty_tokens_no_crash():
    """V2Pipeline returns valid empty response when ASR returns no tokens."""
    pipeline = V2Pipeline(asr=_FakeASRAdapter(tokens=[]), diarization=None)
    with _mock_stream():
        result = await pipeline.run_from_path("/fake/path.wav")
    assert result["segments"] == []
    assert result["raw_words"] == []
