"""Chunked-File Pipeline orchestration with fake adapters.

Tests verify that ChunkedFilePipeline:
- Uses stream_pcm_from_file to read spooled audio.
- Calls ASR adapter with PCM windows.
- Calls diarization adapter when configured.
- Returns a valid WhisperX-style response dict.
- Propagates prompt across ASR chunks.
- Handles empty token output without crashing.

The ffmpeg streaming step is mocked so tests run without subprocess.
"""

from __future__ import annotations

import struct
from unittest.mock import patch

import pytest

from asr_diar_server.audio import AudioInput
from asr_diar_server.core.types import SpeakerSegment, TranscriptToken
from asr_diar_server.pipelines.chunked_file import ChunkedFilePipeline

WHISPERX_KEYS = {"segments", "word_segments", "transcript", "diarization", "raw_words"}

_FAKE_PCM = struct.pack("<1600h", *([0] * 1600))


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
    yield _FAKE_PCM


def _mock_stream():
    return patch(
        "asr_diar_server.pipelines.chunked_file.stream_pcm_from_file",
        new=_single_chunk_stream,
    )


@pytest.mark.asyncio
async def test_chunked_file_pipeline_returns_whisperx_shape():
    pipeline = ChunkedFilePipeline(asr=_FakeASRAdapter(), diarization=None)
    with _mock_stream():
        result = await pipeline.transcribe(AudioInput(b"audio"))
    assert WHISPERX_KEYS.issubset(result.keys())


@pytest.mark.asyncio
async def test_chunked_file_pipeline_passes_prompt_to_asr():
    asr = _FakeASRAdapter()
    pipeline = ChunkedFilePipeline(asr=asr, diarization=None)
    with _mock_stream():
        await pipeline.transcribe(AudioInput(b"audio"), prompt="mi prompt")
    assert asr.last_prompt == "mi prompt"


@pytest.mark.asyncio
async def test_chunked_file_pipeline_uses_diarization_when_provided():
    tokens = [TranscriptToken(start=0.0, end=1.0, text=" hola.", probability=0.9)]
    timeline = [SpeakerSegment(start=0.0, end=2.0, speaker=2)]
    asr = _FakeASRAdapter(tokens=tokens)
    diar = _FakeDiarizationAdapter(timeline=timeline)
    pipeline = ChunkedFilePipeline(asr=asr, diarization=diar)
    with _mock_stream():
        result = await pipeline.transcribe(AudioInput(b"audio"))
    seg = result["segments"][0]
    assert seg["speaker"] == "2"


@pytest.mark.asyncio
async def test_chunked_file_pipeline_empty_tokens_no_crash():
    pipeline = ChunkedFilePipeline(asr=_FakeASRAdapter(tokens=[]), diarization=None)
    with _mock_stream():
        result = await pipeline.transcribe(AudioInput(b"audio"))
    assert result["segments"] == []
    assert result["raw_words"] == []
