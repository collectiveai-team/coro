"""ASR-Only path tests for StreamingPipeline.

Locks down the no-diarization path so changes to diarizer call sites
cannot accidentally regress ASR-only behaviour.
"""

from __future__ import annotations

import json
import struct
from unittest.mock import patch

import pytest

from coro.audio import AudioInput
from coro.core.types import (
    TranscriptDeltaEvent,
    TranscriptToken,
)
from coro.pipelines.done_frame import StreamingDoneFrame
from coro.pipelines.streaming import StreamingPipeline


def _render_done_frame(frame: StreamingDoneFrame) -> dict:
    """Render the done frame's SSE bytes (and close its store), returning the dict."""
    sse = "".join(frame.iter_sse())
    outer = json.loads(sse[len("data: ") :].rstrip("\n"))
    return json.loads(outer["text"])


RESPONSE_KEYS = {"segments", "word_segments", "transcript", "diarization", "raw_words"}

_CHUNK_BYTES = struct.pack("<1600h", *([0] * 1600))
_NUM_CHUNKS = 3


class _FakeASRAdapter:
    def __init__(self, tokens=None):
        self._tokens = tokens or []

    async def transcribe_pcm(
        self, pcm: bytes, *, language: str | None = None, prompt: str | None = None
    ) -> list[TranscriptToken]:
        return list(self._tokens)


class _FailingASRAdapter:
    async def transcribe_pcm(
        self, pcm: bytes, *, language: str | None = None, prompt: str | None = None
    ) -> list[TranscriptToken]:
        raise RuntimeError("ASR failure")


async def _multi_chunk_stream(path: str, chunk_seconds: float = 1.0):
    for _ in range(_NUM_CHUNKS):
        yield _CHUNK_BYTES


def _mock_stream():
    return patch(
        "coro.pipelines.streaming.stream_pcm_from_file",
        new=_multi_chunk_stream,
    )


# ---------------------------------------------------------------------------
# ASR-Only transcribe()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_asr_only_transcribe_succeeds():
    """transcribe() works with no diarization at all."""
    pipeline = StreamingPipeline(asr=_FakeASRAdapter())
    with _mock_stream():
        result = await pipeline.transcribe(AudioInput(b"audio"))
    assert RESPONSE_KEYS.issubset(result.keys())
    assert result["diarization"] == []


@pytest.mark.asyncio
async def test_asr_only_transcribe_empty_diarization_when_no_tokens():
    """Response diarization field is empty list when no tokens and no diarizer configured."""
    pipeline = StreamingPipeline(asr=_FakeASRAdapter(tokens=[]))
    with _mock_stream():
        result = await pipeline.transcribe(AudioInput(b"audio"))
    assert result["diarization"] == []


@pytest.mark.asyncio
async def test_asr_only_transcribe_cleanup_on_success():
    pipeline = StreamingPipeline(asr=_FakeASRAdapter())
    audio = AudioInput(b"audio")
    with _mock_stream():
        await pipeline.transcribe(audio)
    assert audio._temp_path is None


@pytest.mark.asyncio
async def test_asr_only_transcribe_cleanup_on_asr_failure():
    pipeline = StreamingPipeline(asr=_FailingASRAdapter())
    audio = AudioInput(b"audio")
    with _mock_stream(), pytest.raises(RuntimeError, match="ASR failure"):
        await pipeline.transcribe(audio)
    assert audio._temp_path is None


# ---------------------------------------------------------------------------
# ASR-Only stream()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_asr_only_stream_emits_delta_events():
    tokens = [TranscriptToken(start=0.0, end=0.5, text=" hello.", probability=1.0)]
    pipeline = StreamingPipeline(asr=_FakeASRAdapter(tokens=tokens))
    audio = AudioInput(b"audio")
    with _mock_stream():
        events = [e async for e in pipeline.stream(audio)]
    deltas = [e for e in events if isinstance(e, TranscriptDeltaEvent)]
    assert len(deltas) >= 1


@pytest.mark.asyncio
async def test_asr_only_stream_done_has_empty_diarization_when_no_tokens():
    """Done frame payload has empty diarization list when no tokens and no diarizer."""
    pipeline = StreamingPipeline(asr=_FakeASRAdapter(tokens=[]))
    audio = AudioInput(b"audio")
    with _mock_stream():
        events = [e async for e in pipeline.stream(audio)]
    done = [e for e in events if isinstance(e, StreamingDoneFrame)]
    assert len(done) == 1
    parsed = _render_done_frame(done[0])
    assert parsed["diarization"] == []


@pytest.mark.asyncio
async def test_asr_only_stream_cleanup_on_success():
    tokens = [TranscriptToken(start=0.0, end=0.5, text=" hello.", probability=1.0)]
    pipeline = StreamingPipeline(asr=_FakeASRAdapter(tokens=tokens))
    audio = AudioInput(b"audio")
    with _mock_stream():
        events = [e async for e in pipeline.stream(audio)]
    for event in events:
        if isinstance(event, StreamingDoneFrame):
            _render_done_frame(event)
    assert audio._temp_path is None


@pytest.mark.asyncio
async def test_asr_only_stream_cleanup_on_asr_failure():
    pipeline = StreamingPipeline(asr=_FailingASRAdapter())
    audio = AudioInput(b"audio")
    with _mock_stream(), pytest.raises(RuntimeError):
        _ = [e async for e in pipeline.stream(audio)]
    assert audio._temp_path is None


# ---------------------------------------------------------------------------
# No factory constructed in ASR-only config
# ---------------------------------------------------------------------------


def test_no_factory_when_diarization_none():
    """When no streaming_diarizer_factory provided, pipeline has None factory."""
    pipeline = StreamingPipeline(asr=_FakeASRAdapter())
    assert pipeline._streaming_diarizer_factory is None
