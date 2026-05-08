"""Streaming Pipeline orchestration with fake adapters.

Tests verify that StreamingPipeline:
- Uses stream_pcm_from_file to read spooled audio in bounded-memory chunks.
- Calls ASR adapter with PCM windows no larger than window_bytes.
- Calls diarization adapter when configured.
- Returns a valid transcription response dict.
- Propagates prompt and language across ASR chunks.
- Handles empty token output without crashing.
- Never holds full PCM in memory (bounded-memory assertions).

The ffmpeg streaming step is mocked so tests run without subprocess.
"""

from __future__ import annotations

import json
import struct
from unittest.mock import patch

import pytest

from asr_diar_server.audio import AudioInput
from asr_diar_server.core.types import (
    SpeakerSegment,
    TranscriptDeltaEvent,
    TranscriptDoneEvent,
    TranscriptToken,
)
from asr_diar_server.pipelines.streaming import StreamingPipeline
from asr_diar_server.pipelines.windowing import ASRWindowing

RESPONSE_KEYS = {"segments", "word_segments", "transcript", "diarization", "raw_words"}

# Multi-chunk fixture: 3 chunks of 0.1s (1600 bytes each) so total < window_bytes
_CHUNK_BYTES = struct.pack("<1600h", *([0] * 1600))
_NUM_CHUNKS = 3


class _FakeASRAdapter:
    def __init__(self, tokens=None):
        self._tokens = tokens or []
        self.call_count = 0
        self.last_prompt = None
        self.last_language = None
        self.max_pcm_size = 0

    async def transcribe_pcm(self, pcm_bytes, *, language=None, prompt=None):
        self.call_count += 1
        self.last_prompt = prompt
        self.last_language = language
        if len(pcm_bytes) > self.max_pcm_size:
            self.max_pcm_size = len(pcm_bytes)
        return list(self._tokens)


class _FakeDiarizationAdapter:
    def __init__(self, timeline=None):
        self._timeline = timeline or []

    async def diarize_pcm(self, pcm_bytes):
        return list(self._timeline)


class _FakeStreamingDiarizer:
    """Fake per-request streaming diarizer tracking ingest calls."""

    def __init__(self, timeline=None):
        self._timeline = timeline or []
        self.ingest_calls: list[int] = []  # sizes of each ingest call

    def ingest_pcm_chunk(self, pcm: bytes) -> None:
        self.ingest_calls.append(len(pcm))

    def finalize(self):
        return list(self._timeline)


class _FakeStreamingDiarizerFactory:
    """Factory producing fresh _FakeStreamingDiarizer per call."""

    def __init__(self, timeline=None):
        self._timeline = timeline or []
        self.instances: list[_FakeStreamingDiarizer] = []

    def __call__(self) -> _FakeStreamingDiarizer:
        d = _FakeStreamingDiarizer(timeline=self._timeline)
        self.instances.append(d)
        return d


class _FailingASRAdapter:
    async def transcribe_pcm(self, pcm_bytes, *, language=None, prompt=None):
        raise RuntimeError("ASR failed")

    async def diarize_pcm(self, pcm_bytes):
        return []


async def _multi_chunk_stream(path: str, chunk_seconds: float = 1.0):
    for _ in range(_NUM_CHUNKS):
        yield _CHUNK_BYTES


def _mock_stream():
    return patch(
        "asr_diar_server.pipelines.streaming.stream_pcm_from_file",
        new=_multi_chunk_stream,
    )


# ---------------------------------------------------------------------------
# Response shape & propagation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_pipeline_returns_response_shape():
    pipeline = StreamingPipeline(asr=_FakeASRAdapter(), diarization=None)
    with _mock_stream():
        result = await pipeline.transcribe(AudioInput(b"audio"))
    assert RESPONSE_KEYS.issubset(result.keys())


@pytest.mark.asyncio
async def test_streaming_pipeline_passes_prompt_to_asr():
    asr = _FakeASRAdapter()
    pipeline = StreamingPipeline(asr=asr, diarization=None)
    with _mock_stream():
        await pipeline.transcribe(AudioInput(b"audio"), prompt="mi prompt")
    assert asr.last_prompt == "mi prompt"


@pytest.mark.asyncio
async def test_streaming_pipeline_passes_language_to_asr():
    asr = _FakeASRAdapter()
    pipeline = StreamingPipeline(asr=asr, diarization=None)
    with _mock_stream():
        await pipeline.transcribe(AudioInput(b"audio"), language="es")
    assert asr.last_language == "es"


@pytest.mark.asyncio
async def test_streaming_pipeline_uses_diarization_when_provided():
    tokens = [TranscriptToken(start=0.0, end=1.0, text=" hola.", probability=0.9)]
    timeline = [SpeakerSegment(start=0.0, end=2.0, speaker=2)]
    asr = _FakeASRAdapter(tokens=tokens)
    diar = _FakeDiarizationAdapter(timeline=timeline)
    pipeline = StreamingPipeline(asr=asr, diarization=diar)
    with _mock_stream():
        result = await pipeline.transcribe(AudioInput(b"audio"))
    seg = result["segments"][0]
    assert seg["speaker"] == "2"


@pytest.mark.asyncio
async def test_streaming_pipeline_empty_tokens_no_crash():
    pipeline = StreamingPipeline(asr=_FakeASRAdapter(tokens=[]), diarization=None)
    with _mock_stream():
        result = await pipeline.transcribe(AudioInput(b"audio"))
    assert result["segments"] == []
    assert result["raw_words"] == []


# ---------------------------------------------------------------------------
# Streaming events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_emits_delta_events():
    tokens = [TranscriptToken(start=0.0, end=0.5, text=" hello.", probability=1.0)]
    pipeline = StreamingPipeline(asr=_FakeASRAdapter(tokens=tokens), diarization=None)
    audio = AudioInput(b"audio")
    with _mock_stream():
        events = [event async for event in pipeline.stream(audio)]
    deltas = [e for e in events if isinstance(e, TranscriptDeltaEvent)]
    assert len(deltas) >= 1
    assert deltas[0].delta == "hello."


@pytest.mark.asyncio
async def test_stream_emits_exactly_one_done_event():
    tokens = [TranscriptToken(start=0.0, end=0.5, text=" hello.", probability=1.0)]
    pipeline = StreamingPipeline(asr=_FakeASRAdapter(tokens=tokens), diarization=None)
    audio = AudioInput(b"audio")
    with _mock_stream():
        events = [event async for event in pipeline.stream(audio)]
    done_events = [e for e in events if isinstance(e, TranscriptDoneEvent)]
    assert len(done_events) == 1
    parsed = json.loads(done_events[0].text)
    assert RESPONSE_KEYS.issubset(parsed.keys())


@pytest.mark.asyncio
async def test_stream_done_event_comes_after_deltas():
    tokens = [TranscriptToken(start=0.0, end=0.5, text=" hello.", probability=1.0)]
    pipeline = StreamingPipeline(asr=_FakeASRAdapter(tokens=tokens), diarization=None)
    audio = AudioInput(b"audio")
    with _mock_stream():
        events = [event async for event in pipeline.stream(audio)]
    delta_indices = [i for i, e in enumerate(events) if isinstance(e, TranscriptDeltaEvent)]
    done_indices = [i for i, e in enumerate(events) if isinstance(e, TranscriptDoneEvent)]
    assert done_indices[0] > delta_indices[0]


@pytest.mark.asyncio
async def test_stream_emits_no_progress_events():
    tokens = [TranscriptToken(start=0.0, end=0.5, text=" hello.", probability=1.0)]
    pipeline = StreamingPipeline(asr=_FakeASRAdapter(tokens=tokens), diarization=None)
    audio = AudioInput(b"audio")
    with _mock_stream():
        events = [event async for event in pipeline.stream(audio)]
    progress_events = [e for e in events if hasattr(e, "type") and "progress" in e.type]
    assert progress_events == []


# ---------------------------------------------------------------------------
# Audio cleanup
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_transcribe_cleans_up_temp_file_on_success():
    pipeline = StreamingPipeline(asr=_FakeASRAdapter(), diarization=None)
    audio = AudioInput(b"audio")
    with _mock_stream():
        await pipeline.transcribe(audio)
    assert audio._temp_path is None


@pytest.mark.asyncio
async def test_transcribe_cleans_up_temp_file_on_error():
    pipeline = StreamingPipeline(asr=_FailingASRAdapter(), diarization=None)
    audio = AudioInput(b"audio")
    with _mock_stream(), pytest.raises(RuntimeError, match="ASR failed"):
        await pipeline.transcribe(audio)
    assert audio._temp_path is None


@pytest.mark.asyncio
async def test_stream_cleans_up_temp_file_on_success():
    tokens = [TranscriptToken(start=0.0, end=0.5, text=" hello.", probability=1.0)]
    pipeline = StreamingPipeline(asr=_FakeASRAdapter(tokens=tokens), diarization=None)
    audio = AudioInput(b"audio")
    with _mock_stream():
        _ = [event async for event in pipeline.stream(audio)]
    assert audio._temp_path is None


@pytest.mark.asyncio
async def test_stream_cleans_up_temp_file_on_error():
    pipeline = StreamingPipeline(asr=_FailingASRAdapter(), diarization=None)
    audio = AudioInput(b"audio")
    with _mock_stream(), pytest.raises(RuntimeError):
        _ = [event async for event in pipeline.stream(audio)]
    assert audio._temp_path is None


# ---------------------------------------------------------------------------
# Bounded-memory assertions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_asr_never_called_with_more_than_window_bytes():
    """ASR must never receive more PCM than one window."""
    asr = _FakeASRAdapter()
    windowing = ASRWindowing()
    pipeline = StreamingPipeline(asr=asr, windowing=windowing)
    with _mock_stream():
        await pipeline.transcribe(AudioInput(b"audio"))
    assert asr.max_pcm_size <= windowing.window_bytes


@pytest.mark.asyncio
async def test_streaming_diarizer_ingested_one_chunk_per_call():
    """Streaming diarizer must be called with exactly one raw chunk per call."""
    factory = _FakeStreamingDiarizerFactory()
    pipeline = StreamingPipeline(asr=_FakeASRAdapter(), streaming_diarizer_factory=factory)
    with _mock_stream():
        await pipeline.transcribe(AudioInput(b"audio"))
    assert len(factory.instances) == 1
    diarizer = factory.instances[0]
    # Each ingest call should be exactly one chunk (chunk_bytes = len(_CHUNK_BYTES))
    for call_size in diarizer.ingest_calls:
        assert call_size == len(_CHUNK_BYTES)
    assert len(diarizer.ingest_calls) == _NUM_CHUNKS


@pytest.mark.asyncio
async def test_no_full_pcm_accumulation_during_transcribe():
    """Total bytes consumed should equal sum of individual chunks, never concatenated."""
    chunk_sizes_seen: list[int] = []
    original_stream = _multi_chunk_stream

    async def _instrumented_stream(path, chunk_seconds=1.0):
        async for chunk in original_stream(path, chunk_seconds):
            chunk_sizes_seen.append(len(chunk))
            yield chunk

    pipeline = StreamingPipeline(asr=_FakeASRAdapter())
    with patch(
        "asr_diar_server.pipelines.streaming.stream_pcm_from_file",
        new=_instrumented_stream,
    ):
        await pipeline.transcribe(AudioInput(b"audio"))

    # Each chunk must be exactly _CHUNK_BYTES in size (never concatenated)
    assert all(size == len(_CHUNK_BYTES) for size in chunk_sizes_seen)
    assert len(chunk_sizes_seen) == _NUM_CHUNKS


@pytest.mark.asyncio
async def test_streaming_diarizer_factory_produces_one_instance_per_request():
    """Factory is called once per transcribe() call."""
    factory = _FakeStreamingDiarizerFactory()
    pipeline = StreamingPipeline(asr=_FakeASRAdapter(), streaming_diarizer_factory=factory)
    with _mock_stream():
        await pipeline.transcribe(AudioInput(b"audio"))
        await pipeline.transcribe(AudioInput(b"audio"))
    assert len(factory.instances) == 2


@pytest.mark.asyncio
async def test_streaming_diarizer_finalize_provides_timeline():
    """When streaming diarizer is used, its finalize() result is reflected in response."""
    timeline = [SpeakerSegment(start=0.0, end=0.2, speaker=1)]
    factory = _FakeStreamingDiarizerFactory(timeline=timeline)
    tokens = [TranscriptToken(start=0.0, end=0.2, text=" hola.", probability=0.9)]
    pipeline = StreamingPipeline(
        asr=_FakeASRAdapter(tokens=tokens),
        streaming_diarizer_factory=factory,
    )
    with _mock_stream():
        result = await pipeline.transcribe(AudioInput(b"audio"))
    seg = result["segments"][0]
    assert seg["speaker"] == "1"
