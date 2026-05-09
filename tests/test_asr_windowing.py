"""ASR Windowing deep module behavior."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from asr_diar_server.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
from asr_diar_server.core.types import TranscriptDeltaEvent, TranscriptToken, TokenBatchEvent
from asr_diar_server.pipelines.windowing import ASRWindowing


class _FakeASR:
    def __init__(self) -> None:
        self.prompts: list[str | None] = []

    async def transcribe_pcm(self, pcm: bytes, *, language=None, prompt=None):
        self.prompts.append(prompt)
        call = len(self.prompts)
        return [TranscriptToken(start=0.0, end=0.25, text=f" word{call}", probability=1.0)]


def _pcm_seconds(seconds: float) -> bytes:
    return b"\x00\x00" * int(SAMPLE_RATE * seconds)


@pytest.mark.asyncio
async def test_asr_windowing_calls_adapter_for_overlapping_windows():
    asr = _FakeASR()
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.25)

    result = await windowing.transcribe_pcm(
        _pcm_seconds(2.0),
        asr=asr,
        language="es",
        prompt="hint",
    )

    assert len(asr.prompts) == 3
    assert [token.text for token in result.tokens] == [" word1", " word2", " word3"]


@pytest.mark.asyncio
async def test_asr_windowing_streams_delta_per_accepted_window():
    asr = _FakeASR()
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.25)

    events = [
        event
        async for event in windowing.stream_pcm(
            _pcm_seconds(1.2),
            asr=asr,
            language=None,
            prompt=None,
        )
    ]

    delta_events = [e for e in events if isinstance(e, TranscriptDeltaEvent)]
    assert [e.delta for e in delta_events] == ["word1", "word2"]


@pytest.mark.asyncio
async def test_asr_windowing_streams_typed_token_batch_events():
    asr = _FakeASR()
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.25)

    events = [
        event
        async for event in windowing.stream_pcm(
            _pcm_seconds(1.2),
            asr=asr,
            language=None,
            prompt=None,
        )
    ]

    token_events = [e for e in events if isinstance(e, TokenBatchEvent)]
    assert len(token_events) == 2
    assert all(isinstance(e.tokens, list) for e in token_events)
    assert all(isinstance(t, TranscriptToken) for e in token_events for t in e.tokens)


def test_window_bytes_are_even_for_pcm_alignment():
    windowing = ASRWindowing(window_seconds=0.01, overlap_seconds=0.0)

    assert windowing.window_bytes % BYTES_PER_SAMPLE == 0


async def _async_chunks(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_stream_chunks_event_equivalence_with_stream_pcm():
    asr_chunks = _FakeASR()
    asr_pcm = _FakeASR()
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.25)
    pcm = _pcm_seconds(2.0)
    chunk_size = int(SAMPLE_RATE * BYTES_PER_SAMPLE * 0.4)
    chunks = [pcm[i : i + chunk_size] for i in range(0, len(pcm), chunk_size)]

    events_pcm = [
        e
        async for e in windowing.stream_pcm(
            pcm, asr=asr_pcm, language="es", prompt="hint"
        )
    ]
    events_chunks = [
        e
        async for e in windowing.stream_chunks(
            _async_chunks(chunks), asr=asr_chunks, language="es", prompt="hint"
        )
    ]

    deltas_pcm = [e.delta for e in events_pcm if isinstance(e, TranscriptDeltaEvent)]
    deltas_chunks = [
        e.delta for e in events_chunks if isinstance(e, TranscriptDeltaEvent)
    ]
    assert deltas_chunks == deltas_pcm

    batches_pcm = [e.tokens for e in events_pcm if isinstance(e, TokenBatchEvent)]
    batches_chunks = [
        e.tokens for e in events_chunks if isinstance(e, TokenBatchEvent)
    ]
    assert len(batches_chunks) == len(batches_pcm)
    for pcm_toks, chunk_toks in zip(batches_pcm, batches_chunks, strict=True):
        assert [(t.start, t.end, t.text) for t in chunk_toks] == [
            (t.start, t.end, t.text) for t in pcm_toks
        ]


@pytest.mark.asyncio
async def test_stream_chunks_buffer_never_exceeds_window_plus_max_chunk():
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.25)
    pcm = _pcm_seconds(3.0)
    chunk_size = int(SAMPLE_RATE * BYTES_PER_SAMPLE * 0.4)
    chunks = [pcm[i : i + chunk_size] for i in range(0, len(pcm), chunk_size)]
    max_chunk = max(len(c) for c in chunks)

    asr = _FakeASR()
    events = [
        e
        async for e in windowing.stream_chunks(
            _async_chunks(chunks), asr=asr, language=None, prompt=None
        )
    ]

    assert events is not None
    assert windowing._stream_chunks_buffer_highwater <= windowing.window_bytes + max_chunk


@pytest.mark.asyncio
async def test_stream_chunks_processes_partial_tail():
    asr = _FakeASR()
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.0)
    pcm = _pcm_seconds(1.5)
    half = len(pcm) // 2
    chunks = [pcm[:half], pcm[half:]]

    events = [
        e
        async for e in windowing.stream_chunks(
            _async_chunks(chunks), asr=asr, language=None, prompt=None
        )
    ]

    batches = [e for e in events if isinstance(e, TokenBatchEvent)]
    assert len(batches) == 2
    tail_start = batches[1].tokens[0].start
    assert tail_start == pytest.approx(1.0, abs=0.01)


@pytest.mark.asyncio
async def test_stream_chunks_prompt_carry_over_matches_stream_pcm():
    asr_chunks = _FakeASR()
    asr_pcm = _FakeASR()
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.25)
    pcm = _pcm_seconds(2.0)
    chunk_size = int(SAMPLE_RATE * BYTES_PER_SAMPLE * 0.4)
    chunks = [pcm[i : i + chunk_size] for i in range(0, len(pcm), chunk_size)]

    [
        e
        async for e in windowing.stream_pcm(
            pcm, asr=asr_pcm, language="es", prompt="initial"
        )
    ]
    [
        e
        async for e in windowing.stream_chunks(
            _async_chunks(chunks), asr=asr_chunks, language="es", prompt="initial"
        )
    ]

    assert asr_chunks.prompts == asr_pcm.prompts
