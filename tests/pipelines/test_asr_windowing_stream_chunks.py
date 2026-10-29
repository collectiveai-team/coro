"""ASR Windowing over a chunk stream: equivalence with whole-PCM windowing and bounded buffering."""

from __future__ import annotations

import pytest

from coro.core.models import TranscriptDeltaEvent, TokenBatchEvent
from coro.pipelines.windowing import ASRWindowing
from support.windowing import (
    BYTES_PER_SECOND as _BYTES_PER_SECOND,
    FakeASR as _FakeASR,
    async_chunks as _async_chunks,
    grid_chunks as _grid_chunks,
    pcm_seconds as _pcm_seconds,
)


class _WindowRecordingASR:
    """Record the exact window bytes handed to the adapter."""

    def __init__(self) -> None:
        self.windows: list[bytes] = []

    async def transcribe_pcm(self, pcm: bytes, *, language=None, prompt=None):
        self.windows.append(pcm)
        return []


@pytest.mark.asyncio
async def test_asr_windowing_calls_adapter_for_every_planned_window():
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


def _deltas(events: list[object]) -> list[str]:
    return [e.delta for e in events if isinstance(e, TranscriptDeltaEvent)]


def _token_batches(events: list[object]) -> list[list[tuple[float, float, str]]]:
    return [
        [(t.start, t.end, t.text) for t in e.tokens]
        for e in events
        if isinstance(e, TokenBatchEvent)
    ]


@pytest.mark.asyncio
async def test_stream_chunks_event_equivalence_with_stream_pcm():
    asr_chunks = _FakeASR()
    asr_pcm = _FakeASR()
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.25)
    pcm = _pcm_seconds(2.0)
    chunks = _grid_chunks(pcm)

    events_pcm = [
        e async for e in windowing.stream_pcm(pcm, asr=asr_pcm, language="es", prompt="hint")
    ]
    events_chunks = [
        e
        async for e in windowing.stream_chunks(
            _async_chunks(chunks), asr=asr_chunks, language="es", prompt="hint"
        )
    ]

    assert _deltas(events_chunks) == _deltas(events_pcm)
    assert _token_batches(events_chunks) == _token_batches(events_pcm)


@pytest.mark.parametrize(
    ("seconds", "label"),
    [
        (0.5, "shorter than one window"),
        (1.0, "exactly one window"),
        (1.75, "one step past a full window"),
        (2.5, "two steps past a full window"),
        (2.0, "ragged tail"),
        (3.0, "ragged tail, several windows"),
    ],
)
@pytest.mark.asyncio
async def test_stream_chunks_dispatches_the_same_windows_as_stream_pcm(seconds, label):
    """Both paths must hand the adapter byte-identical windows.

    When the stream length is exactly ``window + k * step`` the buffered path
    used to dispatch one extra window containing only the leftover overlap: a
    wasted inference over a region the previous window already covered, with
    the tail transcribed as an isolated overlap-length fragment rather than in
    the context of its full window. Reconciliation discarded the duplicate
    tokens, so the response still looked correct and only the window sequence
    reveals it — which is why this compares windows rather than output.
    """
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.25)
    pcm = _pcm_seconds(seconds)

    from_pcm = _WindowRecordingASR()
    async for _ in windowing.stream_pcm(pcm, asr=from_pcm, language="es", prompt=None):
        pass

    from_chunks = _WindowRecordingASR()
    async for _ in windowing.stream_chunks(
        _async_chunks(_grid_chunks(pcm)), asr=from_chunks, language="es", prompt=None
    ):
        pass

    assert [len(w) for w in from_chunks.windows] == [len(w) for w in from_pcm.windows], label
    assert from_chunks.windows == from_pcm.windows, label


@pytest.mark.asyncio
async def test_stream_chunks_buffer_never_exceeds_window_plus_max_chunk():
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.25)
    pcm = _pcm_seconds(3.0)
    chunks = _grid_chunks(pcm)
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
    # The 0.5 s tail window starts at 1.0 s; its token sits at the midpoint.
    tail_start = batches[1].tokens[0].start
    assert tail_start == pytest.approx(1.25, abs=0.01)


@pytest.mark.asyncio
async def test_stream_chunks_prompt_carry_is_bounded_over_long_stream():
    """Prompt carry must reflect only recent tokens, never the whole transcript.

    With one token emitted per window, after many windows the earliest token
    text must have aged out of the carry, proving retention is bounded and does
    not grow O(audio length).
    """
    asr = _FakeASR()
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.0)
    pcm = _pcm_seconds(80.0)  # ~80 windows, well past the 50-token carry bound
    chunk_size = int(_BYTES_PER_SECOND * 1.0)
    chunks = [pcm[i : i + chunk_size] for i in range(0, len(pcm), chunk_size)]

    _ = [
        e
        async for e in windowing.stream_chunks(
            _async_chunks(chunks), asr=asr, language=None, prompt=None
        )
    ]

    last_prompt = asr.prompts[-1] or ""
    assert len(last_prompt) <= 200
    # word1 (the first emitted token) must have aged out of the bounded carry.
    assert "word1 " not in last_prompt
    # but a recent token must still be present.
    assert f"word{len(asr.prompts) - 1}" in last_prompt
