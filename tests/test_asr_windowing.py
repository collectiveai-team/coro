"""ASR Windowing deep module behavior."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import pytest

from coro.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
from coro.core.models import TranscriptDeltaEvent, TranscriptToken, TokenBatchEvent
from coro.pipelines.windowing import ASRWindowing

_BYTES_PER_SECOND = SAMPLE_RATE * BYTES_PER_SAMPLE


class _FakeASR:
    """Emit one token at the midpoint of whichever window it is handed.

    Mid-window placement keeps the token clear of every reconciliation
    boundary, so tests about prompt carry-over or event shape stay independent
    of overlap reconciliation.
    """

    def __init__(self) -> None:
        self.prompts: list[str | None] = []

    async def transcribe_pcm(self, pcm: bytes, *, language=None, prompt=None):
        self.prompts.append(prompt)
        call = len(self.prompts)
        midpoint = len(pcm) / _BYTES_PER_SECOND / 2
        return [
            TranscriptToken(
                start=midpoint,
                end=midpoint + 0.25,
                text=f" word{call}",
                probability=1.0,
            )
        ]


class _OverlapEchoASR:
    """Emit one token at the very first instant of every window.

    For every window after the first, that instant sits inside the region
    shared with the preceding window, which already transcribed it. This is
    precisely the duplication ASR Windowing must reconcile away.
    """

    def __init__(self) -> None:
        self.prompts: list[str | None] = []

    async def transcribe_pcm(self, pcm: bytes, *, language=None, prompt=None):
        self.prompts.append(prompt)
        call = len(self.prompts)
        return [TranscriptToken(start=0.0, end=0.25, text=f" word{call}", probability=1.0)]


def _pcm_seconds(seconds: float) -> bytes:
    return b"\x00\x00" * int(SAMPLE_RATE * seconds)


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


@pytest.mark.asyncio
async def test_overlapped_region_contributes_tokens_only_once():
    """Replaces the former test that asserted every window's tokens are kept.

    Each window after the first re-transcribes the shared region; only the
    window that owns that region may contribute it to the output.
    """
    asr = _OverlapEchoASR()
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.25)

    result = await windowing.transcribe_pcm(
        _pcm_seconds(2.0),
        asr=asr,
        language="es",
        prompt="hint",
    )

    # Every window is still transcribed ...
    assert len(asr.prompts) == 3
    # ... but the overlap-region duplicates are not accepted.
    assert [token.text for token in result.tokens] == [" word1"]


@pytest.mark.asyncio
async def test_prompt_carry_over_excludes_reconciled_away_tokens():
    asr = _OverlapEchoASR()
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.25)

    await windowing.transcribe_pcm(_pcm_seconds(2.0), asr=asr, language="es", prompt="hint")

    # Windows 2 and 3 contributed nothing, so the carry-over never grows past
    # the single accepted token.
    assert asr.prompts == ["hint", " word1", " word1"]


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


# ---------------------------------------------------------------------------
# Boundary reconciliation across all three transcription paths
# ---------------------------------------------------------------------------

_GRID_SECONDS = 3.0
_GRID_ORIGIN = 0.05
_GRID_STEP = 0.1


def _grid() -> Iterator[tuple[int, float]]:
    """Words laid on the absolute timeline, away from any window boundary."""
    index = 0
    while True:
        at = _GRID_ORIGIN + _GRID_STEP * index
        if at >= _GRID_SECONDS:
            return
        yield index, at
        index += 1


_EXPECTED_WORDS = [f" w{index}" for index, _ in _grid()]


class _ScriptedASR:
    """Transcribe a fixed grid of words placed on the absolute timeline.

    Window ``n`` always starts at ``n * step_seconds``, identically for every
    ASR Windowing path, so the fake can map its window-local view back onto the
    absolute timeline and return the words that genuinely fall inside the
    window — including the ones an overlapping neighbour also sees.
    """

    def __init__(self, step_seconds: float) -> None:
        self._step_seconds = step_seconds
        self.calls = 0
        self.raw_token_count = 0

    async def transcribe_pcm(self, pcm: bytes, *, language=None, prompt=None):
        offset = self.calls * self._step_seconds
        self.calls += 1
        duration = len(pcm) / _BYTES_PER_SECOND
        tokens = [
            TranscriptToken(
                start=at - offset,
                end=at - offset + 0.05,
                text=f" w{index}",
                probability=1.0,
            )
            for index, at in _grid()
            if offset <= at < offset + duration
        ]
        self.raw_token_count += len(tokens)
        return tokens


def _grid_chunks(pcm: bytes) -> list[bytes]:
    chunk_size = int(_BYTES_PER_SECOND * 0.4)
    return [pcm[i : i + chunk_size] for i in range(0, len(pcm), chunk_size)]


@pytest.mark.asyncio
async def test_no_repeated_token_run_at_window_boundaries_in_all_paths():
    """Audio spanning four windows must transcribe each word exactly once."""
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.25)
    step_seconds = windowing.step_bytes / _BYTES_PER_SECOND
    pcm = _pcm_seconds(_GRID_SECONDS)

    batch_asr = _ScriptedASR(step_seconds)
    batch = await windowing.transcribe_pcm(pcm, asr=batch_asr, language="es", prompt=None)

    stream_asr = _ScriptedASR(step_seconds)
    stream_tokens = [
        token
        async for event in windowing.stream_pcm(pcm, asr=stream_asr, language="es", prompt=None)
        if isinstance(event, TokenBatchEvent)
        for token in event.tokens
    ]

    chunks_asr = _ScriptedASR(step_seconds)
    chunk_tokens = [
        token
        async for event in windowing.stream_chunks(
            _async_chunks(_grid_chunks(pcm)), asr=chunks_asr, language="es", prompt=None
        )
        if isinstance(event, TokenBatchEvent)
        for token in event.tokens
    ]

    # The fixture really does exercise the defect: the ASR saw the overlap
    # regions twice and returned more raw tokens than there are words.
    assert batch_asr.calls >= 3
    assert batch_asr.raw_token_count > len(_EXPECTED_WORDS)

    assert [token.text for token in batch.tokens] == _EXPECTED_WORDS
    assert [token.text for token in stream_tokens] == _EXPECTED_WORDS
    assert [token.text for token in chunk_tokens] == _EXPECTED_WORDS


@pytest.mark.asyncio
async def test_streaming_deltas_contain_no_duplicated_text():
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.25)
    step_seconds = windowing.step_bytes / _BYTES_PER_SECOND
    pcm = _pcm_seconds(_GRID_SECONDS)

    stream_deltas = [
        event.delta
        async for event in windowing.stream_pcm(
            pcm, asr=_ScriptedASR(step_seconds), language="es", prompt=None
        )
        if isinstance(event, TranscriptDeltaEvent)
    ]
    chunk_deltas = [
        event.delta
        async for event in windowing.stream_chunks(
            _async_chunks(_grid_chunks(pcm)),
            asr=_ScriptedASR(step_seconds),
            language="es",
            prompt=None,
        )
        if isinstance(event, TranscriptDeltaEvent)
    ]

    expected = [word.strip() for word in _EXPECTED_WORDS]
    assert " ".join(stream_deltas).split() == expected
    assert " ".join(chunk_deltas).split() == expected


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

    deltas_pcm = [e.delta for e in events_pcm if isinstance(e, TranscriptDeltaEvent)]
    deltas_chunks = [e.delta for e in events_chunks if isinstance(e, TranscriptDeltaEvent)]
    assert deltas_chunks == deltas_pcm

    batches_pcm = [e.tokens for e in events_pcm if isinstance(e, TokenBatchEvent)]
    batches_chunks = [e.tokens for e in events_chunks if isinstance(e, TokenBatchEvent)]
    assert len(batches_chunks) == len(batches_pcm)
    for pcm_toks, chunk_toks in zip(batches_pcm, batches_chunks, strict=True):
        assert [(t.start, t.end, t.text) for t in chunk_toks] == [
            (t.start, t.end, t.text) for t in pcm_toks
        ]


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


@pytest.mark.asyncio
async def test_stream_chunks_prompt_carry_over_matches_stream_pcm():
    asr_chunks = _FakeASR()
    asr_pcm = _FakeASR()
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.25)
    pcm = _pcm_seconds(2.0)
    chunks = _grid_chunks(pcm)

    [e async for e in windowing.stream_pcm(pcm, asr=asr_pcm, language="es", prompt="initial")]
    [
        e
        async for e in windowing.stream_chunks(
            _async_chunks(chunks), asr=asr_chunks, language="es", prompt="initial"
        )
    ]

    assert asr_chunks.prompts == asr_pcm.prompts
