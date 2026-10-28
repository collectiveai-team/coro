"""ASR Windowing deep module behavior."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from coro.audio import BYTES_PER_SAMPLE
from coro.core.models import TranscriptDeltaEvent, TranscriptToken, TokenBatchEvent
from coro.pipelines.windowing import ASRWindowing
from support.windowing import (
    BYTES_PER_SECOND as _BYTES_PER_SECOND,
    FakeASR as _FakeASR,
    async_chunks as _async_chunks,
    grid_chunks as _grid_chunks,
    pcm_seconds as _pcm_seconds,
)


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
