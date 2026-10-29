"""ASR Windowing deep module behavior."""

from __future__ import annotations

import pytest

from coro.audio import BYTES_PER_SAMPLE
from coro.core.models import TokenBatchEvent, TranscriptDeltaEvent, TranscriptToken
from coro.pipelines.windowing import ASRWindowing
from support.windowing import (
    BYTES_PER_SECOND,
    FakeASR,
    OverlapEchoASR,
    TimelineGridASR,
    WindowRecordingASR,
    async_chunks,
    grid_chunks,
    pcm_seconds,
    word_grid,
)

WINDOW_SECONDS = 1.0
OVERLAP_SECONDS = 0.25
STEP_SECONDS = WINDOW_SECONDS - OVERLAP_SECONDS
GRID_SECONDS = 3.0
GRID = word_grid(GRID_SECONDS)
EXPECTED_WORDS = [f" w{index}" for index in range(len(GRID))]


def overlapping_windowing() -> ASRWindowing:
    """Return the 1 s window, 0.25 s overlap ASR Windowing every test here shares."""
    return ASRWindowing(window_seconds=WINDOW_SECONDS, overlap_seconds=OVERLAP_SECONDS)


async def collect(events):
    """Drain an event stream into a list."""
    return [event async for event in events]


@pytest.mark.asyncio
async def test_overlapped_region_contributes_tokens_only_once():
    asr = OverlapEchoASR()

    result = await overlapping_windowing().transcribe_pcm(
        pcm_seconds(2.0), asr=asr, language="es", prompt="hint"
    )

    assert len(asr.prompts) == 3
    assert [token.text for token in result.tokens] == [" word1"]


@pytest.mark.asyncio
async def test_prompt_carry_over_excludes_reconciled_away_tokens():
    asr = OverlapEchoASR()

    await overlapping_windowing().transcribe_pcm(
        pcm_seconds(2.0), asr=asr, language="es", prompt="hint"
    )

    assert asr.prompts == ["hint", " word1", " word1"]


@pytest.mark.asyncio
async def test_asr_windowing_streams_delta_per_accepted_window():
    events = await collect(
        overlapping_windowing().stream_pcm(
            pcm_seconds(1.2), asr=FakeASR(), language=None, prompt=None
        )
    )

    assert [e.delta for e in events if isinstance(e, TranscriptDeltaEvent)] == ["word1", "word2"]


@pytest.mark.asyncio
async def test_asr_windowing_streams_typed_token_batch_events():
    events = await collect(
        overlapping_windowing().stream_pcm(
            pcm_seconds(1.2), asr=FakeASR(), language=None, prompt=None
        )
    )

    batches = [e for e in events if isinstance(e, TokenBatchEvent)]
    assert [[t.text for t in batch.tokens] for batch in batches] == [[" word1"], [" word2"]]
    assert all(isinstance(t, TranscriptToken) for batch in batches for t in batch.tokens)


@pytest.mark.asyncio
async def test_every_window_handed_to_the_asr_is_whole_samples():
    """An odd byte count per window is requested: the ASR still sees whole 16-bit samples."""
    asr = WindowRecordingASR()
    windowing = ASRWindowing(window_seconds=321 / BYTES_PER_SECOND, overlap_seconds=0.0)

    await windowing.transcribe_pcm(pcm_seconds(0.1), asr=asr, language=None, prompt=None)

    assert {len(window) % BYTES_PER_SAMPLE for window in asr.windows} == {0}


@pytest.mark.asyncio
async def test_the_grid_fixture_hears_overlap_regions_twice():
    asr = TimelineGridASR(STEP_SECONDS, GRID)

    await overlapping_windowing().transcribe_pcm(
        pcm_seconds(GRID_SECONDS), asr=asr, language="es", prompt=None
    )

    assert asr.heard_tokens > len(EXPECTED_WORDS)


@pytest.mark.asyncio
async def test_transcribe_pcm_keeps_each_word_once_across_window_boundaries():
    result = await overlapping_windowing().transcribe_pcm(
        pcm_seconds(GRID_SECONDS),
        asr=TimelineGridASR(STEP_SECONDS, GRID),
        language="es",
        prompt=None,
    )

    assert [token.text for token in result.tokens] == EXPECTED_WORDS


async def grid_stream_pcm_events() -> list:
    """Stream the word grid through ``stream_pcm`` and return every event."""
    return await collect(
        overlapping_windowing().stream_pcm(
            pcm_seconds(GRID_SECONDS),
            asr=TimelineGridASR(STEP_SECONDS, GRID),
            language="es",
            prompt=None,
        )
    )


async def grid_stream_chunks_events() -> list:
    """Stream the word grid through ``stream_chunks`` in 0.4 s chunks and return every event."""
    return await collect(
        overlapping_windowing().stream_chunks(
            async_chunks(grid_chunks(pcm_seconds(GRID_SECONDS))),
            asr=TimelineGridASR(STEP_SECONDS, GRID),
            language="es",
            prompt=None,
        )
    )


def batch_tokens(events: list) -> list[str]:
    """Return the text of every token of the token batch events."""
    return [t.text for e in events if isinstance(e, TokenBatchEvent) for t in e.tokens]


def delta_words(events: list) -> list[str]:
    """Return the words of the concatenated transcript delta events."""
    return " ".join(e.delta for e in events if isinstance(e, TranscriptDeltaEvent)).split()


@pytest.mark.asyncio
async def test_stream_pcm_keeps_each_word_once_across_window_boundaries():
    assert batch_tokens(await grid_stream_pcm_events()) == EXPECTED_WORDS


@pytest.mark.asyncio
async def test_stream_pcm_deltas_repeat_no_word():
    assert delta_words(await grid_stream_pcm_events()) == [w.strip() for w in EXPECTED_WORDS]


@pytest.mark.asyncio
async def test_stream_chunks_keeps_each_word_once_across_window_boundaries():
    assert batch_tokens(await grid_stream_chunks_events()) == EXPECTED_WORDS


@pytest.mark.asyncio
async def test_stream_chunks_deltas_repeat_no_word():
    assert delta_words(await grid_stream_chunks_events()) == [w.strip() for w in EXPECTED_WORDS]
