"""ASR Windowing deep module behavior."""

from __future__ import annotations

from array import array
from collections.abc import AsyncIterator

import pytest

from coro.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
from coro.core.models import TranscriptDeltaEvent, TranscriptToken, TokenBatchEvent
from coro.pipelines.windowing import ASRWindowing, OverlapTokenAcceptance


class _FakeASR:
    """Emit one token per window, positioned at the window's midpoint.

    The midpoint keeps every token inside its own window's Overlap Token
    Acceptance region, so these tests exercise plumbing rather than acceptance.
    """

    def __init__(self) -> None:
        self.prompts: list[str | None] = []

    async def transcribe_pcm(self, pcm: bytes, *, language=None, prompt=None):
        self.prompts.append(prompt)
        call = len(self.prompts)
        middle = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE) / 2
        return [
            TranscriptToken(start=middle, end=middle + 0.05, text=f" word{call}", probability=1.0)
        ]


def _pcm_seconds(seconds: float) -> bytes:
    return b"\x00\x00" * int(SAMPLE_RATE * seconds)


# MARK: Positional Fixtures
SLOT_SECONDS = 0.1


def _positional_pcm(seconds: float) -> bytes:
    """Build PCM whose samples encode the index of the slot they belong to."""
    samples_per_slot = int(SAMPLE_RATE * SLOT_SECONDS)
    total_samples = int(SAMPLE_RATE * seconds)
    return array("h", (i // samples_per_slot for i in range(total_samples))).tobytes()


class _PositionalASR:
    """Decode ``_positional_pcm`` into one token per slot with window-local times.

    This models a real backend: every window independently transcribes the audio
    it sees, so overlapping windows re-emit the same spoken words, and a word
    only partly inside a window is reported from that window's own edge.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def transcribe_pcm(self, pcm: bytes, *, language=None, prompt=None):
        self.calls += 1
        samples = array("h")
        samples.frombytes(pcm)
        tokens: list[TranscriptToken] = []
        current: int | None = None
        for index, value in enumerate(samples):
            if value == current:
                continue
            current = value
            start = index / SAMPLE_RATE
            tokens.append(
                TranscriptToken(
                    start=start,
                    end=start + SLOT_SECONDS,
                    text=f" w{value}",
                    probability=1.0,
                )
            )
        return tokens


class _ScriptedASR:
    """Return a caller-supplied token list per window, in order."""

    def __init__(self, script: list[list[TranscriptToken]]) -> None:
        self._script = script
        self.calls = 0

    async def transcribe_pcm(self, pcm: bytes, *, language=None, prompt=None):
        tokens = self._script[self.calls] if self.calls < len(self._script) else []
        self.calls += 1
        return tokens


def _token(start: float, text: str) -> TranscriptToken:
    return TranscriptToken(start=start, end=start + 0.1, text=text, probability=1.0)


async def _transcript(windowing: ASRWindowing, pcm: bytes, asr) -> list[str]:
    result = await windowing.transcribe_pcm(pcm, asr=asr, language=None, prompt=None)
    return [token.text.strip() for token in result.tokens]


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
    # The tail window spans 1.0s-1.5s and _FakeASR reports its midpoint.
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
    chunk_size = int(SAMPLE_RATE * BYTES_PER_SAMPLE * 1.0)
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
    chunk_size = int(SAMPLE_RATE * BYTES_PER_SAMPLE * 0.4)
    chunks = [pcm[i : i + chunk_size] for i in range(0, len(pcm), chunk_size)]

    [e async for e in windowing.stream_pcm(pcm, asr=asr_pcm, language="es", prompt="initial")]
    [
        e
        async for e in windowing.stream_chunks(
            _async_chunks(chunks), asr=asr_chunks, language="es", prompt="initial"
        )
    ]

    assert asr_chunks.prompts == asr_pcm.prompts


# MARK: Overlap Token Acceptance
@pytest.mark.asyncio
async def test_audio_shorter_than_one_window_keeps_every_token():
    """A single window has no overlap, so acceptance must drop nothing."""
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.25)
    asr = _PositionalASR()

    words = await _transcript(windowing, _positional_pcm(0.6), asr)

    assert asr.calls == 1
    assert words == [f"w{slot}" for slot in range(6)]


@pytest.mark.asyncio
async def test_exactly_two_windows_emit_each_word_once():
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.25)
    asr = _PositionalASR()

    words = await _transcript(windowing, _positional_pcm(1.5), asr)

    assert asr.calls == 2
    assert words == [f"w{slot}" for slot in range(15)]


@pytest.mark.asyncio
async def test_many_windows_emit_each_word_once_and_in_order():
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.25)
    asr = _PositionalASR()

    words = await _transcript(windowing, _positional_pcm(10.0), asr)

    assert asr.calls > 5
    assert words == [f"w{slot}" for slot in range(100)]


@pytest.mark.asyncio
async def test_word_straddling_a_window_boundary_is_emitted_once_and_whole():
    """The window that heard the whole word wins; the truncated copy is dropped.

    ``hello`` starts at 0.95s and runs past the first window's 1.0s edge, so the
    first window can only report a fragment.  The second window (0.75s-1.75s)
    contains the whole word, and acceptance must prefer it.
    """
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.25)
    asr = _ScriptedASR(
        [
            [_token(0.10, " say"), _token(0.95, " hel")],
            [_token(0.20, " hello"), _token(0.60, " there")],
            [_token(0.05, " there"), _token(0.30, " bye")],
        ]
    )

    words = await _transcript(windowing, _positional_pcm(2.0), asr)

    assert words == ["say", "hello", "there", "bye"]


@pytest.mark.asyncio
async def test_window_yielding_no_tokens_emits_no_events_and_still_advances():
    """Silence in one window must not stall acceptance for the windows after it."""
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.25)
    asr = _ScriptedASR([[_token(0.10, " one")], [], [_token(0.30, " two")]])

    events = [
        e
        async for e in windowing.stream_pcm(
            _positional_pcm(2.0), asr=asr, language=None, prompt=None
        )
    ]

    batches = [e for e in events if isinstance(e, TokenBatchEvent)]
    assert asr.calls == 3
    assert len(batches) == 2
    assert [t.text.strip() for e in batches for t in e.tokens] == ["one", "two"]


@pytest.mark.asyncio
async def test_both_pipelines_produce_identical_transcripts():
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.25)
    pcm = _positional_pcm(4.0)
    chunk_size = int(SAMPLE_RATE * BYTES_PER_SAMPLE * 0.4)
    chunks = [pcm[i : i + chunk_size] for i in range(0, len(pcm), chunk_size)]

    asr_pcm = _PositionalASR()
    tokens_pcm = [
        t
        async for e in windowing.stream_pcm(pcm, asr=asr_pcm, language=None, prompt=None)
        if isinstance(e, TokenBatchEvent)
        for t in e.tokens
    ]

    asr_chunks = _PositionalASR()
    tokens_chunks = [
        t
        async for e in windowing.stream_chunks(
            _async_chunks(chunks), asr=asr_chunks, language=None, prompt=None
        )
        if isinstance(e, TokenBatchEvent)
        for t in e.tokens
    ]

    assert [t.text.strip() for t in tokens_pcm] == [f"w{slot}" for slot in range(40)]
    assert [(t.start, t.end, t.text) for t in tokens_chunks] == [
        (t.start, t.end, t.text) for t in tokens_pcm
    ]
    assert asr_chunks.calls == asr_pcm.calls


def test_overlap_token_acceptance_state_is_scalar():
    """Acceptance may remember a boundary, never a token history."""
    acceptance = OverlapTokenAcceptance()

    for window in range(200):
        acceptance.accept(
            [_token(0.1 * index, f" w{index}") for index in range(10)],
            offset_seconds=float(window),
            cut_seconds=float(window) + 1.0,
        )

    assert all(isinstance(value, float) for value in vars(acceptance).values())
