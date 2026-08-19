"""Flat-memory regression: streaming peak heap is independent of audio length.

Drives the Streaming Pipeline with a fake punctuating ASR over short and long
synthetic audio and measures the peak traced Python heap.  Finalized segments
and raw words spill to the on-disk store, so the peak must not grow
proportionally with audio length — only bounded working buffers stay resident.

Both public response paths are covered, because the guarantee is a property of
the pipeline, not of one of its consumers:

- ``stream()``, whose done frame the SSE layer renders straight to bytes.
- ``transcribe()``, whose JSON body the vendor routes and ``coro run`` render.

The ``transcribe()`` case deliberately reaches across the API boundary and
renders a real ``response_format`` body, because that is where the response is
actually materialised: the pipeline's own ``TranscriptionResult`` and the
route's revalidation of it are two separate O(audio length) copies, and a test
that stopped at the pipeline would miss the second.
"""

from __future__ import annotations

import gc
import logging
import struct
import tracemalloc
from collections.abc import Awaitable, Callable
from unittest.mock import patch

import pytest

from coro.api.openai.transcriptions import ResponseFormat, response_for_format
from coro.audio import SAMPLE_RATE, AudioInput
from coro.cache.adapter import CachingASRAdapter
from coro.cache.store import ASRCacheStore
from coro.core.models import TranscriptToken
from coro.pipelines.done_frame import StreamingDoneFrame
from coro.pipelines.streaming import StreamingPipeline
from coro.pipelines.windowing import ASRWindowing


def _one_second_pcm(index: int) -> bytes:
    """Return one second of PCM whose bytes depend on ``index``.

    Distinct per chunk so that with the ASR window cache enabled every window
    gets its own key and its own row, which is the configuration whose memory
    behaviour is actually in question — identical chunks would collapse into a
    single entry and prove nothing.
    """
    return struct.pack(f"<{SAMPLE_RATE}h", *([index % 4096] * SAMPLE_RATE))


# Window-relative span of the fake ASR's token. It must sit strictly inside the
# window's authoritative span or ASR Windowing discards it: `_seconds_to_bytes`
# floors an overlap of 0.0 s at one sample, so `accept_from` lands half a sample
# *after* the window start and a token at window-relative 0.0 is reconciled away
# for every window but the first — leaving a one-segment transcript no matter
# how long the audio is, and a memory test that measures nothing.
_TOKEN_START = 0.5
_TOKEN_END = 0.9


class _PunctuatingASR:
    """Emits one punctuation-terminated token per window so segments finalize."""

    honours_prompt = False

    def __init__(self) -> None:
        self.n = 0

    async def transcribe_pcm(self, pcm, *, language=None, prompt=None):
        self.n += 1
        return [
            TranscriptToken(
                start=_TOKEN_START, end=_TOKEN_END, text=f" w{self.n}.", probability=1.0
            )
        ]


def _mock_chunks(num_chunks: int):
    async def _gen(path, chunk_seconds: float = 1.0):
        for index in range(num_chunks):
            yield _one_second_pcm(index)

    return patch("coro.pipelines.streaming.stream_pcm_from_file", new=_gen)


async def _drain_sse(pipeline: StreamingPipeline) -> None:
    """Consume stream() the way the SSE layer does: rendered to bytes, discarded."""
    async for event in pipeline.stream(AudioInput(b"x")):
        if isinstance(event, StreamingDoneFrame):
            for _ in event.iter_sse():  # render to bytes, discard (flat)
                pass


async def _drain_json(pipeline: StreamingPipeline) -> None:
    """Consume transcribe() the way a vendor route does: rendered to a JSON body.

    ``verbose_json`` is the heaviest of the three formats — it is the only one
    carrying both segments and words — so it is the format the guarantee has to
    hold for.
    """
    result = await pipeline.transcribe(AudioInput(b"x"))
    body = response_for_format(ResponseFormat.VERBOSE_JSON, result, language="en")
    body.model_dump_json()


Drain = Callable[[StreamingPipeline], Awaitable[None]]


async def _peak_heap_bytes(
    num_chunks: int,
    spill_dir: str,
    *,
    drain: Drain,
    cache: bool = False,
) -> int:
    asr = _PunctuatingASR()
    if cache:
        store = ASRCacheStore(f"{spill_dir}/asr-cache", max_bytes=0, ttl_seconds=0.0)
        asr = CachingASRAdapter(asr, store=store, fingerprint="fp", honours_prompt=False)
    pipeline = StreamingPipeline(
        asr=asr,
        windowing=ASRWindowing(window_seconds=1.0, overlap_seconds=0.0),
        spill_dir=spill_dir,
    )
    with _mock_chunks(num_chunks):
        # Per-window logging allocates transient strings that scale with audio
        # length and pollute the trace; silence it so the measurement reflects
        # only the pipeline's resident data structures (the flat-memory property).
        logging.disable(logging.CRITICAL)
        gc.collect()
        tracemalloc.start()
        try:
            await drain(pipeline)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
            logging.disable(logging.NOTSET)
    return peak


# 40x more audio must not grow the peak heap proportionally. Linear
# accumulation of ~2000 tokens/segments would add well over 1 MB; flat
# rendering keeps the delta to bounded working state.
_SHORT_CHUNKS = 50
_LONG_CHUNKS = 2000
_MAX_GROWTH_BYTES = 256 * 1024


async def _peak_heap_growth(tmp_path, *, drain: Drain, cache: bool) -> tuple[int, int]:
    """Return the short- and long-audio peak heaps for one drain strategy."""
    short_dir = tmp_path / "short"
    long_dir = tmp_path / "long"
    short_dir.mkdir()
    long_dir.mkdir()
    short = await _peak_heap_bytes(_SHORT_CHUNKS, str(short_dir), drain=drain, cache=cache)
    long = await _peak_heap_bytes(_LONG_CHUNKS, str(long_dir), drain=drain, cache=cache)
    return short, long


def _growth_message(short: int, long: int, *, path: str) -> str:
    return (
        f"{path}: peak heap grew {(long - short) / 1024:.1f} KiB for "
        f"{_LONG_CHUNKS // _SHORT_CHUNKS}x audio "
        f"(short={short / 1024:.1f} KiB, long={long / 1024:.1f} KiB)"
    )


@pytest.mark.asyncio
async def test_transcript_grows_with_audio_length(tmp_path):
    """Guard the flat-heap assertions against a transcript that never accumulates.

    Without this, a fixture whose tokens are all reconciled away yields a
    one-segment transcript for any audio length, and both memory tests below
    pass against a fully materialising implementation.
    """
    pipeline = StreamingPipeline(
        asr=_PunctuatingASR(),
        windowing=ASRWindowing(window_seconds=1.0, overlap_seconds=0.0),
        spill_dir=str(tmp_path),
    )
    with _mock_chunks(_SHORT_CHUNKS):
        result = await pipeline.transcribe(AudioInput(b"x"))

    # One accepted, punctuation-terminated token per window becomes one segment,
    # one word and one raw word. `stream_chunks` needs a chunk of lookahead to
    # know a window is not the final one, so N chunks dispatch N-1 full windows
    # and then flush the remaining buffer as the final window: N+1 in total.
    expected = _SHORT_CHUNKS + 1
    assert len(result.segments) == expected
    assert len(result.word_segments) == expected
    assert len(result.raw_words) == expected


@pytest.mark.parametrize("cache", [False, True], ids=["cache-disabled", "cache-enabled"])
@pytest.mark.asyncio
async def test_streaming_peak_heap_is_flat_in_audio_length(tmp_path, cache: bool):
    """The bounded-memory guarantee must hold for the configuration actually deployed.

    Running this with the ASR window cache enabled as well as disabled is what
    stops the cache from quietly reintroducing O(audio length) resident state —
    it commits a row per window, so an implementation that buffered rows instead
    of committing them would show up here and nowhere else.
    """
    short, long = await _peak_heap_growth(tmp_path, drain=_drain_sse, cache=cache)
    assert long - short < _MAX_GROWTH_BYTES, _growth_message(short, long, path="stream()")


@pytest.mark.parametrize("cache", [False, True], ids=["cache-disabled", "cache-enabled"])
@pytest.mark.asyncio
async def test_json_response_peak_heap_is_flat_in_audio_length(tmp_path, cache: bool):
    """The same guarantee must hold on the non-SSE JSON path.

    ``stream=false`` is the default on every vendor endpoint, so this is the
    path most requests take. If it materialises the transcript the Streaming
    Pipeline's defining property applies to a minority of traffic.
    """
    short, long = await _peak_heap_growth(tmp_path, drain=_drain_json, cache=cache)
    assert long - short < _MAX_GROWTH_BYTES, _growth_message(short, long, path="transcribe()")
