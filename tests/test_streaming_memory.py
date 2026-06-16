"""Flat-memory regression: streaming peak heap is independent of audio length.

Drives StreamingPipeline.stream() with a fake punctuating ASR over short and
long synthetic audio, consuming the done frame the way the SSE layer does
(rendered straight to bytes, never materialised).  The peak traced Python heap
must not grow proportionally with audio length: finalized segments and raw
words spill to the on-disk store, leaving only bounded working buffers
resident.
"""

from __future__ import annotations

import struct
import tracemalloc
from unittest.mock import patch

import pytest

from coro.audio import SAMPLE_RATE, AudioInput
from coro.core.types import TranscriptToken
from coro.pipelines.done_frame import StreamingDoneFrame
from coro.pipelines.streaming import StreamingPipeline
from coro.pipelines.windowing import ASRWindowing

_ONE_SECOND_PCM = struct.pack(f"<{SAMPLE_RATE}h", *([0] * SAMPLE_RATE))


class _PunctuatingASR:
    """Emits one punctuation-terminated token per window so segments finalize."""

    def __init__(self) -> None:
        self.n = 0

    async def transcribe_pcm(self, pcm, *, language=None, prompt=None):
        self.n += 1
        return [TranscriptToken(start=0.0, end=0.5, text=f" w{self.n}.", probability=1.0)]


def _mock_chunks(num_chunks: int):
    async def _gen(path, chunk_seconds: float = 1.0):
        for _ in range(num_chunks):
            yield _ONE_SECOND_PCM

    return patch("coro.pipelines.streaming.stream_pcm_from_file", new=_gen)


async def _drain(pipeline: StreamingPipeline) -> None:
    async for event in pipeline.stream(AudioInput(b"x")):
        if isinstance(event, StreamingDoneFrame):
            for _ in event.iter_sse():  # render to bytes, discard (flat)
                pass


async def _peak_heap_bytes(num_chunks: int, spill_dir: str) -> int:
    pipeline = StreamingPipeline(
        asr=_PunctuatingASR(),
        windowing=ASRWindowing(window_seconds=1.0, overlap_seconds=0.0),
        spill_dir=spill_dir,
    )
    with _mock_chunks(num_chunks):
        tracemalloc.start()
        await _drain(pipeline)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return peak


@pytest.mark.asyncio
async def test_streaming_peak_heap_is_flat_in_audio_length(tmp_path):
    short_dir = tmp_path / "a"
    long_dir = tmp_path / "b"
    short_dir.mkdir()
    long_dir.mkdir()
    short = await _peak_heap_bytes(50, str(short_dir))
    long = await _peak_heap_bytes(2000, str(long_dir))

    # 40x more audio (50 -> 2000 windows) must not grow the peak heap
    # proportionally. Linear accumulation of ~2000 tokens/segments would add
    # well over 1 MB; flat streaming keeps the delta to bounded working state.
    assert long - short < 256 * 1024, (
        f"peak heap grew {(long - short) / 1024:.1f} KiB for 40x audio "
        f"(short={short / 1024:.1f} KiB, long={long / 1024:.1f} KiB)"
    )
