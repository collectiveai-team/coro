"""A response must not depend on what happened to be cached.

This is the acceptance property for the whole feature: a cold run, a fully warm
run, and a run over a cache with deliberate holes in it must all produce the
same bytes. If they ever disagree, a cached transcript can no longer be trusted
as much as a fresh one, and the feature is worse than useless.

It runs against a fake ASR Adapter that is a pure function of its input — which
is the determinism property the real backends were measured to have, and which
an opt-in real-model test guards separately.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict
from unittest.mock import patch

import pytest

from coro.audio import SAMPLE_RATE, AudioInput
from coro.cache.adapter import CachingASRAdapter
from coro.cache.store import ASRCacheStore
from coro.core.models import TranscriptToken
from coro.pipelines.full_memory import FullMemoryPipeline
from coro.pipelines.windowing import ASRWindowing

_AUDIO_SECONDS = 3.5
_SAMPLES = int(SAMPLE_RATE * _AUDIO_SECONDS)
# Varying rather than silent, so every window has distinct bytes and therefore a
# distinct key: identical windows would collapse into one entry and make a
# "fully cached" run indistinguishable from a one-window one.
_PCM = struct.pack(f"<{_SAMPLES}h", *[(index % 997) - 498 for index in range(_SAMPLES)])

_WINDOW_COUNT = 4
# Segments break on punctuation, not on window boundaries: each window emits one
# sentence-terminated token plus one unterminated one, and the tail merges.
_SEGMENT_COUNT = 5


class _PureASR:
    """An ASR Adapter that is a pure function of its window, like the real ones."""

    honours_prompt = False

    def __init__(self) -> None:
        self.calls = 0

    async def transcribe_pcm(self, pcm, *, language=None, prompt=None):
        self.calls += 1
        digest = hashlib.blake2b(pcm, digest_size=4).hexdigest()
        return [
            TranscriptToken(start=0.1, end=0.9, text=f" {digest}.", probability=0.75),
            TranscriptToken(start=0.9, end=1.2, text=f" {language or 'auto'}", probability=0.5),
        ]


class _LossyStore:
    """Store wrapper that silently drops selected writes, leaving cache holes.

    Modelling gaps this way keeps the test on the store's public surface instead
    of reaching into rows, and it is what actually happens in the field: a run
    that dies part-way leaves exactly this shape behind.
    """

    def __init__(self, inner: ASRCacheStore, *, drop_indices: set[int]) -> None:
        self._inner = inner
        self._drop_indices = drop_indices
        self._writes = 0

    def get(self, key: str):
        return self._inner.get(key)

    def put(self, key: str, tokens) -> None:
        index = self._writes
        self._writes += 1
        if index not in self._drop_indices:
            self._inner.put(key, tokens)


async def _identity_pcm(_path: str) -> bytes:
    """Stand in for ffmpeg decoding the spooled upload: the fixture is already PCM."""
    return _PCM


async def _transcribe(store, *, language: str | None = "es") -> tuple[str, int]:
    """Run the pipeline through the cache, returning the response JSON and model calls."""
    inner = _PureASR()
    adapter = CachingASRAdapter(inner, store=store, fingerprint="fp", honours_prompt=False)
    pipeline = FullMemoryPipeline(
        asr=adapter, windowing=ASRWindowing(window_seconds=1.0, overlap_seconds=0.0)
    )
    with patch("coro.pipelines.full_memory.convert_path_to_pcm_bytes", new=_identity_pcm):
        result = await pipeline.transcribe(AudioInput(_PCM), language=language)
    return json.dumps(asdict(result)), inner.calls


@pytest.fixture
def store(tmp_path):
    with ASRCacheStore(str(tmp_path / "cache"), max_bytes=0, ttl_seconds=0.0) as opened:
        yield opened


@pytest.mark.asyncio
async def test_a_cold_run_reaches_the_model_for_every_window(store):
    _, calls = await _transcribe(store)
    assert calls == _WINDOW_COUNT


@pytest.mark.asyncio
async def test_a_fully_cached_run_is_byte_identical_and_reaches_no_model(store):
    cold, _ = await _transcribe(store)
    warm, calls = await _transcribe(store)

    assert warm == cold
    assert calls == 0


@pytest.mark.asyncio
async def test_a_gapped_cache_is_byte_identical_and_recomputes_only_the_gaps(store):
    lossy = _LossyStore(store, drop_indices={1, 2})
    cold, _ = await _transcribe(lossy)

    repaired, calls = await _transcribe(store)

    assert repaired == cold
    assert calls == 2


@pytest.mark.asyncio
async def test_a_run_that_died_part_way_resumes_rather_than_restarting(store):
    """Per-window commits mean a killed run keeps everything it already did."""
    lossy = _LossyStore(store, drop_indices={2, 3})
    await _transcribe(lossy)

    _, calls = await _transcribe(store)

    assert calls == 2


@pytest.mark.asyncio
async def test_a_changed_language_is_not_served_from_the_cache(store):
    spanish, _ = await _transcribe(store, language="es")
    english, calls = await _transcribe(store, language="en")

    assert english != spanish
    assert calls == _WINDOW_COUNT


@pytest.mark.asyncio
async def test_the_parity_fixture_actually_exercises_a_response(store):
    """Guard the parity assertions against three equally empty responses."""
    payload = json.loads((await _transcribe(store))[0])
    # Non-emptiness is the whole specification of this guard: it exists so the
    # byte-parity assertions above cannot pass on three equally empty responses.
    assert len(payload["segments"]) == _SEGMENT_COUNT
    assert len(payload["word_segments"]) == _WINDOW_COUNT * 2
    assert len(payload["raw_words"]) == _WINDOW_COUNT * 2
