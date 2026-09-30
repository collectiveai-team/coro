"""Concurrent streaming requests must not race on the shared Sortformer model.

NeMo reads the latency-tier parameters off ``model.sortformer_modules`` inside
``forward_streaming_step``. Writing them onto the shared model around each call
and restoring afterwards interleaves under concurrency: request A restores the
checkpoint values while request B is still mid-call (B runs a chunk with the
wrong parameters), then B "restores" the tier values it saved from A — leaving
the shared model permanently retuned. Both are reproduced here
deterministically by forcing that interleaving with thread events.
"""

from __future__ import annotations

import os
import threading

import pytest
import torch

from coro.backends.diarization.nemo.streaming import (
    NemoStreamingDiarizerFactory,
    get_latency_tier_params,
)

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
ATTRS = ("chunk_len", "chunk_right_context", "fifo_len", "spkcache_update_period", "spkcache_len")
# What diar_streaming_sortformer_4spk-v2 ships with, in ATTRS order.
CHECKPOINT_CONFIG = (
    ("chunk_len", 188),
    ("chunk_right_context", 1),
    ("fifo_len", 0),
    ("spkcache_update_period", 188),
    ("spkcache_len", 188),
)
TIER = "low"


def _snapshot(obj) -> tuple[tuple[str, int], ...]:
    """The five streaming parameters of ``obj`` as comparable (name, value) pairs."""
    return tuple((name, getattr(obj, name)) for name in ATTRS)


class _FakeSortformerModules:
    """Plain object so attribute reads/writes behave like NeMo's, not a mock's."""

    def __init__(self):
        for name, value in CHECKPOINT_CONFIG:
            setattr(self, name, value)
        self.subsampling_factor = 8
        self.n_spk = 4

    def _check_streaming_parameters(self):
        return None

    def init_streaming_state(self, *, batch_size, async_streaming, device):
        return object()


class _FakeSortformerModel:
    """Records the parameters NeMo would read during each streaming step."""

    def __init__(self, hook):
        self.device = torch.device("cpu")
        self.sortformer_modules = _FakeSortformerModules()
        self.seen: list[tuple[str, tuple[tuple[str, int], ...]]] = []
        self._hook = hook

    def forward_streaming_step(self, signal, length, state, total_preds, **_kwargs):
        self._hook()
        # Read through ``self``, exactly as NeMo does.
        self.seen.append((threading.current_thread().name, _snapshot(self.sortformer_modules)))
        return state, torch.cat([total_preds, torch.full((1, 6, 4), 0.01)], dim=1)


def _preprocessor(*, input_signal, length):
    frames = max(1, input_signal.shape[-1] // 160)
    return torch.zeros(1, 128, frames), torch.tensor([frames])


def _one_chunk_of_pcm() -> bytes:
    params = get_latency_tier_params(TIER)
    frames = params.chunk_len + params.chunk_right_context
    return b"\x00\x00" * int(frames * 8 * 0.01 * SAMPLE_RATE)


def test_overlapping_streaming_requests_neither_misread_nor_retune_the_model():
    """A returns while B is mid-call: B must still see its tier; the model stays as loaded."""
    b_inside = threading.Event()
    a_returned = threading.Event()

    def _hook():
        if threading.current_thread().name == "A":
            assert b_inside.wait(timeout=5), "request B never reached the model"
        else:
            b_inside.set()
            assert a_returned.wait(timeout=5), "request A never returned"

    model = _FakeSortformerModel(_hook)
    factory = NemoStreamingDiarizerFactory(model, tier=TIER)
    pcm = _one_chunk_of_pcm()

    def _request():
        diarizer = factory()
        diarizer._preprocessor = _preprocessor
        diarizer.ingest_pcm_chunk(pcm)
        if threading.current_thread().name == "A":
            a_returned.set()

    threads = [threading.Thread(target=_request, name=name) for name in ("A", "B")]
    for thread in threads:
        thread.start()
    # The event waits above bound every thread, so a plain join cannot hang; a
    # missed rendezvous leaves that request out of ``seen`` and fails below.
    for thread in threads:
        thread.join()

    tier = _snapshot(get_latency_tier_params(TIER))
    assert dict(model.seen) == {"A": tier, "B": tier}
    assert _snapshot(model.sortformer_modules) == CHECKPOINT_CONFIG


@pytest.mark.skipif(
    os.environ.get("CORO_RUN_REAL_MODEL_TESTS", "0") != "1",
    reason="Set CORO_RUN_REAL_MODEL_TESTS=1 to run with real model checkpoints.",
)
def test_real_model_concurrent_streams_match_sequential_and_leave_the_model_alone():
    """The same scenario on the real Sortformer, with the interleaving left to chance."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from coro.audio import convert_path_to_pcm_bytes
    from coro.backends.diarization.nemo.diarization import build_nemo_diarization_adapter
    from coro.bench.data import WARMUP_AUDIO_PATH

    model = build_nemo_diarization_adapter(
        "nvidia/diar_streaming_sortformer_4spk-v2", device="cpu"
    ).model
    before = _snapshot(model.sortformer_modules)
    factory = NemoStreamingDiarizerFactory(model, tier=TIER)
    pcm = asyncio.run(convert_path_to_pcm_bytes(str(WARMUP_AUDIO_PATH))) * 2
    piece = SAMPLE_RATE * BYTES_PER_SAMPLE // 2  # 0.5 s, like an upload being streamed

    def _stream(_=None):
        diarizer = factory()
        for start in range(0, len(pcm), piece):
            diarizer.ingest_pcm_chunk(pcm[start : start + piece])
        return diarizer.finalize()

    sequential = _stream()
    with ThreadPoolExecutor(max_workers=2) as pool:
        concurrent = list(pool.map(_stream, range(2)))

    assert {s.speaker for s in sequential} == {1}
    assert concurrent == [sequential, sequential]
    assert _snapshot(model.sortformer_modules) == before
