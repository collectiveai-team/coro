"""Adapter Concurrency Policy: admission control and non-serialised inference.

Covers the shared :mod:`coro.backends.asr.concurrency` primitives and each ASR
Adapter's declared policy:

- faster-whisper and onnx-asr run concurrently (no lock); two simultaneous
  transcriptions overlap in time.
- onnx-genai serialises (one permit) because its backend publishes no
  thread-safety guarantee for concurrent generators on a shared model.
- Work beyond the queue-depth cap is rejected with AsrCapacityError carrying a
  retry hint, rather than queued indefinitely.

No real ASR model is loaded; adapters are driven against stub backends whose
"inference" is a sleep, so overlap is observable as wall time.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time

import numpy as np
import pytest

from coro.backends.asr.concurrency import (
    DEFAULT_RETRY_AFTER_SECONDS,
    AdmissionController,
    AsrCapacityError,
    build_admission_controller,
    resolve_max_concurrency,
)
from coro.backends.asr.faster_whisper import FasterWhisperASRAdapter
from coro.backends.asr.onnx_asr import OnnxAsrASRAdapter
from coro.backends.asr.onnx_genai import OnnxGenaiASRAdapter

# One stub "inference" takes this long. Long enough that serialisation is
# unambiguous, short enough to keep the suite fast.
_WORK_SECONDS = 0.15
_PCM = np.zeros(1600, dtype=np.int16).tobytes()


# ---------------------------------------------------------------------------
# resolve_max_concurrency
# ---------------------------------------------------------------------------


def test_explicit_concurrency_is_used_verbatim():
    """A positive configured value is the permit count."""
    assert resolve_max_concurrency(7) == 7


def test_auto_concurrency_never_serialises():
    """Auto-sizing (0) always leaves room for at least two concurrent calls."""
    assert resolve_max_concurrency(0) >= 2


def test_auto_concurrency_tracks_core_count():
    """Auto-sizing scales with the host core count, bounding total thread demand."""
    cores = os.cpu_count() or 1
    assert resolve_max_concurrency(0) == max(2, cores // 4)


# ---------------------------------------------------------------------------
# AdmissionController
# ---------------------------------------------------------------------------


async def test_admission_allows_calls_up_to_the_permit_count():
    """Permits are held concurrently up to max_concurrency."""
    controller = AdmissionController(max_concurrency=3, max_queue_depth=0)
    peak = 0

    async def _hold():
        nonlocal peak
        async with controller.admit():
            peak = max(peak, controller.in_flight)
            await asyncio.sleep(_WORK_SECONDS)

    await asyncio.gather(*(_hold() for _ in range(3)))
    assert peak == 3


async def test_admission_bounds_in_flight_calls():
    """A fourth call waits rather than exceeding a three-permit controller."""
    controller = AdmissionController(max_concurrency=3, max_queue_depth=8)
    peak = 0

    async def _hold():
        nonlocal peak
        async with controller.admit():
            peak = max(peak, controller.in_flight)
            await asyncio.sleep(_WORK_SECONDS)

    await asyncio.gather(*(_hold() for _ in range(6)))
    assert peak == 3
    assert controller.in_flight == 0


async def test_admission_rejects_past_queue_depth_cap():
    """Beyond permits + queue depth, callers are rejected with a retry hint."""
    controller = AdmissionController(max_concurrency=1, max_queue_depth=1)
    started = asyncio.Event()

    async def _hold():
        async with controller.admit():
            started.set()
            await asyncio.sleep(_WORK_SECONDS)

    holder = asyncio.create_task(_hold())
    await started.wait()
    waiter = asyncio.create_task(_hold())
    await asyncio.sleep(0)  # let the waiter reach the semaphore

    with pytest.raises(AsrCapacityError) as excinfo:
        async with controller.admit():
            pass

    assert excinfo.value.retry_after_seconds == DEFAULT_RETRY_AFTER_SECONDS
    await asyncio.gather(holder, waiter)


async def test_admission_releases_permit_on_failure():
    """A failing call still releases its permit."""
    controller = AdmissionController(max_concurrency=1, max_queue_depth=0)

    with pytest.raises(RuntimeError, match="boom"):
        async with controller.admit():
            msg = "boom"
            raise RuntimeError(msg)

    assert controller.in_flight == 0
    async with controller.admit():
        assert controller.in_flight == 1


def test_serialized_policy_forces_one_permit():
    """A serialised adapter gets one permit regardless of configuration."""
    controller = build_admission_controller(max_concurrency=8, max_queue_depth=4, serialized=True)
    assert controller.max_concurrency == 1
    assert controller.max_queue_depth == 4


# ---------------------------------------------------------------------------
# Adapter stubs
# ---------------------------------------------------------------------------


class _SleepingOnnxAsrModel:
    """Stub onnx-asr model whose recognize() blocks for a fixed duration."""

    def __init__(self) -> None:
        self.concurrent = 0
        self.peak_concurrent = 0
        self._guard = threading.Lock()

    def recognize(self, audio, **_kwargs):
        with self._guard:
            self.concurrent += 1
            self.peak_concurrent = max(self.peak_concurrent, self.concurrent)
        time.sleep(_WORK_SECONDS)
        with self._guard:
            self.concurrent -= 1
        from types import SimpleNamespace

        return SimpleNamespace(tokens=[" hi"], timestamps=[0.0], logprobs=None)


class _SleepingWhisperModel:
    """Stub faster-whisper model whose transcribe() blocks for a fixed duration."""

    def __init__(self) -> None:
        self.concurrent = 0
        self.peak_concurrent = 0
        self._guard = threading.Lock()

    def transcribe(self, audio, **_kwargs):
        with self._guard:
            self.concurrent += 1
            self.peak_concurrent = max(self.peak_concurrent, self.concurrent)
        time.sleep(_WORK_SECONDS)
        with self._guard:
            self.concurrent -= 1
        return iter([]), None


async def _elapsed_for_two(adapter) -> float:
    """Run two transcriptions simultaneously and return the wall time taken."""
    started = time.perf_counter()
    await asyncio.gather(adapter.transcribe_pcm(_PCM), adapter.transcribe_pcm(_PCM))
    return time.perf_counter() - started


# ---------------------------------------------------------------------------
# Per-adapter concurrency policy
# ---------------------------------------------------------------------------


async def test_onnx_asr_adapter_runs_two_requests_concurrently():
    """Two simultaneous onnx-asr transcriptions overlap in time."""
    model = _SleepingOnnxAsrModel()
    adapter = OnnxAsrASRAdapter(
        model, admission=AdmissionController(max_concurrency=2, max_queue_depth=4)
    )

    elapsed = await _elapsed_for_two(adapter)

    assert model.peak_concurrent == 2
    assert elapsed < 2 * _WORK_SECONDS


async def test_faster_whisper_adapter_runs_two_requests_concurrently():
    """Two simultaneous faster-whisper transcriptions overlap in time."""
    model = _SleepingWhisperModel()
    adapter = FasterWhisperASRAdapter(
        model, admission=AdmissionController(max_concurrency=2, max_queue_depth=4)
    )

    elapsed = await _elapsed_for_two(adapter)

    assert model.peak_concurrent == 2
    assert elapsed < 2 * _WORK_SECONDS


@pytest.mark.parametrize(
    "adapter_factory",
    [
        lambda: OnnxAsrASRAdapter(_SleepingOnnxAsrModel()),
        lambda: FasterWhisperASRAdapter(_SleepingWhisperModel()),
    ],
    ids=["onnx-asr", "faster-whisper"],
)
def test_concurrent_adapters_hold_no_lock(adapter_factory):
    """The concurrent adapters carry no serialising lock attribute."""
    adapter = adapter_factory()
    assert not hasattr(adapter, "_lock")
    assert adapter.admission.max_concurrency >= 2


async def test_onnx_genai_adapter_serialises_by_policy():
    """onnx-genai keeps a one-permit policy: model calls never overlap."""
    adapter = OnnxGenaiASRAdapter(object(), chunk_samples=8960, sample_rate=16000)
    assert adapter.admission.max_concurrency == 1

    peak = 0
    concurrent = 0
    guard = threading.Lock()

    def _fake_stream(_audio, _lang_id):
        nonlocal peak, concurrent
        with guard:
            concurrent += 1
            peak = max(peak, concurrent)
        time.sleep(_WORK_SECONDS)
        with guard:
            concurrent -= 1
        return []

    adapter._stream = _fake_stream
    await _elapsed_for_two(adapter)

    assert peak == 1


async def test_adapter_rejects_beyond_queue_depth_with_retry_hint():
    """An adapter past its queue-depth cap raises AsrCapacityError, not a 500."""
    model = _SleepingOnnxAsrModel()
    adapter = OnnxAsrASRAdapter(
        model, admission=AdmissionController(max_concurrency=1, max_queue_depth=0)
    )

    busy = asyncio.create_task(adapter.transcribe_pcm(_PCM))
    while adapter.admission.in_flight == 0:  # wait for the permit to be taken
        await asyncio.sleep(0)

    with pytest.raises(AsrCapacityError) as excinfo:
        await adapter.transcribe_pcm(_PCM)

    assert excinfo.value.retry_after_seconds > 0
    assert "capacity" in excinfo.value.message
    await busy
