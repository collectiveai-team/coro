"""NemoDiarizationAdapter (batch Sortformer) — direct forward, postprocessing, gate.

The adapter calls the model's ``forward`` directly and never NeMo's
``diarize()`` wrapper, whose per-call state on the shared model made concurrent
requests race (issue #81, ADR 0021). The resolved Diarization Post-Processing
Configuration flows through the shared gated helper, and the Speaker-Count
Post-Processing Gate reverts to NeMo's baseline above the ceiling (ADR 0010).
No real NeMo model is loaded — the model handle is a fake exposing ``forward``.
"""

from __future__ import annotations

import asyncio
import struct
import threading
from types import SimpleNamespace

import pytest
import torch

from coro.backends.diarization.nemo import postprocessing
from coro.backends.diarization.nemo.diarization import (
    NemoDiarizationAdapter,
    prepare_for_inference,
)
from coro.core.models import SpeakerSegment

# 1 second at 16 kHz mono 16-bit; one known sample to check scaling.
_FAKE_PCM = struct.pack("<16000h", *([16384] + [0] * 15999))


def _preds_with_active_speakers(n_active: int, *, n_spk: int = 8, frames: int = 12):
    """Raw activity matrix where exactly ``n_active`` speakers are clearly present."""
    preds = torch.zeros(1, frames, n_spk)
    for spk in range(n_active):
        preds[0, :, spk] = 0.99
    return preds


class _FakeSortformerModel:
    """Stand-in exposing the ``forward`` call surface; ``diarize()`` is forbidden."""

    def __init__(self, preds=None, *, on_forward=None):
        self.device = torch.device("cpu")
        self._cfg = SimpleNamespace(encoder=SimpleNamespace(subsampling_factor=8))
        self.calls: list[dict] = []
        self._preds = preds if preds is not None else _preds_with_active_speakers(2, n_spk=4)
        self._on_forward = on_forward

    def forward(self, *, audio_signal, audio_signal_length):
        self.calls.append({"signal": audio_signal.clone(), "length": audio_signal_length.clone()})
        if self._on_forward is not None:
            self._on_forward()
        return self._preds.clone()

    def diarize(self, **_kwargs):
        raise AssertionError("diarize() races on shared model state; the adapter must not use it")


@pytest.fixture
def loaded_params(monkeypatch):
    """Record which post-processing YAML the gated helper loads, if any."""
    seen: list[str | None] = []

    def _record(postprocessing_yaml):
        seen.append(postprocessing_yaml)
        return postprocessing.baseline_postprocessing_params()

    monkeypatch.setattr(postprocessing, "load_postprocessing_params", _record)
    return seen


@pytest.mark.asyncio
async def test_forward_receives_normalised_in_memory_pcm():
    """PCM goes straight to forward as float32 in [-1, 1] — no temp WAV, no diarize()."""
    model = _FakeSortformerModel()
    adapter = NemoDiarizationAdapter(model)

    timeline = await adapter.diarize_pcm(_FAKE_PCM)

    (call,) = model.calls
    assert call["signal"].shape == (1, 16000)
    assert call["signal"].dtype == torch.float32
    assert call["signal"][0, 0].item() == pytest.approx(16384 / 32768, abs=1e-9)
    assert call["length"].tolist() == [16000]
    assert all(isinstance(s, SpeakerSegment) for s in timeline)
    assert {s.speaker for s in timeline} == {1, 2}


@pytest.mark.asyncio
async def test_unconfigured_postprocessing_uses_the_baseline(loaded_params):
    """No override reaches the helper as ``None``, i.e. NeMo's baseline."""
    adapter = NemoDiarizationAdapter(_FakeSortformerModel())

    await adapter.diarize_pcm(_FAKE_PCM)

    assert loaded_params == [None]


@pytest.mark.asyncio
async def test_resolved_postprocessing_yaml_is_applied(loaded_params):
    """A resolved Diarization Post-Processing Configuration path is the one loaded."""
    adapter = NemoDiarizationAdapter(
        _FakeSortformerModel(), postprocessing_yaml="/resolved/dihard3-dev.yaml"
    )

    await adapter.diarize_pcm(_FAKE_PCM)

    assert loaded_params == ["/resolved/dihard3-dev.yaml"]


def test_postprocessing_yaml_property_exposes_resolved_value():
    """The Backend Adapter Factory reads this to build a matching streaming factory."""
    adapter = NemoDiarizationAdapter(_FakeSortformerModel(), postprocessing_yaml="/some/path.yaml")
    # Round-tripping the constructor argument unchanged is the whole contract of
    # this accessor, so the "self-confirming literal" is the assertion's point.
    assert adapter.postprocessing_yaml == "/some/path.yaml"  # falsegreen: ignore


def test_postprocessing_yaml_property_defaults_to_none():
    adapter = NemoDiarizationAdapter(_FakeSortformerModel())
    assert adapter.postprocessing_yaml is None


# ---------------------------------------------------------------------------
# Speaker-Count Post-Processing Gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_open_applies_tuned_thresholds(loaded_params):
    """At or below the ceiling, the configured thresholds are used."""
    model = _FakeSortformerModel(_preds_with_active_speakers(4, n_spk=8))
    adapter = NemoDiarizationAdapter(model, postprocessing_yaml="/tuned.yaml", max_speakers=4)

    timeline = await adapter.diarize_pcm(_FAKE_PCM)

    assert loaded_params == ["/tuned.yaml"]
    assert len({s.speaker for s in timeline}) == 4


@pytest.mark.asyncio
async def test_gate_closed_bypasses_tuned_thresholds(loaded_params):
    """Above the ceiling the tuned set is never loaded; the baseline is used instead."""
    model = _FakeSortformerModel(_preds_with_active_speakers(5, n_spk=8))
    adapter = NemoDiarizationAdapter(model, postprocessing_yaml="/tuned.yaml", max_speakers=4)

    timeline = await adapter.diarize_pcm(_FAKE_PCM)

    assert loaded_params == []
    assert len({s.speaker for s in timeline}) == 5


@pytest.mark.asyncio
async def test_gate_needs_no_second_inference_pass():
    """The gate is evaluated on the same forward output."""
    model = _FakeSortformerModel(_preds_with_active_speakers(5, n_spk=8))
    adapter = NemoDiarizationAdapter(model, postprocessing_yaml=None, max_speakers=4)

    await adapter.diarize_pcm(_FAKE_PCM)

    assert len(model.calls) == 1


# ---------------------------------------------------------------------------
# Concurrency (issue #81, ADR 0021)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_requests_run_in_parallel_on_the_shared_model():
    """Two requests must be inside forward at the same time, and both succeed.

    The barrier only releases once both calls have reached forward, so any
    serialisation (a lock, a one-permit semaphore) breaks it and fails the test.
    """
    barrier = threading.Barrier(2, timeout=5)
    model = _FakeSortformerModel(on_forward=barrier.wait)
    adapter = NemoDiarizationAdapter(model)

    first, second = await asyncio.gather(
        adapter.diarize_pcm(_FAKE_PCM), adapter.diarize_pcm(_FAKE_PCM)
    )

    assert len(model.calls) == 2
    assert first == second
    assert {s.speaker for s in first} == {1, 2}


def test_prepare_for_inference_fixes_what_diarize_toggled_per_call():
    """Eval mode, dither and pad_to are set once at load instead of per call."""
    featurizer = SimpleNamespace(dither=1e-5, pad_to=16)
    model = SimpleNamespace(preprocessor=SimpleNamespace(featurizer=featurizer), evaluated=False)
    model.eval = lambda: setattr(model, "evaluated", True)

    prepare_for_inference(model)

    assert model.evaluated is True
    assert featurizer.dither == 0.0
    assert featurizer.pad_to == 0
