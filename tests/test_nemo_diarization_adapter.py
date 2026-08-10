"""NemoDiarizationAdapter (batch Sortformer) — postprocessing threading and gate.

Verifies the batch adapter passes its resolved Diarization Post-Processing
Configuration value straight through to the real NeMo ``diarize(...,
postprocessing_yaml=...)`` kwarg, and that the Speaker-Count Post-Processing
Gate reverts to NeMo's baseline when the estimated speaker count exceeds the
ceiling. See ADR 0010. No real NeMo model is loaded — the model handle is a
fake exposing only ``.diarize``.
"""

from __future__ import annotations

import struct

import pytest
import torch

from coro.backends.diarization.nemo.diarization import NemoDiarizationAdapter
from coro.core.models import SpeakerSegment

# 1 second of silence at 16 kHz mono 16-bit.
_FAKE_PCM = struct.pack("<16000h", *([0] * 16000))


def _preds_with_active_speakers(n_active: int, *, n_spk: int = 8, frames: int = 200):
    """Raw activity matrix where exactly ``n_active`` speakers are clearly present."""
    preds = torch.zeros(1, frames, n_spk)
    for spk in range(n_active):
        preds[0, :, spk] = 0.99
    return preds


class _FakeSortformerModel:
    """Minimal stand-in exposing only the ``.diarize`` call surface used here."""

    def __init__(self, preds=None):
        self.calls: list[dict] = []
        self._preds = preds if preds is not None else _preds_with_active_speakers(2, n_spk=4)

    def diarize(self, *, audio, batch_size, postprocessing_yaml, include_tensor_outputs=False):
        self.calls.append(
            {
                "audio": audio,
                "batch_size": batch_size,
                "postprocessing_yaml": postprocessing_yaml,
                "include_tensor_outputs": include_tensor_outputs,
            }
        )
        lines = [["0.00 1.00 speaker_0"]]
        if include_tensor_outputs:
            return lines, [self._preds]
        return lines


@pytest.mark.asyncio
async def test_diarize_pcm_passes_none_by_default():
    """The default (no override) reaches diarize() as postprocessing_yaml=None."""
    model = _FakeSortformerModel()
    adapter = NemoDiarizationAdapter(model)

    timeline = await adapter.diarize_pcm(_FAKE_PCM)

    assert model.calls[0]["postprocessing_yaml"] is None
    assert all(isinstance(s, SpeakerSegment) for s in timeline)


@pytest.mark.asyncio
async def test_diarize_pcm_passes_resolved_postprocessing_yaml():
    """A resolved Diarization Post-Processing Configuration path reaches diarize()."""
    model = _FakeSortformerModel()
    adapter = NemoDiarizationAdapter(model, postprocessing_yaml="/resolved/dihard3-dev.yaml")

    await adapter.diarize_pcm(_FAKE_PCM)

    assert model.calls[0]["postprocessing_yaml"] == "/resolved/dihard3-dev.yaml"


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
async def test_gate_open_keeps_nemo_postprocessed_output():
    """At or below the ceiling, NeMo's own tuned output stands unmodified."""
    model = _FakeSortformerModel(_preds_with_active_speakers(4, n_spk=8))
    adapter = NemoDiarizationAdapter(
        model,
        postprocessing_yaml="/resolved/dihard3-dev.yaml",
        max_speakers=4,
    )

    timeline = await adapter.diarize_pcm(_FAKE_PCM)

    # NeMo's single line survives verbatim: no coro-side recomputation.
    assert [(s.start, s.end) for s in timeline] == [(0.0, 1.0)]


@pytest.mark.asyncio
async def test_gate_closed_bypasses_tuned_thresholds():
    """Above the ceiling the tuned set is dropped and segments are re-derived.

    Five clearly-present speakers exceed the default ceiling, so the adapter
    must stop trusting NeMo's tuned output and recompute from the raw
    predictions with the plain baseline instead.
    """
    model = _FakeSortformerModel(_preds_with_active_speakers(5, n_spk=8))
    adapter = NemoDiarizationAdapter(
        model,
        postprocessing_yaml="/resolved/dihard3-dev.yaml",
        max_speakers=4,
    )

    timeline = await adapter.diarize_pcm(_FAKE_PCM)

    # Recomputed from the raw matrix: one long segment per active speaker,
    # not NeMo's single canned line.
    assert [(s.start, s.end) for s in timeline] != [(0.0, 1.0)]
    assert len({s.speaker for s in timeline}) == 5


@pytest.mark.asyncio
async def test_gate_is_inert_without_a_configured_preset():
    """With no tuned set configured there is nothing to gate off."""
    model = _FakeSortformerModel(_preds_with_active_speakers(6, n_spk=8))
    adapter = NemoDiarizationAdapter(model, max_speakers=4)

    timeline = await adapter.diarize_pcm(_FAKE_PCM)

    assert [(s.start, s.end) for s in timeline] == [(0.0, 1.0)]


@pytest.mark.asyncio
async def test_batch_requests_raw_tensors_for_the_gate():
    """The gate must not cost a second inference pass."""
    model = _FakeSortformerModel()
    adapter = NemoDiarizationAdapter(model, postprocessing_yaml="/resolved/x.yaml")

    await adapter.diarize_pcm(_FAKE_PCM)

    assert len(model.calls) == 1
    assert model.calls[0]["include_tensor_outputs"] is True
