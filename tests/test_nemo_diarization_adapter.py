"""NemoDiarizationAdapter (batch Sortformer) — postprocessing threading.

Verifies the batch adapter passes its resolved Diarization Post-Processing
Configuration value straight through to the real NeMo ``diarize(...,
postprocessing_yaml=...)`` kwarg. See ADR 0010. No real NeMo model is
loaded — the model handle is a fake exposing only ``.diarize``.
"""

from __future__ import annotations

import struct

import pytest

from coro.backends.diarization.nemo.diarization import NemoDiarizationAdapter
from coro.core.models import SpeakerSegment

# 1 second of silence at 16 kHz mono 16-bit.
_FAKE_PCM = struct.pack("<16000h", *([0] * 16000))


class _FakeSortformerModel:
    """Minimal stand-in exposing only the ``.diarize`` call surface used here."""

    def __init__(self):
        self.calls: list[dict] = []

    def diarize(self, *, audio, batch_size, postprocessing_yaml):
        self.calls.append(
            {
                "audio": audio,
                "batch_size": batch_size,
                "postprocessing_yaml": postprocessing_yaml,
            }
        )
        return [["0.00 1.00 speaker_0"]]


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
