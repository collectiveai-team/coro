"""Diarization Backend Adapter Factory dispatch and streaming capability.

Verifies the factory dispatches to the right provider builder, reports
streaming capability correctly, builds the streaming diarizer only for
streaming-capable providers, and rejects unknown providers. No real NeMo
model is loaded — the provider builder is patched.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from coro.backends.diarization import factory


def test_supports_streaming_nemo_true():
    """NeMo is a streaming-capable Diarization Backend Provider."""
    assert factory.supports_streaming("nemo") is True


def test_supports_streaming_pyannote_false():
    """Unknown/batch-only providers are not streaming-capable."""
    assert factory.supports_streaming("pyannote") is False
    assert factory.supports_streaming("bogus") is False


def test_build_diarization_adapter_dispatches_to_nemo():
    """The nemo provider routes to the NeMo adapter builder with device passthrough."""
    sentinel = object()
    with patch(
        "coro.backends.diarization.nemo.diarization.build_nemo_diarization_adapter",
        return_value=sentinel,
    ) as mock_build:
        adapter = factory.build_diarization_adapter("nemo", "some/model", device="cpu")

    assert adapter is sentinel
    mock_build.assert_called_once_with("some/model", device="cpu")


def test_build_diarization_adapter_unknown_provider_raises():
    """An unknown Diarization Backend Provider fails fast."""
    with pytest.raises(ValueError, match="Unknown diarization backend provider"):
        factory.build_diarization_adapter("bogus", "model")


def test_build_streaming_diarizer_factory_nemo():
    """A NeMo adapter yields a streaming factory bound to its shared model."""
    from coro.backends.diarization.nemo.diarization import NemoDiarizationAdapter

    fake_model = object()
    adapter = NemoDiarizationAdapter(fake_model)

    with patch(
        "coro.backends.diarization.nemo.streaming.NemoStreamingDiarizerFactory"
    ) as mock_factory:
        factory.build_streaming_diarizer_factory("nemo", adapter, tier="low")

    mock_factory.assert_called_once_with(fake_model, tier="low")


def test_build_streaming_diarizer_factory_rejects_non_streaming_provider():
    """A provider without streaming support cannot build a streaming factory."""
    with pytest.raises(ValueError, match="does not support streaming"):
        factory.build_streaming_diarizer_factory("pyannote", object())
