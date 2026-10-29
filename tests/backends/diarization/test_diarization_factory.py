"""Diarization Backend Adapter Factory dispatch and streaming capability.

Verifies the factory dispatches to the right provider builder, reports
streaming capability correctly, builds the streaming diarizer only for
streaming-capable providers, and rejects unknown providers. No real NeMo
model is loaded: the tests give the factory in-memory provider builders and
an in-memory Sortformer model.
"""

from __future__ import annotations

import pytest

from coro.backends.diarization import factory
from coro.backends.diarization.nemo.diarization import NemoDiarizationAdapter
from coro.backends.diarization.nemo.streaming import NemoStreamingDiarizerFactory
from support.fake_backend_builders import (
    ConfiguredDiarization,
    load_configured_diarization_builder,
)
from support.fake_sortformer import FakeStreamingSortformer


def test_supports_streaming_nemo_true():
    """NeMo is a streaming-capable Diarization Backend Provider."""
    assert factory.supports_streaming("nemo") is True


def test_supports_streaming_pyannote_false():
    """Unknown/batch-only providers are not streaming-capable."""
    assert factory.supports_streaming("pyannote") is False
    assert factory.supports_streaming("bogus") is False


@pytest.mark.parametrize(
    ("provider", "module", "builder"),
    [
        (
            "nemo",
            "coro.backends.diarization.nemo.diarization",
            "build_nemo_diarization_adapter",
        ),
        ("pyannote", "coro.backends.diarization.pyannote", "build_pyannote_diarization_adapter"),
    ],
)
def test_each_provider_loads_its_own_builder(provider, module, builder):
    """Every Diarization Backend Provider resolves to the builder of its own module."""
    loaded = factory.load_diarization_adapter_builder(provider)
    assert (loaded.__module__, loaded.__name__) == (module, builder)


def test_build_diarization_adapter_dispatches_to_nemo():
    """The nemo provider routes to the NeMo adapter builder with device passthrough."""
    adapter = factory.build_diarization_adapter(
        "nemo", "some/model", device="cpu", load_builder=load_configured_diarization_builder
    )

    assert adapter == ConfiguredDiarization(
        provider="nemo",
        model="some/model",
        options={"device": "cpu", "postprocessing": None, "max_speakers": 4},
    )


def test_build_diarization_adapter_passes_postprocessing_to_nemo():
    """A Diarization Post-Processing Configuration value reaches the NeMo builder."""
    adapter = factory.build_diarization_adapter(
        "nemo",
        "some/model",
        device="cpu",
        postprocessing="dihard3-dev",
        load_builder=load_configured_diarization_builder,
    )

    assert adapter == ConfiguredDiarization(
        provider="nemo",
        model="some/model",
        options={"device": "cpu", "postprocessing": "dihard3-dev", "max_speakers": 4},
    )


def test_build_diarization_adapter_ignores_postprocessing_for_pyannote():
    """pyannote has no post-processing concept; the kwarg is not forwarded to it."""
    adapter = factory.build_diarization_adapter(
        "pyannote",
        "some/model",
        device="cpu",
        postprocessing="dihard3-dev",
        load_builder=load_configured_diarization_builder,
    )

    assert adapter == ConfiguredDiarization(
        provider="pyannote",
        model="some/model",
        options={"device": "cpu", "hf_token": None},
    )


def test_build_diarization_adapter_unknown_provider_raises():
    """An unknown Diarization Backend Provider fails fast."""
    with pytest.raises(ValueError, match="Unknown diarization backend provider"):
        factory.build_diarization_adapter("bogus", "model")


def test_build_streaming_diarizer_factory_nemo():
    """A NeMo adapter yields a streaming factory bound to the requested tier."""
    adapter = NemoDiarizationAdapter(FakeStreamingSortformer())

    streaming = factory.build_streaming_diarizer_factory("nemo", adapter, tier="low")

    assert isinstance(streaming, NemoStreamingDiarizerFactory)
    assert (streaming.tier, streaming.postprocessing_yaml, streaming.max_speakers) == (
        "low",
        None,
        4,
    )


def test_build_streaming_diarizer_factory_reuses_resolved_postprocessing():
    """The streaming factory reuses the batch adapter's already-resolved value.

    Resolved once by the batch adapter's construction, never re-resolved or
    re-validated for the streaming path. See ADR 0010.
    """
    adapter = NemoDiarizationAdapter(
        FakeStreamingSortformer(), postprocessing_yaml="/resolved/path.yaml"
    )

    streaming = factory.build_streaming_diarizer_factory("nemo", adapter, tier="low")

    assert isinstance(streaming, NemoStreamingDiarizerFactory)
    assert (streaming.tier, streaming.postprocessing_yaml, streaming.max_speakers) == (
        "low",
        "/resolved/path.yaml",
        4,
    )


def test_build_streaming_diarizer_factory_rejects_non_streaming_provider():
    """A provider without streaming support cannot build a streaming factory."""
    with pytest.raises(ValueError, match="does not support streaming"):
        factory.build_streaming_diarizer_factory("pyannote", object())
