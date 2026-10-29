"""ASR Backend Adapter Factory dispatch.

Verifies the factory routes each ASR Backend Provider to its builder with the
right options from Server Startup Selection, rejects unknown providers, and
warns at startup when a provider-specific setting is configured for a provider
that ignores it. No real ASR model is loaded: the tests give the factory
in-memory provider builders.
"""

from __future__ import annotations

import logging

import pytest

from coro.backends.asr.factory import (
    build_asr_adapter,
    load_asr_adapter_builder,
    warn_ignored_asr_settings,
)
from coro.settings import ServerSettings
from support.fake_backend_builders import ConfiguredASR, load_configured_asr_builder


def _build(settings: ServerSettings) -> ConfiguredASR:
    adapter = build_asr_adapter(settings, load_builder=load_configured_asr_builder)
    assert isinstance(adapter, ConfiguredASR)
    return adapter


@pytest.mark.parametrize(
    ("provider", "module", "builder"),
    [
        ("faster-whisper", "coro.backends.asr.faster_whisper", "build_asr_adapter"),
        ("onnx-asr", "coro.backends.asr.onnx_asr", "build_onnx_asr_adapter"),
        (
            "onnx-parakeet-prompt",
            "coro.backends.asr.onnx_parakeet_prompt",
            "build_onnx_parakeet_prompt_adapter",
        ),
        (
            "onnx-canary-split",
            "coro.backends.asr.onnx_canary_split",
            "build_onnx_canary_split_adapter",
        ),
        ("onnx-genai", "coro.backends.asr.onnx_genai", "build_onnx_genai_adapter"),
        ("nemo", "coro.backends.asr.nemo", "build_nemo_asr_adapter"),
    ],
)
def test_each_provider_loads_its_own_builder(provider, module, builder):
    """Every ASR Backend Provider resolves to the builder of its own module."""
    loaded = load_asr_adapter_builder(provider)
    assert (loaded.__module__, loaded.__name__) == (module, builder)


def test_dispatches_to_faster_whisper():
    """The default faster-whisper provider routes to its builder."""
    settings = ServerSettings(backend_asr="faster-whisper", model_asr="m")

    assert _build(settings) == ConfiguredASR(
        provider="faster-whisper",
        model="m",
        options={
            "device": settings.asr_device,
            "compute_type": settings.asr_compute_type,
            "max_concurrency": settings.asr_max_concurrency,
            "max_queue_depth": settings.asr_max_queue_depth,
        },
    )


def test_dispatches_to_onnx_asr():
    """The onnx-asr provider routes to its builder with VAD options."""
    settings = ServerSettings(backend_asr="onnx-asr", model_asr="m", asr_onnx_vad="enabled")

    adapter = _build(settings)

    assert adapter.provider == "onnx-asr"
    assert adapter.options["vad_enabled"] is True


def test_dispatches_to_onnx_parakeet_prompt():
    """The onnx-parakeet-prompt provider routes to its builder with quantization."""
    settings = ServerSettings(
        backend_asr="onnx-parakeet-prompt", model_asr="m", asr_quantization="static_qdq_v3"
    )

    assert _build(settings) == ConfiguredASR(
        provider="onnx-parakeet-prompt",
        model="m",
        options={
            "device": settings.asr_device,
            "quantization": "static_qdq_v3",
            "max_queue_depth": settings.asr_max_queue_depth,
        },
    )


def _no_hf_token_in_env(monkeypatch):
    """Keep a developer's real CORO_HF_TOKEN/HF_TOKEN out of these assertions."""
    for name in ("CORO_HF_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def test_dispatches_to_onnx_canary_split(monkeypatch):
    """The onnx-canary-split provider routes to its builder with both quantization selectors."""
    _no_hf_token_in_env(monkeypatch)
    settings = ServerSettings(
        backend_asr="onnx-canary-split",
        model_asr="m",
        asr_quantization="static_qdq_v3",
        asr_decoder_quantization="dynamic_v1_quint8",
        _env_file=None,
    )

    assert _build(settings) == ConfiguredASR(
        provider="onnx-canary-split",
        model="m",
        options={
            "device": settings.asr_device,
            "quantization": "static_qdq_v3",
            "decoder_quantization": "dynamic_v1_quint8",
            "max_concurrency": settings.asr_max_concurrency,
            "max_queue_depth": settings.asr_max_queue_depth,
            "hf_token": None,
            "fallback_language": settings.asr_fallback_language,
        },
    )


def test_dispatches_to_onnx_canary_split_with_hf_token(monkeypatch):
    """A configured ServerSettings.hf_token reaches the canary-split builder."""
    monkeypatch.setenv("CORO_HF_TOKEN", "secret-token")
    settings = ServerSettings(
        backend_asr="onnx-canary-split",
        model_asr="m",
        _env_file=None,
    )

    forwarded = _build(settings).options["hf_token"]
    assert forwarded == "secret-token"


def test_dispatches_to_onnx_genai():
    """The onnx-genai provider routes to its builder."""
    settings = ServerSettings(backend_asr="onnx-genai", model_asr="m")

    assert _build(settings) == ConfiguredASR(
        provider="onnx-genai",
        model="m",
        options={
            "device": settings.asr_device,
            "quantization": settings.asr_quantization,
            "max_queue_depth": settings.asr_max_queue_depth,
        },
    )


def test_dispatches_to_nemo():
    """The nemo provider routes to its builder without provider-specific knobs."""
    settings = ServerSettings(backend_asr="nemo", model_asr="m")

    assert _build(settings) == ConfiguredASR(
        provider="nemo",
        model="m",
        options={
            "device": settings.asr_device,
            "max_queue_depth": settings.asr_max_queue_depth,
        },
    )


def test_unknown_provider_raises():
    """An unknown ASR Backend Provider fails fast."""
    settings = ServerSettings(backend_asr="onnx-asr", model_asr="m")
    object.__setattr__(settings, "backend_asr", "bogus")
    with pytest.raises(ValueError, match="Unknown ASR backend provider"):
        build_asr_adapter(settings)


# ---------------------------------------------------------------------------
# Cross-provider setting leakage warnings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "overrides", "expected"),
    [
        ("onnx-asr", {"asr_compute_type": "int8"}, ["asr_compute_type"]),
        ("onnx-genai", {"asr_compute_type": "int8"}, ["asr_compute_type"]),
        ("faster-whisper", {"asr_quantization": "int8"}, ["asr_quantization"]),
        ("onnx-genai", {"asr_quantization": "int8"}, ["asr_quantization"]),
        ("nemo", {"asr_quantization": "int8"}, ["asr_quantization"]),
        ("onnx-parakeet-prompt", {"asr_compute_type": "int8"}, ["asr_compute_type"]),
        (
            "onnx-asr",
            {"asr_decoder_quantization": "dynamic_v1_quint8"},
            ["asr_decoder_quantization"],
        ),
        (
            "onnx-parakeet-prompt",
            {"asr_decoder_quantization": "dynamic_v1_quint8"},
            ["asr_decoder_quantization"],
        ),
        ("faster-whisper", {"asr_onnx_vad": "enabled"}, ["asr_onnx_vad"]),
        ("faster-whisper", {"asr_onnx_vad_threshold": 0.4}, ["asr_onnx_vad_threshold"]),
        ("onnx-genai", {"asr_max_concurrency": 8}, ["asr_max_concurrency"]),
        ("onnx-parakeet-prompt", {"asr_max_concurrency": 8}, ["asr_max_concurrency"]),
    ],
)
def test_warns_when_a_setting_is_ignored_by_the_provider(provider, overrides, expected, caplog):
    """A knob set for a provider that ignores it is reported at startup."""
    settings = ServerSettings(backend_asr=provider, model_asr="m", **overrides)
    with caplog.at_level(logging.WARNING, logger="coro.backends.asr.factory"):
        ignored = warn_ignored_asr_settings(settings)

    assert ignored == expected
    assert f"CORO_{expected[0].upper()}" in caplog.text
    assert provider in caplog.text


@pytest.mark.parametrize(
    ("provider", "overrides"),
    [
        ("faster-whisper", {"asr_compute_type": "int8"}),
        ("onnx-asr", {"asr_quantization": "int8"}),
        ("onnx-parakeet-prompt", {"asr_quantization": "static_qdq_v3"}),
        ("onnx-canary-split", {"asr_quantization": "static_qdq_v3"}),
        ("onnx-canary-split", {"asr_decoder_quantization": "dynamic_v1_quint8"}),
        ("onnx-asr", {"asr_onnx_vad": "enabled", "asr_onnx_vad_threshold": 0.4}),
        ("onnx-asr", {"asr_max_concurrency": 8}),
        ("onnx-canary-split", {"asr_max_concurrency": 8}),
    ],
)
def test_no_warning_when_the_provider_honours_the_setting(provider, overrides):
    """Settings the selected provider actually honours are not reported."""
    settings = ServerSettings(backend_asr=provider, model_asr="m", **overrides)
    assert warn_ignored_asr_settings(settings) == []


def test_no_warning_for_unset_settings():
    """Leaving provider-specific knobs at their defaults is never reported."""
    settings = ServerSettings(backend_asr="onnx-asr", model_asr="m")
    assert warn_ignored_asr_settings(settings) == []


def test_no_warning_for_the_default_model_slugs_own_filled_settings():
    """A model slug's own quantization fill is never mistaken for an operator setting."""
    settings = ServerSettings(_env_file=None)  # default: canary-1b-v2 slug
    assert settings.asr_quantization == "static_qdq_v4_pct_excl"
    assert settings.asr_decoder_quantization == "dynamic_v1_quint8"
    assert warn_ignored_asr_settings(settings) == []


def test_slug_filled_quantization_is_not_warned_about_when_backend_asr_is_overridden(caplog):
    """An operator overriding backend_asr away from the default slug's own backend.

    asr_quantization/asr_decoder_quantization were filled by the slug, not by
    the operator, so no CORO_ASR_*QUANTIZATION warning should name a setting
    the operator never touched -- even though the newly-selected backend
    (onnx-asr) does not honour asr_decoder_quantization at all.
    """
    settings = ServerSettings(_env_file=None, backend_asr="onnx-asr")
    assert settings.asr_decoder_quantization == "dynamic_v1_quint8"  # slug-filled, unused here
    with caplog.at_level(logging.WARNING, logger="coro.backends.asr.factory"):
        ignored = warn_ignored_asr_settings(settings)
    assert ignored == []
    assert "asr_decoder_quantization" not in caplog.text


def test_build_emits_the_ignored_setting_warning(caplog):
    """The warning is emitted as part of building the adapter at startup."""
    settings = ServerSettings(backend_asr="onnx-asr", model_asr="m", asr_compute_type="int8")
    with caplog.at_level(logging.WARNING, logger="coro.backends.asr.factory"):
        _build(settings)

    assert "CORO_ASR_COMPUTE_TYPE" in caplog.text
