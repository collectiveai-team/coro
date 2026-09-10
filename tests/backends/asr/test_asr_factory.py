"""ASR Backend Adapter Factory dispatch.

Verifies the factory routes each ASR Backend Provider to its builder with the
right options from Server Startup Selection, rejects unknown providers, and
warns at startup when a provider-specific setting is configured for a provider
that ignores it. No real ASR model is loaded — provider builders are patched.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from coro.backends.asr.factory import build_asr_adapter, warn_ignored_asr_settings
from coro.settings import ServerSettings


def test_dispatches_to_faster_whisper():
    """The default faster-whisper provider routes to its builder."""
    settings = ServerSettings(backend_asr="faster-whisper", model_asr="m")
    sentinel = object()
    with patch(
        "coro.backends.asr.faster_whisper.build_asr_adapter", return_value=sentinel
    ) as mock_build:
        adapter = build_asr_adapter(settings)

    assert adapter is sentinel
    mock_build.assert_called_once_with(
        "m",
        device=settings.asr_device,
        compute_type=settings.asr_compute_type,
        max_concurrency=settings.asr_max_concurrency,
        max_queue_depth=settings.asr_max_queue_depth,
    )


def test_dispatches_to_onnx_asr():
    """The onnx-asr provider routes to its builder with VAD options."""
    settings = ServerSettings(backend_asr="onnx-asr", model_asr="m", asr_onnx_vad="enabled")
    sentinel = object()
    with patch(
        "coro.backends.asr.onnx_asr.build_onnx_asr_adapter", return_value=sentinel
    ) as mock_build:
        adapter = build_asr_adapter(settings)

    assert adapter is sentinel
    _, kwargs = mock_build.call_args
    assert kwargs["vad_enabled"] is True


def test_dispatches_to_onnx_parakeet_prompt():
    """The onnx-parakeet-prompt provider routes to its builder with quantization."""
    settings = ServerSettings(
        backend_asr="onnx-parakeet-prompt", model_asr="m", asr_quantization="static_qdq_v3"
    )
    sentinel = object()
    with patch(
        "coro.backends.asr.onnx_parakeet_prompt.build_onnx_parakeet_prompt_adapter",
        return_value=sentinel,
    ) as mock_build:
        adapter = build_asr_adapter(settings)

    assert adapter is sentinel
    mock_build.assert_called_once_with(
        "m",
        device=settings.asr_device,
        quantization="static_qdq_v3",
        max_queue_depth=settings.asr_max_queue_depth,
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
    sentinel = object()
    with patch(
        "coro.backends.asr.onnx_canary_split.build_onnx_canary_split_adapter",
        return_value=sentinel,
    ) as mock_build:
        adapter = build_asr_adapter(settings)

    assert adapter is sentinel
    mock_build.assert_called_once_with(
        "m",
        device=settings.asr_device,
        quantization="static_qdq_v3",
        decoder_quantization="dynamic_v1_quint8",
        max_concurrency=settings.asr_max_concurrency,
        max_queue_depth=settings.asr_max_queue_depth,
        hf_token=None,
        fallback_language=settings.asr_fallback_language,
    )


def test_dispatches_to_onnx_canary_split_with_hf_token(monkeypatch):
    """A configured ServerSettings.hf_token reaches the canary-split builder."""
    monkeypatch.setenv("CORO_HF_TOKEN", "secret-token")
    settings = ServerSettings(
        backend_asr="onnx-canary-split",
        model_asr="m",
        _env_file=None,
    )
    sentinel = object()
    with patch(
        "coro.backends.asr.onnx_canary_split.build_onnx_canary_split_adapter",
        return_value=sentinel,
    ) as mock_build:
        adapter = build_asr_adapter(settings)

    assert adapter is sentinel
    forwarded = mock_build.call_args.kwargs["hf_token"]
    assert forwarded is not None and forwarded == "secret-token"


def test_dispatches_to_onnx_genai():
    """The onnx-genai provider routes to its builder."""
    settings = ServerSettings(backend_asr="onnx-genai", model_asr="m")
    sentinel = object()
    with patch(
        "coro.backends.asr.onnx_genai.build_onnx_genai_adapter", return_value=sentinel
    ) as mock_build:
        adapter = build_asr_adapter(settings)

    assert adapter is sentinel
    mock_build.assert_called_once()


def test_dispatches_to_nemo():
    """The nemo provider routes to its builder without provider-specific knobs."""
    settings = ServerSettings(backend_asr="nemo", model_asr="m")
    sentinel = object()
    with patch(
        "coro.backends.asr.nemo.build_nemo_asr_adapter", return_value=sentinel
    ) as mock_build:
        adapter = build_asr_adapter(settings)

    assert adapter is sentinel
    mock_build.assert_called_once_with(
        "m",
        device=settings.asr_device,
        max_queue_depth=settings.asr_max_queue_depth,
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
    with (
        caplog.at_level(logging.WARNING, logger="coro.backends.asr.factory"),
        patch("coro.backends.asr.onnx_asr.build_onnx_asr_adapter", return_value=object()),
    ):
        build_asr_adapter(settings)

    assert "CORO_ASR_COMPUTE_TYPE" in caplog.text
