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


def test_unknown_provider_raises():
    """An unknown ASR Backend Provider fails fast."""
    settings = ServerSettings(model_asr="m")
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
        ("faster-whisper", {"asr_onnx_vad": "enabled"}, ["asr_onnx_vad"]),
        ("faster-whisper", {"asr_onnx_vad_threshold": 0.4}, ["asr_onnx_vad_threshold"]),
        ("onnx-genai", {"asr_max_concurrency": 8}, ["asr_max_concurrency"]),
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
        ("onnx-asr", {"asr_onnx_vad": "enabled", "asr_onnx_vad_threshold": 0.4}),
        ("onnx-asr", {"asr_max_concurrency": 8}),
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


def test_build_emits_the_ignored_setting_warning(caplog):
    """The warning is emitted as part of building the adapter at startup."""
    settings = ServerSettings(backend_asr="onnx-asr", model_asr="m", asr_compute_type="int8")
    with (
        caplog.at_level(logging.WARNING, logger="coro.backends.asr.factory"),
        patch("coro.backends.asr.onnx_asr.build_onnx_asr_adapter", return_value=object()),
    ):
        build_asr_adapter(settings)

    assert "CORO_ASR_COMPUTE_TYPE" in caplog.text
