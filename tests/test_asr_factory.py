"""ASR Backend Adapter Factory dispatch.

Verifies the factory routes each ASR Backend Provider to its builder with the
right options from Server Startup Selection, and rejects unknown providers. No
real ASR model is loaded — provider builders are patched.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from coro.backends.asr.factory import build_asr_adapter
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
        "m", device=settings.asr_device, compute_type=settings.asr_compute_type
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
