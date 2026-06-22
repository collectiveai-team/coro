"""ASR Backend Adapter Factory.

Dispatches on the configured ASR Backend Provider to build an ASR Adapter,
keeping provider selection and per-provider argument mapping out of the
application factory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coro.core.protocols import ASRAdapter
    from coro.settings import ServerSettings


def build_asr_adapter(settings: ServerSettings) -> ASRAdapter:
    """Build an ASR Adapter for the configured ASR Backend Provider.

    Args:
        settings: Server Startup Selection providing the ASR Backend Provider,
            ASR Model Selection, and provider-specific options.

    Returns:
        A ready-to-use ASR Adapter.

    Raises:
        ValueError: If the ASR Backend Provider is unknown.

    """
    provider = settings.backend_asr

    if provider == "onnx-asr":
        from coro.backends.asr.onnx_asr import build_onnx_asr_adapter

        return build_onnx_asr_adapter(
            settings.model_asr,
            device=settings.asr_device,
            quantization=settings.asr_quantization,
            vad_enabled=settings.asr_onnx_vad == "enabled",
            vad_threshold=settings.asr_onnx_vad_threshold,
        )

    if provider == "onnx-genai":
        from coro.backends.asr.onnx_genai import build_onnx_genai_adapter

        return build_onnx_genai_adapter(
            settings.model_asr,
            device=settings.asr_device,
            quantization=settings.asr_quantization,
        )

    if provider == "faster-whisper":
        from coro.backends.asr.faster_whisper import build_asr_adapter as build_faster_whisper

        return build_faster_whisper(
            settings.model_asr,
            device=settings.asr_device,
            compute_type=settings.asr_compute_type,
        )

    msg = f"Unknown ASR backend provider: {provider!r}"
    raise ValueError(msg)
