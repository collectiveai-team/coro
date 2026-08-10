"""ASR Backend Adapter Factory.

Dispatches on the configured ASR Backend Provider to build an ASR Adapter,
keeping provider selection and per-provider argument mapping out of the
application factory.

It is also where Server Startup Selection values that the selected provider
silently ignores are surfaced as warnings. A no-op knob that logs nothing makes
benchmark results uninterpretable: the run looks configured but is not.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coro.core.protocols import ASRAdapter
    from coro.settings import ServerSettings

logger = logging.getLogger(__name__)


# MARK: Cross-Provider Setting Leakage
# Each entry maps a provider-specific setting to the ASR Backend Providers that
# actually honour it. Anything set for a provider outside its set is a no-op.
_PROVIDER_SPECIFIC_SETTINGS: tuple[tuple[str, frozenset[str]], ...] = (
    ("asr_compute_type", frozenset({"faster-whisper"})),
    ("asr_quantization", frozenset({"onnx-asr"})),
    ("asr_onnx_vad", frozenset({"onnx-asr"})),
    ("asr_onnx_vad_threshold", frozenset({"onnx-asr"})),
    ("asr_max_concurrency", frozenset({"faster-whisper", "onnx-asr"})),
)


def warn_ignored_asr_settings(settings: ServerSettings) -> list[str]:
    """Warn about ASR settings the configured Backend Provider ignores.

    A setting counts as configured when its value differs from its declared
    default, so leaving a knob unset is never reported.

    Args:
        settings: Server Startup Selection to inspect.

    Returns:
        Names of the configured-but-ignored settings, in declaration order.

    """
    provider = settings.backend_asr
    fields = type(settings).model_fields
    ignored: list[str] = []

    for name, honouring_providers in _PROVIDER_SPECIFIC_SETTINGS:
        if provider in honouring_providers:
            continue
        value = getattr(settings, name)
        if value == fields[name].default:
            continue
        ignored.append(name)
        logger.warning(
            "Setting CORO_%s=%r is ignored by the '%s' ASR Backend Provider "
            "(honoured by: %s). It will have no effect on this run.",
            name.upper(),
            value,
            provider,
            ", ".join(sorted(honouring_providers)),
        )

    return ignored


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
    warn_ignored_asr_settings(settings)

    if provider == "onnx-asr":
        from coro.backends.asr.onnx_asr import build_onnx_asr_adapter

        return build_onnx_asr_adapter(
            settings.model_asr,
            device=settings.asr_device,
            quantization=settings.asr_quantization,
            vad_enabled=settings.asr_onnx_vad == "enabled",
            vad_threshold=settings.asr_onnx_vad_threshold,
            max_concurrency=settings.asr_max_concurrency,
            max_queue_depth=settings.asr_max_queue_depth,
        )

    if provider == "onnx-genai":
        from coro.backends.asr.onnx_genai import build_onnx_genai_adapter

        return build_onnx_genai_adapter(
            settings.model_asr,
            device=settings.asr_device,
            quantization=settings.asr_quantization,
            max_queue_depth=settings.asr_max_queue_depth,
        )

    if provider == "faster-whisper":
        from coro.backends.asr.faster_whisper import build_asr_adapter as build_faster_whisper

        return build_faster_whisper(
            settings.model_asr,
            device=settings.asr_device,
            compute_type=settings.asr_compute_type,
            max_concurrency=settings.asr_max_concurrency,
            max_queue_depth=settings.asr_max_queue_depth,
        )

    msg = f"Unknown ASR backend provider: {provider!r}"
    raise ValueError(msg)
