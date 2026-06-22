"""Diarization Backend Adapter Factory.

Dispatches on the configured Diarization Backend Provider to build a
Diarization Adapter, reports which providers support the Streaming Pipeline,
and constructs the per-request streaming diarizer for streaming-capable
providers. Keeps provider selection out of the application factory and
encapsulates backend-specific construction details (e.g. the NeMo model
handle the streaming factory needs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coro.core.protocols import DiarizationAdapter, StreamingDiarizerFactory


# Diarization Backend Providers that can drive the Streaming Pipeline.
_STREAMING_CAPABLE: frozenset[str] = frozenset({"nemo"})


def supports_streaming(provider: str) -> bool:
    """Return True when a Diarization Backend Provider supports streaming."""
    return provider in _STREAMING_CAPABLE


def build_diarization_adapter(
    provider: str,
    model_diarization: str,
    *,
    device: str = "auto",
    hf_token: str | None = None,
) -> DiarizationAdapter:
    """Build a Diarization Adapter for the configured provider.

    Args:
        provider: Diarization Backend Provider selector (e.g. ``nemo``).
        model_diarization: Diarization Model Selection (HF-style id or path).
        device: ``auto``/``cuda``/``cpu`` device selector.
        hf_token: HuggingFace token for gated models; ignored by providers
            that do not require one.

    Returns:
        A ready-to-use Diarization Adapter.

    Raises:
        ValueError: If the provider is unknown.

    """
    if provider == "nemo":
        from coro.backends.diarization.nemo.diarization import build_nemo_diarization_adapter

        return build_nemo_diarization_adapter(model_diarization, device=device)

    if provider == "pyannote":
        from coro.backends.diarization.pyannote import build_pyannote_diarization_adapter

        return build_pyannote_diarization_adapter(
            model_diarization, device=device, hf_token=hf_token
        )

    msg = f"Unknown diarization backend provider: {provider!r}"
    raise ValueError(msg)


def build_streaming_diarizer_factory(
    provider: str,
    adapter: DiarizationAdapter,
    *,
    tier: str = "very-high",
) -> StreamingDiarizerFactory:
    """Build a StreamingDiarizerFactory for a streaming-capable provider.

    Args:
        provider: Diarization Backend Provider selector.
        adapter: The batch Diarization Adapter built for the same provider;
            its shared model is reused by the streaming factory.
        tier: Diarization Latency tier for the streaming diarizer.

    Returns:
        A StreamingDiarizerFactory bound to the shared model.

    Raises:
        ValueError: If the provider does not support streaming.

    """
    if provider == "nemo":
        from coro.backends.diarization.nemo.diarization import NemoDiarizationAdapter
        from coro.backends.diarization.nemo.streaming import NemoStreamingDiarizerFactory

        if not isinstance(adapter, NemoDiarizationAdapter):
            msg = "Streaming diarizer factory for 'nemo' requires a NemoDiarizationAdapter."
            raise TypeError(msg)
        return NemoStreamingDiarizerFactory(adapter.model, tier=tier)

    msg = f"Diarization backend provider {provider!r} does not support streaming."
    raise ValueError(msg)
