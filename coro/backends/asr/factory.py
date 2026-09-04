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
    from collections.abc import Callable

    from coro.core.models import TranscriptToken
    from coro.core.protocols import ASRAdapter
    from coro.settings import ServerSettings

logger = logging.getLogger(__name__)


# MARK: Prompt Capability
# Whether each ASR Backend Provider honours the prompt, known without building
# the adapter. The ASR window cache needs this before any model is loaded, since
# a fully-cached run must never load one. ``test_asr_factory`` asserts it agrees
# with the ``honours_prompt`` each adapter class declares, so the two cannot drift.
PROVIDER_HONOURS_PROMPT: dict[str, bool] = {
    "onnx-asr": False,
    "onnx-genai": False,
    "nemo": False,
    "onnx-parakeet-prompt": False,
    "onnx-canary-split": False,
    "faster-whisper": True,
}


# MARK: Cross-Provider Setting Leakage
# Each entry maps a provider-specific setting to the ASR Backend Providers that
# actually honour it. Anything set for a provider outside its set is a no-op.
_PROVIDER_SPECIFIC_SETTINGS: tuple[tuple[str, frozenset[str]], ...] = (
    ("asr_compute_type", frozenset({"faster-whisper"})),
    ("asr_quantization", frozenset({"onnx-asr", "onnx-parakeet-prompt", "onnx-canary-split"})),
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

    if provider == "onnx-parakeet-prompt":
        from coro.backends.asr.onnx_parakeet_prompt import build_onnx_parakeet_prompt_adapter

        return build_onnx_parakeet_prompt_adapter(
            settings.model_asr,
            device=settings.asr_device,
            quantization=settings.asr_quantization,
            max_queue_depth=settings.asr_max_queue_depth,
        )

    if provider == "onnx-canary-split":
        from coro.backends.asr.onnx_canary_split import build_onnx_canary_split_adapter

        return build_onnx_canary_split_adapter(
            settings.model_asr,
            device=settings.asr_device,
            quantization=settings.asr_quantization,
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

    if provider == "nemo":
        from coro.backends.asr.nemo import build_nemo_asr_adapter

        return build_nemo_asr_adapter(
            settings.model_asr,
            device=settings.asr_device,
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


# MARK: Deferred Construction
class LazyASRAdapter:
    """An ASR Adapter that builds the real one on first inference.

    Loading a model costs seconds and hundreds of megabytes, and a fully-cached
    run needs neither: every window is answered from disk. Deferring construction
    is what makes such a run genuinely fast rather than merely inference-free.

    Server startup keeps eager construction so Server Warmup and readiness
    semantics are unchanged; this is for the offline command.
    """

    def __init__(self, build: Callable[[], ASRAdapter], *, honours_prompt: bool) -> None:
        """Defer adapter construction.

        Args:
            build: Zero-argument builder returning the real ASR Adapter.
            honours_prompt: The provider's declared prompt capability, known
                without building, so a cache fingerprint can be derived first.

        """
        self._build = build
        self._adapter: ASRAdapter | None = None
        self.honours_prompt = honours_prompt

    @property
    def loaded(self) -> bool:
        """Whether the real adapter has been constructed."""
        return self._adapter is not None

    def resolve(self) -> ASRAdapter:
        """Return the real adapter, constructing it on first call."""
        if self._adapter is None:
            self._adapter = self._build()
        return self._adapter

    async def transcribe_pcm(
        self,
        pcm: bytes,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> list[TranscriptToken]:
        """Build the adapter if needed, then transcribe through it."""
        return await self.resolve().transcribe_pcm(pcm, language=language, prompt=prompt)


# MARK: ASR Adapter Stack
def build_asr_adapter_stack(settings: ServerSettings, *, lazy: bool = False) -> ASRAdapter:
    """Build the ASR Adapter, wrapped by the ASR window cache when it is enabled.

    This is the cache's only integration point. Every pipeline and every route
    reaches the model through the returned object, so the cache cannot drift out
    of sync with them.

    Args:
        settings: Server Startup Selection.
        lazy: Defer construction of the real adapter until the first window that
            actually misses the cache.

    Returns:
        An ASR Adapter, possibly a ``CachingASRAdapter`` decorator.

    """
    provider = settings.backend_asr
    honours_prompt = PROVIDER_HONOURS_PROMPT.get(provider, True)

    inner: ASRAdapter
    if lazy:
        inner = LazyASRAdapter(lambda: build_asr_adapter(settings), honours_prompt=honours_prompt)
    else:
        inner = build_asr_adapter(settings)

    if settings.asr_cache != "enabled":
        return inner

    from coro.cache.adapter import CachingASRAdapter
    from coro.cache.fingerprint import asr_fingerprint
    from coro.cache.store import ASRCacheStore

    directory = settings.asr_cache_dir or ""
    store = ASRCacheStore(
        directory,
        max_bytes=settings.asr_cache_max_mb * 1024 * 1024,
        ttl_seconds=settings.asr_cache_ttl_days * 86400.0,
    )
    fingerprint = asr_fingerprint(settings, honours_prompt=honours_prompt)
    logger.info(
        "ASR window cache enabled dir=%s fingerprint=%s entries=%d bytes=%d",
        directory,
        fingerprint,
        store.entry_count(),
        store.total_bytes(),
    )
    return CachingASRAdapter(
        inner, store=store, fingerprint=fingerprint, honours_prompt=honours_prompt
    )
