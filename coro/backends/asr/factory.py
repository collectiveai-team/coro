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


# MARK: Auto-LID Capability
# Whether each ASR Backend Provider exposes ``detect_language``, known without
# building the adapter -- exactly the same reason ``PROVIDER_HONOURS_PROMPT``
# exists: ``LazyASRAdapter.detect_language`` must not force-load a model for a
# provider that has no auto-LID at all (that would defeat laziness's whole
# point on every request without an explicit language). Any provider absent
# here is assumed ``False``. ``test_asr_factory`` asserts this agrees with the
# ``detect_language`` each adapter class actually declares.
PROVIDER_DETECTS_LANGUAGE: dict[str, bool] = {
    "onnx-canary-split": True,
}


# MARK: Cross-Provider Setting Leakage
# Each entry maps a provider-specific setting to the ASR Backend Providers that
# actually honour it. Anything set for a provider outside its set is a no-op.
_PROVIDER_SPECIFIC_SETTINGS: tuple[tuple[str, frozenset[str]], ...] = (
    ("asr_compute_type", frozenset({"faster-whisper"})),
    ("asr_quantization", frozenset({"onnx-asr", "onnx-parakeet-prompt", "onnx-canary-split"})),
    ("asr_decoder_quantization", frozenset({"onnx-canary-split"})),
    ("asr_onnx_vad", frozenset({"onnx-asr"})),
    ("asr_onnx_vad_threshold", frozenset({"onnx-asr"})),
    ("asr_max_concurrency", frozenset({"faster-whisper", "onnx-asr", "onnx-canary-split"})),
)


def warn_ignored_asr_settings(settings: ServerSettings) -> list[str]:
    """Warn about ASR settings the configured Backend Provider ignores.

    A setting counts as configured when the *operator* actually set it (CLI
    flag, env var, or constructor kwarg) -- ``settings._explicit_fields``,
    captured by ``ServerSettings.resolve_model_slug`` before its own
    slug-filling mutations, so a value a model slug filled (e.g. the default
    slug's ``asr_decoder_quantization``) is never misattributed to the
    operator merely because it now differs from the field's declared default.

    Args:
        settings: Server Startup Selection to inspect.

    Returns:
        Names of the configured-but-ignored settings, in declaration order.

    """
    provider = settings.backend_asr
    explicit = settings._explicit_fields
    ignored: list[str] = []

    for name, honouring_providers in _PROVIDER_SPECIFIC_SETTINGS:
        if provider in honouring_providers:
            continue
        if name not in explicit:
            continue
        value = getattr(settings, name)
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
            decoder_quantization=settings.asr_decoder_quantization,
            max_concurrency=settings.asr_max_concurrency,
            max_queue_depth=settings.asr_max_queue_depth,
            hf_token=settings.hf_token.get_secret_value() if settings.hf_token else None,
            fallback_language=settings.asr_fallback_language,
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

    def __init__(
        self,
        build: Callable[[], ASRAdapter],
        *,
        honours_prompt: bool,
        detects_language: bool = False,
    ) -> None:
        """Defer adapter construction.

        Args:
            build: Zero-argument builder returning the real ASR Adapter.
            honours_prompt: The provider's declared prompt capability, known
                without building, so a cache fingerprint can be derived first.
            detects_language: The provider's declared auto-LID capability,
                known without building (see :data:`PROVIDER_DETECTS_LANGUAGE`)
                -- without it, :meth:`detect_language` would force-load the
                model on every request without an explicit language, for
                every provider, defeating laziness for the common case.

        """
        self._build = build
        self._adapter: ASRAdapter | None = None
        self.honours_prompt = honours_prompt
        self._detects_language = detects_language

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

    async def detect_language(self, pcm: bytes) -> str | None:
        """Resolve the adapter and forward, but only for a provider with auto-LID.

        Without ``detects_language`` gating this, EVERY lazily-wrapped
        request without an explicit language would force-load the model on
        window 1 just to learn it has no ``detect_language`` -- defeating
        laziness's whole point (a fully-cached run must load nothing) for
        every provider, not just auto-LID-capable ones. When it *is*
        capable, the same lazy load ``transcribe_pcm`` would have paid on a
        cache miss simply happens one call earlier.
        """
        if not self._detects_language:
            return None
        detect = getattr(self.resolve(), "detect_language", None)
        if detect is None:
            return None
        return await detect(pcm)


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
    detects_language = PROVIDER_DETECTS_LANGUAGE.get(provider, False)

    inner: ASRAdapter
    if lazy:
        inner = LazyASRAdapter(
            lambda: build_asr_adapter(settings),
            honours_prompt=honours_prompt,
            detects_language=detects_language,
        )
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
