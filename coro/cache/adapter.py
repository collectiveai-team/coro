"""The ASR window cache's single integration point.

:class:`CachingASRAdapter` satisfies the ASR Adapter protocol and wraps the real
adapter. Everything a key needs — the window PCM, the language and the prompt —
is already present in ``transcribe_pcm``'s signature, so the decorator needs no
change to ASR Windowing, to either pipeline, or to any route, and therefore
covers the Full-Memory Pipeline, the Streaming Pipeline and the live socket
alike.

Store access happens on worker threads. The store is shared and has a busy
timeout, so unlike the uncontended per-request spill store it can block, and
blocking the event loop would make enabling the cache degrade concurrency.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from coro.cache.fingerprint import resolved_language_key, window_key

if TYPE_CHECKING:
    from coro.cache.store import ASRCacheStore
    from coro.core.models import TranscriptToken

logger = logging.getLogger(__name__)


def unwrap_asr_adapter(adapter: Any) -> Any:
    """Return the adapter beneath any ASR window cache decorator.

    Server Warmup uses this: warmup exists to prove the model loads and runs, so
    serving it from cache would let readiness report success on every start after
    the first without anything being loaded.

    Args:
        adapter: An ASR Adapter, possibly a :class:`CachingASRAdapter`.

    Returns:
        The wrapped adapter, or ``adapter`` itself when it is not a decorator.

    """
    inner = getattr(adapter, "inner", None)
    return adapter if inner is None else inner


def adapter_honours_prompt(adapter: Any) -> bool:
    """Return whether an ASR Adapter honours the prompt it is given.

    Args:
        adapter: An ASR Adapter.

    Returns:
        The adapter's declared capability, defaulting to ``True`` when it
        declares none — the pessimistic answer, which costs hit rate rather than
        correctness.

    """
    return bool(getattr(adapter, "honours_prompt", True))


class CachingASRAdapter:
    """An ASR Adapter that serves previously-computed windows from disk."""

    def __init__(
        self,
        inner: Any,
        *,
        store: ASRCacheStore,
        fingerprint: str,
        honours_prompt: bool | None = None,
    ) -> None:
        """Wrap an ASR Adapter with the ASR window cache.

        Args:
            inner: The ASR Adapter to consult on a miss.
            store: Shared window store.
            fingerprint: Digest of everything outside the request that can change
                a prediction.
            honours_prompt: Whether ``inner`` honours the prompt. Defaults to the
                capability ``inner`` declares. Must match the value that produced
                ``fingerprint``.

        """
        self._inner = inner
        self._store = store
        self._fingerprint = fingerprint
        self.honours_prompt = (
            adapter_honours_prompt(inner) if honours_prompt is None else honours_prompt
        )
        # A backend that publishes a fallback_language (currently only
        # onnx-canary-split) also reduces a request language to a base
        # subtag and resolves an absent one to that fallback -- the cache key
        # must collapse the same way, or es-US/es (or an absent language)
        # would each get their own cache entry despite decoding identically.
        self._fallback_language: str | None = getattr(inner, "fallback_language", None)
        self._hits = 0
        self._misses = 0

    @property
    def inner(self) -> Any:
        """The wrapped ASR Adapter, for callers that must reach the real model."""
        return self._inner

    async def detect_language(self, pcm: bytes) -> str | None:
        """Forward to the wrapped adapter's auto-LID, when it has one.

        Plain passthrough, never cached: detection is a cheap, separate
        inference call (see ``OnnxCanarySplitASRAdapter.detect_language``),
        and what the window cache keys on is the *resolved* language a window
        was actually decoded with, not the detection call itself. Pipelines
        duck-type this the same way (``getattr(asr, "detect_language",
        None)``), so a backend without one is simply absent here too.
        """
        detect = getattr(self._inner, "detect_language", None)
        if detect is None:
            return None
        return await detect(pcm)

    @property
    def fingerprint(self) -> str:
        """Digest identifying the configuration this decorator caches under."""
        return self._fingerprint

    @property
    def hits(self) -> int:
        """Windows served from the cache since construction."""
        return self._hits

    @property
    def misses(self) -> int:
        """Windows that reached the model since construction."""
        return self._misses

    async def transcribe_pcm(
        self,
        pcm: bytes,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> list[TranscriptToken]:
        """Return the window's tokens, from cache when this exact window is known.

        Args:
            pcm: Canonical PCM bytes for one ASR Windowing window.
            language: Requested language, normalised into the key.
            prompt: Carried prompt. Entered into the key only when the wrapped
                backend honours it, so a prompt-inert backend gets independent
                per-window keys.

        Returns:
            The window's transcript tokens.

        """
        language_key = (
            resolved_language_key(language, fallback_language=self._fallback_language)
            if self._fallback_language is not None
            else None
        )
        key = window_key(
            pcm,
            fingerprint=self._fingerprint,
            language=language,
            language_key=language_key,
            prompt=prompt if self.honours_prompt else None,
        )
        cached = await asyncio.to_thread(self._store.get, key)
        if cached is not None:
            self._hits += 1
            logger.debug("asr_cache hit key=%s tokens=%d", key[:12], len(cached))
            return cached

        self._misses += 1
        tokens = await self._inner.transcribe_pcm(pcm, language=language, prompt=prompt)
        # Committed per window, so a run killed part-way keeps everything it
        # already transcribed and the next attempt resumes where it stopped.
        await asyncio.to_thread(self._store.put, key, tokens)
        return tokens
