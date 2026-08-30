"""ASR window cache: content-addressed reuse of per-window ASR results.

The cache stores only digests and transcript tokens — never audio, never
decoded PCM — so its footprint is a few megabytes per hour of audio regardless
of how large the inputs were.

Its single integration point is :class:`~coro.cache.adapter.CachingASRAdapter`,
a decorator satisfying the ASR Adapter protocol. Every pipeline reaches the
model through that protocol, so wrapping it once at the ASR Backend Adapter
Factory covers the Full-Memory Pipeline, the Streaming Pipeline and the live
socket without any of them knowing the cache exists.
"""

from __future__ import annotations

from coro.cache.adapter import CachingASRAdapter, unwrap_asr_adapter
from coro.cache.directory import CacheDirectoryError, default_cache_dir, resolve_cache_dir
from coro.cache.fingerprint import (
    CACHE_FORMAT_VERSION,
    asr_fingerprint,
    fingerprint_components,
    normalise_language,
    window_key,
)
from coro.cache.store import ASRCacheStore

__all__ = [
    "CACHE_FORMAT_VERSION",
    "ASRCacheStore",
    "CacheDirectoryError",
    "CachingASRAdapter",
    "asr_fingerprint",
    "default_cache_dir",
    "fingerprint_components",
    "normalise_language",
    "resolve_cache_dir",
    "unwrap_asr_adapter",
    "window_key",
]
