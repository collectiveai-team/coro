"""ASR window cache directory resolution.

The cache is persistent, so unlike the transcript spill store its default is
the user cache directory rather than the system temp dir — a cache the OS
reclaims on reboot would miss on exactly the re-runs it exists to serve.

A RAM-backed directory is rejected for the same reason the spill store rejects
one: it would compete for the memory the cache is meant to save, and would lose
every entry on restart. Resolution runs during Strict Startup Validation, so a
misconfiguration fails before the first request rather than on it.
"""

from __future__ import annotations

import os
from pathlib import Path

from coro.fsinfo import filesystem_type, is_ram_backed

_SET_HINT = (
    "Set CORO_ASR_CACHE_DIR (or --asr-cache-dir) to a directory on real disk, "
    "or run with CORO_ASR_CACHE=disabled."
)


class CacheDirectoryError(ValueError):
    """The configured ASR window cache directory cannot hold a persistent cache."""


def default_cache_dir() -> str:
    """Return the default ASR window cache directory under the user cache root."""
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    cache_root = Path(xdg_cache) if xdg_cache else Path.home() / ".cache"
    return str(cache_root / "coro" / "asr-window-cache")


def resolve_cache_dir(configured: str | None) -> str:
    """Resolve the effective ASR window cache directory, rejecting RAM-backed ones.

    Args:
        configured: Explicitly configured directory, or ``None`` to use the default.

    Returns:
        A directory path on a filesystem that is not known to be RAM-backed. The
        directory is not created here; the store creates it on demand.

    Raises:
        CacheDirectoryError: When the resolved directory is RAM-backed.

    """
    resolved = configured if configured is not None else default_cache_dir()
    if is_ram_backed(resolved) is True:
        raise CacheDirectoryError(
            f"ASR window cache directory {resolved!r} is on a RAM-backed filesystem "
            f"({filesystem_type(resolved)}). The cache exists to save recomputation "
            f"across runs, so a directory that lives in host memory both competes "
            f"with the memory it was meant to save and loses every entry on restart. "
            f"{_SET_HINT}"
        )
    return resolved
