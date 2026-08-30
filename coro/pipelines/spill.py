"""Transcript spill-directory resolution.

The Streaming Pipeline spills its growing transcript to a per-request on-disk
store so host memory stays flat on arbitrarily long audio.  That guarantee is
void when the spill directory lives on a RAM-backed filesystem: on most Linux
distributions ``/tmp`` is ``tmpfs``, so the system temp dir — the historical
default — silently kept the whole transcript in memory.

This module makes that failure impossible to ship silently:

- :func:`resolve_spill_dir` picks a real-disk default when none is configured.
- An explicitly configured RAM-backed directory raises :class:`SpillDirectoryError`,
  which Strict Startup Validation surfaces before the server serves requests.

The RAM-backed detection itself lives in :mod:`coro.fsinfo`, shared with the
persistent ASR window cache, which has the same requirement for the same reason.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from coro.fsinfo import RAM_BACKED_FS_TYPES, filesystem_type, is_ram_backed

__all__ = [
    "RAM_BACKED_FS_TYPES",
    "SpillDirectoryError",
    "default_spill_dir_candidates",
    "filesystem_type",
    "is_ram_backed",
    "resolve_spill_dir",
]

_SET_HINT = (
    "Set CORO_TRANSCRIPT_SPILL_DIR (or --transcript-spill-dir) to a directory "
    "on real disk, or run with CORO_PIPELINE=full-memory."
)


class SpillDirectoryError(ValueError):
    """The configured transcript spill directory cannot keep host memory flat."""


def default_spill_dir_candidates() -> list[str]:
    """Return the default spill directories to try, best first.

    The system temp dir is preferred because the OS already reclaims it; the
    user cache dir is the fallback for the common case where temp is tmpfs.
    """
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    cache_root = Path(xdg_cache) if xdg_cache else Path.home() / ".cache"
    return [tempfile.gettempdir(), str(cache_root / "coro" / "transcript-spill")]


def resolve_spill_dir(configured: str | None) -> str:
    """Resolve the effective transcript spill directory, rejecting RAM-backed ones.

    Args:
        configured: Explicitly configured directory, or ``None`` to pick a default.

    Returns:
        A directory path on a filesystem that is not known to be RAM-backed.
        The directory is not created here; the spill store creates it on demand.

    Raises:
        SpillDirectoryError: When the configured directory is RAM-backed, or
            when every default candidate is.

    """
    if configured is not None:
        if is_ram_backed(configured) is True:
            raise SpillDirectoryError(
                f"Transcript spill directory {configured!r} is on a RAM-backed "
                f"filesystem ({filesystem_type(configured)}). The Streaming Pipeline "
                f"spills the transcript there to keep host memory flat, so a "
                f"RAM-backed directory defeats the spill entirely. {_SET_HINT}"
            )
        return configured

    rejected: list[str] = []
    for candidate in default_spill_dir_candidates():
        if is_ram_backed(candidate) is not True:
            return candidate
        rejected.append(f"{candidate} ({filesystem_type(candidate)})")

    raise SpillDirectoryError(
        "No default transcript spill directory is on real disk; every candidate "
        f"is RAM-backed: {', '.join(rejected)}. The Streaming Pipeline needs a "
        f"real-disk directory to keep host memory flat. {_SET_HINT}"
    )
