"""Transcript spill-directory resolution and RAM-backed filesystem detection.

The Streaming Pipeline spills its growing transcript to a per-request on-disk
store so host memory stays flat on arbitrarily long audio.  That guarantee is
void when the spill directory lives on a RAM-backed filesystem: on most Linux
distributions ``/tmp`` is ``tmpfs``, so the system temp dir — the historical
default — silently kept the whole transcript in memory.

This module makes that failure impossible to ship silently:

- :func:`resolve_spill_dir` picks a real-disk default when none is configured.
- An explicitly configured RAM-backed directory raises :class:`SpillDirectoryError`,
  which Strict Startup Validation surfaces before the server serves requests.

Detection is Linux-specific (``/proc/self/mountinfo``).  Where the filesystem
type cannot be determined the directory is accepted rather than rejected: an
undetermined mount is not evidence of a RAM-backed one.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

# Filesystem types whose pages are host memory, never storage.
RAM_BACKED_FS_TYPES = frozenset({"tmpfs", "ramfs", "devtmpfs"})

_MOUNTINFO_PATH = Path("/proc/self/mountinfo")
_OCTAL_ESCAPE = re.compile(r"\\([0-7]{3})")

_SET_HINT = (
    "Set CORO_TRANSCRIPT_SPILL_DIR (or --transcript-spill-dir) to a directory "
    "on real disk, or run with CORO_PIPELINE=full-memory."
)


class SpillDirectoryError(ValueError):
    """The configured transcript spill directory cannot keep host memory flat."""


def _unescape_mount_field(value: str) -> str:
    """Decode the octal escapes mountinfo uses for spaces, tabs and newlines."""
    return _OCTAL_ESCAPE.sub(lambda match: chr(int(match.group(1), 8)), value)


def _mount_table() -> list[tuple[Path, str]]:
    """Return ``(mount_point, filesystem_type)`` pairs, empty when unavailable.

    Parses ``/proc/self/mountinfo``, whose optional-field section is terminated
    by a lone ``-``; the filesystem type is the first field after it.
    """
    try:
        text = _MOUNTINFO_PATH.read_text(encoding="utf-8")
    except OSError:
        return []

    table: list[tuple[Path, str]] = []
    for line in text.splitlines():
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        before_fields = before.split()
        after_fields = after.split()
        if len(before_fields) < 5 or not after_fields:
            continue
        table.append((Path(_unescape_mount_field(before_fields[4])), after_fields[0]))
    return table


def _nearest_existing(path: Path) -> Path:
    """Return ``path`` or its closest existing ancestor.

    A spill directory may not exist yet; its filesystem is then the one that
    will host it once created.
    """
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def filesystem_type(path: str | os.PathLike[str]) -> str | None:
    """Return the filesystem type hosting ``path``, or None when undeterminable.

    Args:
        path: Directory path, which need not exist yet.

    Returns:
        The mount's filesystem type (e.g. ``"ext4"``, ``"tmpfs"``), or ``None``
        when the mount table is unreadable or covers no ancestor of ``path``.

    """
    try:
        target = _nearest_existing(Path(path).resolve())
    except OSError:
        return None

    best: tuple[Path, str] | None = None
    for mount_point, fs_type in _mount_table():
        if target != mount_point and mount_point not in target.parents:
            continue
        # Longest matching mount point wins: /var/lib beats / for /var/lib/x.
        if best is None or len(mount_point.parts) > len(best[0].parts):
            best = (mount_point, fs_type)
    return best[1] if best is not None else None


def is_ram_backed(path: str | os.PathLike[str]) -> bool | None:
    """Return whether ``path`` lives on a RAM-backed filesystem.

    Args:
        path: Directory path, which need not exist yet.

    Returns:
        ``True`` or ``False`` when the filesystem type is known, ``None`` when
        it could not be determined (non-Linux hosts, unreadable mount table).

    """
    fs_type = filesystem_type(path)
    if fs_type is None:
        return None
    return fs_type in RAM_BACKED_FS_TYPES


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
