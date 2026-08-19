"""Filesystem probing: which mount hosts a path, and is it RAM-backed.

Two on-disk stores depend on their directory being real storage rather than
host memory: the Streaming Pipeline's per-request transcript spill store, and
the persistent ASR window cache. Both would silently defeat their own purpose on
a ``tmpfs`` directory — on most Linux distributions ``/tmp`` is exactly that —
so both resolve their directory through the same probe rather than each
re-implementing the detection.

Detection is Linux-specific (``/proc/self/mountinfo``). Where the filesystem
type cannot be determined the directory is accepted rather than rejected: an
undetermined mount is not evidence of a RAM-backed one.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# Filesystem types whose pages are host memory, never storage.
RAM_BACKED_FS_TYPES = frozenset({"tmpfs", "ramfs", "devtmpfs"})

_MOUNTINFO_PATH = Path("/proc/self/mountinfo")
_OCTAL_ESCAPE = re.compile(r"\\([0-7]{3})")


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

    A store directory may not exist yet; its filesystem is then the one that
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
