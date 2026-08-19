"""Synthetic mount table used to exercise RAM-backed-filesystem rejection.

Shared by the transcript spill store and the ASR Window Cache, which reject
RAM-backed directories for the same reason and through the same probe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT_MOUNT_ENTRY = "25 30 8:1 / / rw,relatime shared:1 - ext4 /dev/sda1 rw"


def mountinfo_line(index: int, mount_point: Path, fs_type: str) -> str:
    """Render one /proc/self/mountinfo line for a mount point and filesystem."""
    escaped = str(mount_point).replace(" ", r"\040")
    return f"{index} 25 0:{index} / {escaped} rw,relatime shared:{index} - {fs_type} {fs_type} rw"


@dataclass
class FakeMounts:
    """Directories installed into a synthetic mount table, addressable by name."""

    directories: dict[str, Path] = field(default_factory=dict)

    def path(self, name: str) -> Path:
        """Return the real directory registered under ``name``."""
        return self.directories[name]
