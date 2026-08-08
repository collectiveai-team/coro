"""RAM-backed transcript spill directories are avoided by default or rejected.

The Streaming Pipeline's flat-memory guarantee depends on the spill store
living on real disk. The historical default was the system temp dir, which is
tmpfs on most Linux distributions — so the store that exists to keep host
memory flat spilled straight back into memory, silently.

The mount table is faked so these tests assert real parsing and real selection
logic without depending on how the host happens to be mounted.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import ValidationError

from coro.pipelines import spill
from coro.pipelines.spill import (
    SpillDirectoryError,
    filesystem_type,
    is_ram_backed,
    resolve_spill_dir,
)
from coro.settings import ServerSettings

_ROOT_ENTRY = "25 30 8:1 / / rw,relatime shared:1 - ext4 /dev/sda1 rw"


def _mountinfo_line(index: int, mount_point: Path, fs_type: str) -> str:
    """Render one /proc/self/mountinfo line for a mount point and filesystem."""
    escaped = str(mount_point).replace(" ", r"\040")
    return f"{index} 25 0:{index} / {escaped} rw,relatime shared:{index} - {fs_type} {fs_type} rw"


@dataclass
class _FakeMounts:
    """Directories installed into a synthetic mount table, addressable by name."""

    directories: dict[str, Path] = field(default_factory=dict)

    def path(self, name: str) -> Path:
        """Return the real directory registered under ``name``."""
        return self.directories[name]


@pytest.fixture
def fake_mounts(monkeypatch, tmp_path):
    """Install a synthetic mount table describing directories under tmp_path."""

    def _install(**fs_type_by_name: str) -> _FakeMounts:
        mounts = _FakeMounts()
        lines = [_ROOT_ENTRY]
        for index, (name, fs_type) in enumerate(fs_type_by_name.items(), start=26):
            directory = (tmp_path / name).resolve()
            directory.mkdir(parents=True, exist_ok=True)
            mounts.directories[name] = directory
            lines.append(_mountinfo_line(index, directory, fs_type))
        mountinfo = tmp_path / "mountinfo"
        mountinfo.write_text("\n".join(lines) + "\n", encoding="utf-8")
        monkeypatch.setattr(spill, "_MOUNTINFO_PATH", mountinfo)
        return mounts

    return _install


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_filesystem_type_reports_the_longest_matching_mount(fake_mounts):
    mounts = fake_mounts(ram="tmpfs", disk="xfs")
    assert filesystem_type(mounts.path("ram")) == "tmpfs"
    assert filesystem_type(mounts.path("disk")) == "xfs"


def test_filesystem_type_of_an_unlisted_path_falls_back_to_its_ancestor(fake_mounts):
    fake_mounts(disk="xfs")
    assert filesystem_type("/") == "ext4"


def test_filesystem_type_covers_a_directory_that_does_not_exist_yet(fake_mounts):
    mounts = fake_mounts(ram="tmpfs")
    assert filesystem_type(mounts.path("ram") / "not" / "created" / "yet") == "tmpfs"


@pytest.mark.parametrize("fs_type", sorted(spill.RAM_BACKED_FS_TYPES))
def test_is_ram_backed_flags_every_ram_backed_filesystem(fake_mounts, fs_type: str):
    mounts = fake_mounts(volatile=fs_type)
    assert is_ram_backed(mounts.path("volatile")) is True


def test_is_ram_backed_is_false_on_real_disk(fake_mounts):
    mounts = fake_mounts(disk="btrfs")
    assert is_ram_backed(mounts.path("disk")) is False


def test_is_ram_backed_is_undetermined_without_a_mount_table(monkeypatch, tmp_path):
    monkeypatch.setattr(spill, "_MOUNTINFO_PATH", tmp_path / "absent")
    assert is_ram_backed(tmp_path) is None


def test_mount_points_containing_spaces_are_decoded(monkeypatch, tmp_path):
    directory = (tmp_path / "spill dir").resolve()
    directory.mkdir()
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        "\n".join([_ROOT_ENTRY, _mountinfo_line(26, directory, "tmpfs")]) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(spill, "_MOUNTINFO_PATH", mountinfo)
    assert is_ram_backed(directory) is True


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_configured_real_disk_directory_is_used_as_is(fake_mounts):
    mounts = fake_mounts(disk="ext4")
    assert resolve_spill_dir(str(mounts.path("disk"))) == str(mounts.path("disk"))


def test_configured_ram_backed_directory_is_rejected(fake_mounts):
    mounts = fake_mounts(ram="tmpfs")
    with pytest.raises(SpillDirectoryError) as excinfo:
        resolve_spill_dir(str(mounts.path("ram")))

    message = str(excinfo.value)
    assert "RAM-backed" in message
    assert "tmpfs" in message
    assert "CORO_TRANSCRIPT_SPILL_DIR" in message


def test_default_prefers_the_temp_dir_when_it_is_real_disk(monkeypatch, fake_mounts):
    mounts = fake_mounts(disk="ext4")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(mounts.path("disk")))
    assert resolve_spill_dir(None) == str(mounts.path("disk"))


def test_default_avoids_a_ram_backed_temp_dir(monkeypatch, fake_mounts):
    mounts = fake_mounts(ram="tmpfs", cache="ext4")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(mounts.path("ram")))
    monkeypatch.setenv("XDG_CACHE_HOME", str(mounts.path("cache")))

    resolved = resolve_spill_dir(None)

    assert is_ram_backed(resolved) is not True
    assert Path(resolved).is_relative_to(mounts.path("cache"))


def test_default_fails_loudly_when_every_candidate_is_ram_backed(monkeypatch, fake_mounts):
    mounts = fake_mounts(ram="tmpfs", cache="ramfs")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(mounts.path("ram")))
    monkeypatch.setenv("XDG_CACHE_HOME", str(mounts.path("cache")))

    with pytest.raises(SpillDirectoryError, match="RAM-backed"):
        resolve_spill_dir(None)


# ---------------------------------------------------------------------------
# Strict Startup Validation
# ---------------------------------------------------------------------------


def test_streaming_startup_resolves_a_real_disk_spill_dir(monkeypatch, fake_mounts):
    mounts = fake_mounts(disk="ext4")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(mounts.path("disk")))

    settings = ServerSettings(pipeline="streaming", _env_file=None)

    assert settings.transcript_spill_dir == str(mounts.path("disk"))


def test_streaming_startup_rejects_a_ram_backed_spill_dir(fake_mounts):
    mounts = fake_mounts(ram="tmpfs")
    with pytest.raises(ValidationError, match="RAM-backed"):
        ServerSettings(
            pipeline="streaming",
            transcript_spill_dir=str(mounts.path("ram")),
            _env_file=None,
        )


def test_full_memory_startup_leaves_the_spill_dir_untouched(fake_mounts):
    mounts = fake_mounts(ram="tmpfs")
    settings = ServerSettings(
        pipeline="full-memory",
        transcript_spill_dir=str(mounts.path("ram")),
        _env_file=None,
    )

    assert settings.transcript_spill_dir == str(mounts.path("ram"))
    assert ServerSettings(pipeline="full-memory", _env_file=None).transcript_spill_dir is None
