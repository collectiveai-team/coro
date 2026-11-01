"""The usable CPU budget: affinity mask capped by a cgroup CPU quota.

Each test lays out a cgroup filesystem under ``tmp_path`` the way a container
runtime mounts it at ``/sys/fs/cgroup``, and reads the budget through it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from coro.backends.asr.cpu_budget import (
    affinity_cpu_count,
    cgroup_cpu_quota,
    parse_cgroup_v1_cfs,
    parse_cgroup_v2_cpu_max,
    usable_cpu_count,
)


def _cgroup_v2(root: Path, cpu_max: str) -> Path:
    (root / "cpu.max").write_text(cpu_max, encoding="ascii")
    return root


def _cgroup_v1(root: Path, quota: str, period: str, controller: str = "cpu") -> Path:
    directory = root / controller
    directory.mkdir()
    (directory / "cpu.cfs_quota_us").write_text(quota, encoding="ascii")
    (directory / "cpu.cfs_period_us").write_text(period, encoding="ascii")
    return root


@pytest.mark.parametrize(
    ("text", "cores"),
    [
        ("400000 100000\n", 4.0),
        ("250000 100000\n", 2.5),
        ("max 100000\n", None),
        ("", None),
        ("garbage\n", None),
        ("abc 100000\n", None),
        ("100000 0\n", None),
    ],
)
def test_cgroup_v2_cpu_max_is_quota_over_period(text, cores):
    """``cpu.max`` grants quota/period cores; ``max`` and malformed text grant no quota."""
    assert parse_cgroup_v2_cpu_max(text) == cores


@pytest.mark.parametrize(
    ("quota", "period", "cores"),
    [
        ("200000\n", "100000\n", 2.0),
        ("-1\n", "100000\n", None),
        ("x\n", "100000\n", None),
    ],
)
def test_cgroup_v1_cfs_is_quota_over_period(quota, period, cores):
    """A CFS quota grants quota/period cores; ``-1`` grants no quota."""
    assert parse_cgroup_v1_cfs(quota, period) == cores


def test_no_cgroup_filesystem_means_no_quota(tmp_path):
    """A host with nothing to read under the cgroup root reports no quota."""
    assert cgroup_cpu_quota(tmp_path) is None
    assert usable_cpu_count(tmp_path) == affinity_cpu_count()


def test_cgroup_v1_quota_is_read_from_either_controller_name(tmp_path):
    """Runtimes mount the v1 CPU controller as ``cpu`` or ``cpu,cpuacct``."""
    root = _cgroup_v1(tmp_path, "100000", "100000", controller="cpu,cpuacct")
    assert usable_cpu_count(root) == 1


def test_quota_caps_the_affinity_mask(tmp_path):
    """A one-core limit yields one usable core, whatever the machine has."""
    assert usable_cpu_count(_cgroup_v2(tmp_path, "100000 100000")) == 1


def test_fractional_quota_rounds_up(tmp_path):
    """Half a core still runs one thread; it is not rounded down to zero."""
    assert usable_cpu_count(_cgroup_v2(tmp_path, "50000 100000")) == 1


def test_quota_above_the_affinity_mask_does_not_raise_it(tmp_path):
    """A quota larger than the cores the process may run on leaves the mask as the budget."""
    cores = affinity_cpu_count()
    assert usable_cpu_count(_cgroup_v2(tmp_path, f"{(cores + 8) * 100000} 100000")) == cores


def test_unlimited_quota_leaves_the_affinity_mask(tmp_path):
    """``cpu.max`` of ``max`` is no limit: the budget is the affinity mask."""
    assert usable_cpu_count(_cgroup_v2(tmp_path, "max 100000")) == affinity_cpu_count()


def test_v1_quota_caps_the_affinity_mask(tmp_path):
    """The v1 CFS quota caps the budget the same way as v2 ``cpu.max``."""
    assert usable_cpu_count(_cgroup_v1(tmp_path, "100000", "100000")) == 1
