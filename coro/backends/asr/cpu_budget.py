"""The CPU cores this process may use, which auto-sized ASR concurrency is drawn from.

``os.cpu_count()`` reports every core of the machine. Inside a container with a
CPU limit (a Kubernetes ``resources.limits.cpu``, ``docker run --cpus``) the
process may use far fewer: the limit is a cgroup CPU quota, enforced by
throttling. Sizing concurrency from the machine's cores oversubscribes the quota
and every in-flight call is throttled.

The budget is the CPU affinity mask, capped by the quota of the cgroup mounted
at ``/sys/fs/cgroup``. Under a cgroup namespace, which container runtimes create
by default on cgroup v2, that is the container's own cgroup. On a bare host it is
the root cgroup, which has no quota, so the budget is the affinity mask.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

CGROUP_ROOT = Path("/sys/fs/cgroup")


# MARK: Quota Parsing
def parse_cgroup_v2_cpu_max(text: str) -> float | None:
    """Return the cores a cgroup v2 ``cpu.max`` grants, or ``None`` when unlimited.

    Args:
        text: Contents of ``cpu.max``: ``"<quota> <period>"`` in microseconds,
            or ``"max <period>"`` when there is no quota.

    Returns:
        Quota divided by period, or ``None`` for no quota or an unreadable value.

    """
    fields = text.split()
    if len(fields) != 2 or fields[0] == "max":
        return None
    return _quota_cores(fields[0], fields[1])


def parse_cgroup_v1_cfs(quota_text: str, period_text: str) -> float | None:
    """Return the cores a cgroup v1 CFS quota grants, or ``None`` when unlimited.

    Args:
        quota_text: Contents of ``cpu.cfs_quota_us``; ``-1`` means no quota.
        period_text: Contents of ``cpu.cfs_period_us``.

    Returns:
        Quota divided by period, or ``None`` for no quota or an unreadable value.

    """
    return _quota_cores(quota_text.strip(), period_text.strip())


def _quota_cores(quota: str, period: str) -> float | None:
    try:
        quota_us, period_us = int(quota), int(period)
    except ValueError:
        return None
    if quota_us <= 0 or period_us <= 0:
        return None
    return quota_us / period_us


# MARK: Host Reads
def cgroup_cpu_quota(cgroup_root: Path = CGROUP_ROOT) -> float | None:
    """Read the CPU quota of the cgroup mounted at ``cgroup_root``.

    Args:
        cgroup_root: Mount point of the cgroup filesystem.

    Returns:
        Cores the quota grants, or ``None`` when there is no quota or no
        cgroup filesystem to read.

    """
    v2 = _read(cgroup_root / "cpu.max")
    if v2 is not None:
        return parse_cgroup_v2_cpu_max(v2)
    for controller in ("cpu", "cpu,cpuacct"):
        quota = _read(cgroup_root / controller / "cpu.cfs_quota_us")
        period = _read(cgroup_root / controller / "cpu.cfs_period_us")
        if quota is not None and period is not None:
            return parse_cgroup_v1_cfs(quota, period)
    return None


def affinity_cpu_count() -> int:
    """Return the number of cores in this process's CPU affinity mask."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0)) or 1
    return os.cpu_count() or 1


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError):
        return None


# MARK: Budget
def usable_cpu_count(cgroup_root: Path = CGROUP_ROOT) -> int:
    """Return the cores this process may use: its affinity mask, capped by a cgroup quota.

    A fractional quota rounds up, so a 2.5-core limit counts as 3 cores and a
    0.5-core limit as 1.

    Args:
        cgroup_root: Mount point of the cgroup filesystem.

    Returns:
        Usable core count (always >= 1).

    """
    cpus = affinity_cpu_count()
    quota = cgroup_cpu_quota(cgroup_root)
    if quota is None:
        return cpus
    return max(1, min(cpus, math.ceil(quota)))
