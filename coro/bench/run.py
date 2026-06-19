"""Process-tree resource sampling for the Resource Benchmark.

Reads per-process memory/CPU/IO counters from /proc for the full server
process tree. Consumed by coro.bench.sampling. Heavy imports are
kept out so importing coro.bench.cli stays lightweight.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

from coro.bench.models.resource import ProcessTreeSample


CLOCK_TICKS = os.sysconf(os.sysconf_names["SC_CLK_TCK"])


@dataclass
class SmapsRollup:
    """Resident-memory fields read from ``/proc/<pid>/smaps_rollup``."""

    pss: int = 0
    private_clean: int = 0
    private_dirty: int = 0


@dataclass
class ProcIo:
    """IO counters read from ``/proc/<pid>/io``."""

    rchar: int = 0
    wchar: int = 0
    read_bytes: int = 0
    write_bytes: int = 0


@dataclass
class ProcStat:
    """CPU/thread fields read from ``/proc/<pid>/stat``."""

    utime: int = 0
    stime: int = 0
    num_threads: int = 0


@dataclass
class ProcStatus:
    """Virtual-memory fields read from ``/proc/<pid>/status``."""

    vmrss: int = 0
    vmsize: int = 0


def _get_process_tree_pids(root_pid: int) -> set[int]:
    """Return all PIDs in the process tree rooted at root_pid."""
    pids = {root_pid}
    try:
        result = subprocess.run(
            ["pgrep", "-P", str(root_pid)],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.strip().splitlines():
            child = int(line)
            pids |= _get_process_tree_pids(child)
    except Exception:
        pass
    return pids


def _read_proc_smaps_rollup(pid: int) -> SmapsRollup:
    try:
        path = f"/proc/{pid}/smaps_rollup"
        data: dict[str, int] = {}
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0].endswith(":"):
                    key = parts[0][:-1]
                    try:
                        data[key] = int(parts[1])
                    except ValueError:
                        pass
        return SmapsRollup(
            pss=data.get("Pss", 0),
            private_clean=data.get("Private_Clean", 0),
            private_dirty=data.get("Private_Dirty", 0),
        )
    except Exception:
        return SmapsRollup()


def _read_proc_io(pid: int) -> ProcIo:
    try:
        data: dict[str, int] = {}
        with open(f"/proc/{pid}/io") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    try:
                        data[parts[0].rstrip(":")] = int(parts[1])
                    except ValueError:
                        pass
        return ProcIo(
            rchar=data.get("rchar", 0),
            wchar=data.get("wchar", 0),
            read_bytes=data.get("read_bytes", 0),
            write_bytes=data.get("write_bytes", 0),
        )
    except Exception:
        return ProcIo()


def _read_proc_stat(pid: int) -> ProcStat:
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().split()
        return ProcStat(
            utime=int(fields[13]),
            stime=int(fields[14]),
            num_threads=int(fields[19]),
        )
    except Exception:
        return ProcStat()


def _read_proc_status(pid: int) -> ProcStatus:
    try:
        data: dict[str, int] = {}
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0].endswith(":"):
                    key = parts[0][:-1]
                    if key in ("VmRSS", "VmSize"):
                        try:
                            data[key] = int(parts[1])
                        except ValueError:
                            pass
        return ProcStatus(vmrss=data.get("VmRSS", 0), vmsize=data.get("VmSize", 0))
    except Exception:
        return ProcStatus()


def sample_process_tree(root_pid: int) -> ProcessTreeSample:
    """Sample resource metrics for the full Server Process Tree."""
    pids = _get_process_tree_pids(root_pid)
    total_pss = total_uss = total_rss = total_vsz = 0
    total_utime = total_stime = total_threads = 0
    total_rchar = total_wchar = total_read_bytes = total_write_bytes = 0

    for pid in pids:
        smaps = _read_proc_smaps_rollup(pid)
        total_pss += smaps.pss
        total_uss += smaps.private_clean + smaps.private_dirty
        io = _read_proc_io(pid)
        total_rchar += io.rchar
        total_wchar += io.wchar
        total_read_bytes += io.read_bytes
        total_write_bytes += io.write_bytes
        stat = _read_proc_stat(pid)
        total_utime += stat.utime
        total_stime += stat.stime
        total_threads += stat.num_threads
        status = _read_proc_status(pid)
        total_rss += status.vmrss
        total_vsz += status.vmsize

    return ProcessTreeSample(
        pids=pids,
        pss_kb=total_pss,
        uss_kb=total_uss,
        rss_kb=total_rss,
        vsz_kb=total_vsz,
        cpu_user_s=total_utime / CLOCK_TICKS,
        cpu_system_s=total_stime / CLOCK_TICKS,
        rchar=total_rchar,
        wchar=total_wchar,
        read_bytes=total_read_bytes,
        write_bytes=total_write_bytes,
        thread_count=total_threads,
    )
