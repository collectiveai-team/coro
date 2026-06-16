"""Process-tree resource sampling for the Resource Benchmark.

Reads per-process memory/CPU/IO counters from /proc for the full server
process tree. Consumed by coro.bench.sampling. Heavy imports are
kept out so importing coro.bench.cli stays lightweight.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any


CLOCK_TICKS = os.sysconf(os.sysconf_names["SC_CLK_TCK"])


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


def _read_proc_smaps_rollup(pid: int) -> dict[str, int]:
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
        return data
    except Exception:
        return {}


def _read_proc_io(pid: int) -> dict[str, int]:
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
        return data
    except Exception:
        return {}


def _read_proc_stat(pid: int) -> dict[str, Any]:
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().split()
        return {
            "utime": int(fields[13]),
            "stime": int(fields[14]),
            "num_threads": int(fields[19]),
        }
    except Exception:
        return {}


def _read_proc_status(pid: int) -> dict[str, int]:
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
        return data
    except Exception:
        return {}


def sample_process_tree(root_pid: int) -> dict[str, Any]:
    """Sample resource metrics for the full Server Process Tree."""
    pids = _get_process_tree_pids(root_pid)
    total_pss = total_uss = total_rss = total_vsz = 0
    total_utime = total_stime = total_threads = 0
    total_rchar = total_wchar = total_read_bytes = total_write_bytes = 0

    for pid in pids:
        smaps = _read_proc_smaps_rollup(pid)
        total_pss += smaps.get("Pss", 0)
        total_uss += smaps.get("Private_Clean", 0) + smaps.get("Private_Dirty", 0)
        io = _read_proc_io(pid)
        total_rchar += io.get("rchar", 0)
        total_wchar += io.get("wchar", 0)
        total_read_bytes += io.get("read_bytes", 0)
        total_write_bytes += io.get("write_bytes", 0)
        stat = _read_proc_stat(pid)
        total_utime += stat.get("utime", 0)
        total_stime += stat.get("stime", 0)
        total_threads += stat.get("num_threads", 0)
        status = _read_proc_status(pid)
        total_rss += status.get("VmRSS", 0)
        total_vsz += status.get("VmSize", 0)

    return {
        "pids": pids,
        "pss_kb": total_pss,
        "uss_kb": total_uss,
        "rss_kb": total_rss,
        "vsz_kb": total_vsz,
        "cpu_user_s": total_utime / CLOCK_TICKS,
        "cpu_system_s": total_stime / CLOCK_TICKS,
        "rchar": total_rchar,
        "wchar": total_wchar,
        "read_bytes": total_read_bytes,
        "write_bytes": total_write_bytes,
        "thread_count": total_threads,
    }
