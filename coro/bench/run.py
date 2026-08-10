"""Process-tree resource sampling for the Resource Benchmark.

Reads per-process memory/CPU/IO counters from /proc for the full server
process tree. Consumed by coro.bench.sampling. Heavy imports are
kept out so importing coro.bench.cli stays lightweight.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

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


def _stat_fields_after_comm(pid: int) -> list[str] | None:
    """Return ``/proc/<pid>/stat`` fields from ``state`` onward, or None.

    A process's ``comm`` is parenthesised and may itself contain spaces and
    parentheses, so splitting the whole line shifts every field after it.
    Slicing at the *last* ``)`` is the documented way to parse this — see
    proc(5). The returned list is 0-indexed from field 3, so proc(5) field N
    is at index ``N - 3``.
    """
    try:
        with open(f"/proc/{pid}/stat") as f:
            line = f.read()
        return line[line.rindex(")") + 1 :].split()
    except (OSError, ValueError):
        return None


@dataclass(frozen=True)
class ProcessTreeIndex:
    """Parent-to-children index over every process visible in /proc."""

    children: dict[int, list[int]] = field(default_factory=dict)

    def children_of(self, pid: int) -> list[int]:
        return self.children.get(pid, [])


def _read_child_pid_map() -> ProcessTreeIndex:
    """Index parent PID to child PIDs with a single scan of /proc.

    The previous implementation shelled out to ``pgrep -P`` once per PID,
    recursively, on *every* sample. At the 0.25 s sampling interval that is a
    fork storm proportional to the tree size — measured at 1176 subprocesses
    and 34 s in a single benchmark test — and the sampler's own forks perturb
    the CPU utilisation it exists to measure. One /proc scan costs no forks.
    """
    children: dict[int, list[int]] = {}
    try:
        entries = list(os.scandir("/proc"))
    except OSError:
        return ProcessTreeIndex(children)

    for entry in entries:
        if not entry.name.isdigit():
            continue
        fields = _stat_fields_after_comm(int(entry.name))
        # index 1 == proc(5) field 4 == ppid
        if fields is None or len(fields) < 2:
            continue
        try:
            ppid = int(fields[1])
        except ValueError:
            continue
        children.setdefault(ppid, []).append(int(entry.name))
    return ProcessTreeIndex(children)


def _get_process_tree_pids(root_pid: int) -> set[int]:
    """Return all PIDs in the process tree rooted at root_pid."""
    index = _read_child_pid_map()
    pids = {root_pid}
    pending = [root_pid]
    while pending:
        for child in index.children_of(pending.pop()):
            if child not in pids:
                pids.add(child)
                pending.append(child)
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
    fields = _stat_fields_after_comm(pid)
    if fields is None:
        return ProcStat()
    try:
        # proc(5) fields 14, 15 and 20, offset by the 3 that precede `state`.
        return ProcStat(
            utime=int(fields[11]),
            stime=int(fields[12]),
            num_threads=int(fields[17]),
        )
    except (IndexError, ValueError):
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
