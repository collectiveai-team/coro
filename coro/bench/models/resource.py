"""Aggregated process-tree resource sample model.

Boundary model between ``bench.run`` (which samples ``/proc``) and
``bench.sampling`` (which records rows). The per-process parsing structs
(``SmapsRollup``, ``ProcIo``, ``ProcStat``, ``ProcStatus``) stay local to
``bench.run`` because nothing outside that module consumes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProcessTreeSample:
    """Aggregated resource metrics for a full Server Process Tree."""

    pids: set[int] = field(default_factory=set)
    pss_kb: int = 0
    uss_kb: int = 0
    rss_kb: int = 0
    vsz_kb: int = 0
    cpu_user_s: float = 0.0
    cpu_system_s: float = 0.0
    rchar: int = 0
    wchar: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    thread_count: int = 0


@dataclass
class ResourceBaseline:
    """Post-warmup memory baseline used for prediction-memory deltas."""

    baseline_pss_kb: int | str = ""
    baseline_vram_mib: float | str = ""
