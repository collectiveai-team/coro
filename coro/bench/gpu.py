"""Per-process GPU resource sampling via NVML.

``sample_gpu`` attributes VRAM to the **Server Process Tree** rather than
reporting whole-device usage. Whole-device totals are retained as context.

Attributing VRAM to the server's own PIDs prevents other GPU tenants — and the
CUDA caching reserve of unrelated processes — from inflating the headline VRAM
number. It also makes the baseline-corrected ``peak_vram_delta_mib`` a fair
cross-pipeline comparison, because two runs no longer differ merely by whatever
idle pool happened to be resident on the device when they started.

Returned keys:

- ``server_vram_mib``    - VRAM used by processes in ``pids`` (process-attributed)
- ``total_gpu_mem_mib``  - sum of total VRAM across all devices (MiB)
- ``total_gpu_used_mib`` - whole-device used VRAM across all devices (MiB)
- ``gpu_util_pct``       - mean GPU utilisation across all devices (%)

All values are empty strings when NVML is unavailable so callers can treat the
absence of GPU data uniformly.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field

_MIB = 1024**2


@dataclass
class GpuSample:
    """Process-attributed GPU columns.

    Fields hold ``""`` (empty string) when NVML is unavailable so callers can
    treat the absence of GPU data uniformly, mirroring the CSV output.
    """

    server_vram_mib: float | str = ""
    total_gpu_mem_mib: float | str = ""
    total_gpu_used_mib: float | str = ""
    gpu_util_pct: float | str = ""


@dataclass
class GpuDevice:
    """Raw per-device NVML sample.

    ``mem_total``/``mem_used`` are bytes, ``util`` is percent, and ``procs`` is
    a list of ``(pid, used_bytes_or_None)`` for compute and graphics processes.
    """

    mem_total: int
    mem_used: int
    util: float
    procs: list[tuple[int, int | None]] = field(default_factory=list)


def _read_devices() -> list[GpuDevice] | None:
    """Read raw per-device memory, utilisation, and per-process VRAM from NVML.

    Returns ``None`` when NVML is unavailable (no driver, no GPU, import error)
    so callers can emit empty GPU columns.
    """
    try:
        import pynvml
    except Exception:
        return None

    try:
        pynvml.nvmlInit()
    except Exception:
        return None

    try:
        devices: list[GpuDevice] = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                util = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            except Exception:
                util = 0.0

            procs: list[tuple[int, int | None]] = []
            for query in (
                pynvml.nvmlDeviceGetComputeRunningProcesses,
                pynvml.nvmlDeviceGetGraphicsRunningProcesses,
            ):
                try:
                    for proc in query(handle):
                        used = getattr(proc, "usedGpuMemory", None)
                        procs.append((int(proc.pid), used if isinstance(used, int) else None))
                except Exception:
                    continue

            devices.append(
                GpuDevice(
                    mem_total=int(mem.total),
                    mem_used=int(mem.used),
                    util=util,
                    procs=procs,
                )
            )
        return devices
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _aggregate(devices: list[GpuDevice], pids: Collection[int] | None) -> GpuSample:
    """Aggregate raw device samples into the GPU columns.

    When ``pids`` is given, ``server_vram_mib`` sums only the VRAM used by
    processes whose PID is in the Server Process Tree, so a CPU-only run (no
    server PID on the GPU) reports 0.0. When ``pids`` is ``None`` it falls back
    to whole-device used VRAM for baseline/legacy callers.
    """
    total_mem = sum(d.mem_total for d in devices)
    total_used = sum(d.mem_used for d in devices)
    util = sum(d.util for d in devices) / len(devices)

    if pids is None:
        server_vram_bytes = total_used
    else:
        pidset = set(pids)
        server_vram_bytes = sum(
            used for d in devices for pid, used in d.procs if used is not None and pid in pidset
        )

    return GpuSample(
        server_vram_mib=round(server_vram_bytes / _MIB, 1),
        total_gpu_mem_mib=round(total_mem / _MIB, 1),
        total_gpu_used_mib=round(total_used / _MIB, 1),
        gpu_util_pct=round(util, 1),
    )


def sample_gpu(pids: Collection[int] | None = None) -> GpuSample:
    """Sample GPU memory and utilisation, attributing VRAM to ``pids``.

    ``pids`` is the Server Process Tree PID set. Pass it so the reported
    ``server_vram_mib`` reflects only the server's own GPU memory rather than
    whole-device usage. Returns empty strings when NVML is unavailable.
    """
    devices = _read_devices()
    if not devices:
        return GpuSample()
    return _aggregate(devices, pids)
