"""GPU resource sampling via pynvml.

Provides a single public function, ``sample_gpu``, that returns per-tick GPU
metrics aggregated across all visible NVIDIA devices.  When pynvml is not
installed or no CUDA devices are present the function returns empty strings so
callers can treat the absence of GPU data uniformly.
"""

from __future__ import annotations

from typing import Any

# Lazy module-level initialisation state so we only call nvmlInit once.
_nvml_available: bool | None = None  # None = not yet probed


def _ensure_nvml() -> bool:
    """Return True if pynvml was successfully initialised, False otherwise."""
    global _nvml_available
    if _nvml_available is not None:
        return _nvml_available
    try:
        import pynvml

        pynvml.nvmlInit()
        _nvml_available = True
    except Exception:
        _nvml_available = False
    return _nvml_available


def sample_gpu() -> dict[str, Any]:
    """Sample GPU memory and utilisation across all visible NVIDIA devices.

    Returns a dict with these keys:

    - ``server_vram_mib``    - sum of used VRAM across all devices (MiB)
    - ``total_gpu_mem_mib``  - sum of total VRAM across all devices (MiB)
    - ``total_gpu_used_mib`` - alias of server_vram_mib (kept for schema compat)
    - ``gpu_util_pct``       - mean GPU utilisation across all devices (%)

    All values are empty strings when pynvml is unavailable or raises.
    """
    empty: dict[str, Any] = {
        "server_vram_mib": "",
        "total_gpu_mem_mib": "",
        "total_gpu_used_mib": "",
        "gpu_util_pct": "",
    }

    if not _ensure_nvml():
        return empty

    try:
        import pynvml

        count = pynvml.nvmlDeviceGetCount()
        if count == 0:
            return empty

        total_mem = 0
        used_mem = 0
        util_sum = 0.0

        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            total_mem += mem.total
            used_mem += mem.used
            util_sum += util.gpu

        return {
            "server_vram_mib": round(used_mem / 1024**2, 1),
            "total_gpu_mem_mib": round(total_mem / 1024**2, 1),
            "total_gpu_used_mib": round(used_mem / 1024**2, 1),
            "gpu_util_pct": round(util_sum / count, 1),
        }
    except Exception:
        return empty
