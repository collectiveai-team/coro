"""GPU resource sampling via torch.cuda.

Provides a single public function, ``sample_gpu``, that returns per-tick GPU
metrics aggregated across all visible CUDA devices.  When CUDA is not available
the function returns empty strings so callers can treat the absence of GPU data
uniformly.
"""

from __future__ import annotations

from typing import Any


def sample_gpu() -> dict[str, Any]:
    """Sample GPU memory and utilisation across all visible CUDA devices.

    Returns a dict with these keys:

    - ``server_vram_mib``    - sum of used VRAM across all devices (MiB)
    - ``total_gpu_mem_mib``  - sum of total VRAM across all devices (MiB)
    - ``total_gpu_used_mib`` - alias of server_vram_mib (kept for schema compat)
    - ``gpu_util_pct``       - mean GPU utilisation across all devices (%)

    All values are empty strings when CUDA is unavailable or raises.
    """
    empty: dict[str, Any] = {
        "server_vram_mib": "",
        "total_gpu_mem_mib": "",
        "total_gpu_used_mib": "",
        "gpu_util_pct": "",
    }

    try:
        import torch

        if not torch.cuda.is_available():
            return empty

        count = torch.cuda.device_count()
        if count == 0:
            return empty

        total_mem = 0
        used_mem = 0
        util_sum = 0.0

        for i in range(count):
            free, total = torch.cuda.mem_get_info(i)
            total_mem += total
            used_mem += total - free
            util_sum += torch.cuda.utilization(i)

        return {
            "server_vram_mib": round(used_mem / 1024**2, 1),
            "total_gpu_mem_mib": round(total_mem / 1024**2, 1),
            "total_gpu_used_mib": round(used_mem / 1024**2, 1),
            "gpu_util_pct": round(util_sum / count, 1),
        }
    except Exception:
        return empty
