"""Process-attributed GPU sample model.

Boundary model emitted by ``bench.gpu.sample_gpu`` and consumed by
``bench.sampling``. The raw per-device NVML struct (``GpuDevice``) stays local
to ``bench.gpu`` because it is a private read detail, never imported elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass


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
