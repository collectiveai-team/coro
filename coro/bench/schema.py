"""Stable Resource Schema for Resource CSV output.

The column list is a package constant so benchmark analysis notebooks
and downstream tooling can import it without parsing CSV headers.
"""

from __future__ import annotations

RESOURCE_FIELDNAMES: list[str] = [
    "ts_epoch",
    "elapsed_s",
    "sample_dt_s",
    "root_pid",
    "process_count",
    "new_pids",
    "gone_pids",
    "rss_kb",
    "pss_kb",
    "uss_kb",
    "vsz_kb",
    "cpu_user_s",
    "cpu_system_s",
    "cpu_total_s",
    "cpu_pct",
    "thread_count",
    "io_rchar_bytes",
    "io_wchar_bytes",
    "io_read_bytes",
    "io_write_bytes",
    "io_rchar_bps",
    "io_wchar_bps",
    "io_read_bps",
    "io_write_bps",
    "server_vram_mib",
    "total_gpu_mem_mib",
    "total_gpu_used_mib",
    "gpu_util_pct",
    "baseline_pss_kb",
    "peak_pss_delta_kb",
    "baseline_vram_mib",
    "peak_vram_delta_mib",
    "observed_hardware_profile",
    "audio_seconds",
    "wall_seconds",
    "transcription_throughput",
    "time_to_first_delta_s",
    "sampling_warning",
]
"""Stable Resource Schema columns written to every Resource CSV row."""
