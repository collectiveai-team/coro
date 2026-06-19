"""Background resource sampler wrapping sample_process_tree."""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from coro.bench.gpu import sample_gpu
from coro.bench.models.resource import ProcessTreeSample, ResourceBaseline
from coro.bench.schema import RESOURCE_FIELDNAMES

SampleFn = Callable[[int], ProcessTreeSample]


def _default_sample_fn(root_pid: int) -> ProcessTreeSample:
    from coro.bench.run import sample_process_tree

    return sample_process_tree(root_pid)


def sample_resource_baseline(
    pid: int,
    sample_fn: SampleFn | None = None,
) -> ResourceBaseline:
    """Capture process/GPU memory after warmup for prediction-memory deltas."""
    raw = (sample_fn or _default_sample_fn)(pid)
    gpu = sample_gpu(raw.pids)
    return ResourceBaseline(
        baseline_pss_kb=raw.pss_kb,
        baseline_vram_mib=gpu.server_vram_mib,
    )


class Sampler:
    def __init__(
        self,
        pid: int,
        interval: float = 0.25,
        sample_fn: SampleFn | None = None,
    ) -> None:
        self.pid = pid
        self.interval = interval
        self._sample_fn = sample_fn or _default_sample_fn
        self.samples: list[dict[str, Any]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time: float = 0.0
        self._prev_sample: dict[str, Any] | None = None
        self._prev_pids: set[int] = set()
        self._sampling_warning: bool = False

    def start(self) -> None:
        self._start_time = time.monotonic()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def backfill(self, **fields: Any) -> None:
        for row in self.samples:
            for key, value in fields.items():
                if key in RESOURCE_FIELDNAMES:
                    row[key] = value

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESOURCE_FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.samples)

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            ts = time.time()
            elapsed = time.monotonic() - self._start_time
            raw = self._sample_fn(self.pid)

            new_pids = raw.pids - self._prev_pids
            gone_pids = self._prev_pids - raw.pids
            self._prev_pids.update(raw.pids)

            cpu_pct = 0.0
            sample_dt = self.interval
            if self._prev_sample is not None:
                dt = elapsed - float(self._prev_sample.get("elapsed_s", elapsed))
                if dt > 0:
                    du = raw.cpu_user_s - self._prev_sample.get("cpu_user_s", 0.0)
                    ds = raw.cpu_system_s - self._prev_sample.get("cpu_system_s", 0.0)
                    cpu_pct = 100.0 * (du + ds) / dt
                    sample_dt = dt
                    if dt > self.interval * 2:
                        self._sampling_warning = True

            gpu = sample_gpu(raw.pids)
            server_vram = gpu.server_vram_mib
            profile = (
                "cpu+gpu"
                if server_vram not in ("", None) and float(server_vram) > 0
                else "cpu-only"
            )

            row: dict[str, Any] = {
                "ts_epoch": round(ts, 3),
                "elapsed_s": round(elapsed, 3),
                "sample_dt_s": round(sample_dt, 3),
                "root_pid": self.pid,
                "process_count": len(raw.pids),
                "new_pids": len(new_pids),
                "gone_pids": len(gone_pids),
                "rss_kb": raw.rss_kb,
                "pss_kb": raw.pss_kb,
                "uss_kb": raw.uss_kb,
                "vsz_kb": raw.vsz_kb,
                "cpu_user_s": round(raw.cpu_user_s, 3),
                "cpu_system_s": round(raw.cpu_system_s, 3),
                "cpu_total_s": round(raw.cpu_user_s + raw.cpu_system_s, 3),
                "cpu_pct": round(cpu_pct, 2),
                "thread_count": raw.thread_count,
                "io_rchar_bytes": raw.rchar,
                "io_wchar_bytes": raw.wchar,
                "io_read_bytes": raw.read_bytes,
                "io_write_bytes": raw.write_bytes,
                "io_rchar_bps": 0.0,
                "io_wchar_bps": 0.0,
                "io_read_bps": 0.0,
                "io_write_bps": 0.0,
                **asdict(gpu),
                "baseline_pss_kb": "",
                "peak_pss_delta_kb": "",
                "baseline_vram_mib": "",
                "peak_vram_delta_mib": "",
                "observed_hardware_profile": profile,
                "audio_seconds": "",
                "wall_seconds": "",
                "transcription_throughput": "",
                "time_to_first_delta_s": "",
                "sampling_warning": "",
            }

            if self._prev_sample is not None:
                dt = float(row["sample_dt_s"])
                if dt > 0:
                    row["io_rchar_bps"] = round(
                        (raw.rchar - self._prev_sample.get("io_rchar_bytes", 0)) / dt, 1
                    )
                    row["io_wchar_bps"] = round(
                        (raw.wchar - self._prev_sample.get("io_wchar_bytes", 0)) / dt, 1
                    )
                    row["io_read_bps"] = round(
                        (raw.read_bytes - self._prev_sample.get("io_read_bytes", 0)) / dt, 1
                    )
                    row["io_write_bps"] = round(
                        (raw.write_bytes - self._prev_sample.get("io_write_bytes", 0)) / dt, 1
                    )

            self._prev_sample = {
                "elapsed_s": elapsed,
                "cpu_user_s": raw.cpu_user_s,
                "cpu_system_s": raw.cpu_system_s,
                "io_rchar_bytes": raw.rchar,
                "io_wchar_bytes": raw.wchar,
                "io_read_bytes": raw.read_bytes,
                "io_write_bytes": raw.write_bytes,
            }
            self.samples.append(row)
            self._stop_event.wait(self.interval)
