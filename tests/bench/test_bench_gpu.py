"""Tests for per-process GPU VRAM attribution in bench.gpu."""

from __future__ import annotations

from coro.bench import gpu
from coro.bench.gpu import GpuDevice, _aggregate, sample_gpu
from coro.bench.models.gpu import GpuSample

MIB = 1024**2


def _dev(total_mib, used_mib, util, procs):
    """Build a raw :class:`GpuDevice` like _read_devices returns.

    ``procs`` is a list of ``(pid, used_mib_or_None)`` tuples.
    """
    return GpuDevice(
        mem_total=total_mib * MIB,
        mem_used=used_mib * MIB,
        util=util,
        procs=[(pid, None if mem is None else mem * MIB) for pid, mem in procs],
    )


class TestAggregate:
    def test_attributes_vram_to_server_tree_pids(self):
        dev = _dev(16000, 5000, 80.0, [(100, 1200), (200, 3000)])
        out = _aggregate([dev], pids={100})
        assert out.server_vram_mib == 1200.0
        assert out.total_gpu_used_mib == 5000.0
        assert out.total_gpu_mem_mib == 16000.0
        assert out.gpu_util_pct == 80.0

    def test_sums_tree_pids_across_devices(self):
        d0 = _dev(16000, 4000, 50.0, [(100, 1000), (200, 500)])
        d1 = _dev(16000, 2000, 70.0, [(100, 800), (300, 2000)])
        out = _aggregate([d0, d1], pids={100, 200})
        assert out.server_vram_mib == 2300.0  # 1000 + 500 + 800
        assert out.total_gpu_used_mib == 6000.0
        assert out.gpu_util_pct == 60.0

    def test_cpu_only_when_tree_uses_no_gpu(self):
        # Driver / other tenant holds 366 MiB but no server PID is present.
        dev = _dev(16000, 366, 0.0, [(999, 366)])
        out = _aggregate([dev], pids={100, 200})
        assert out.server_vram_mib == 0.0

    def test_ignores_unavailable_process_memory(self):
        dev = _dev(16000, 5000, 10.0, [(100, None), (100, 700)])
        out = _aggregate([dev], pids={100})
        assert out.server_vram_mib == 700.0

    def test_without_pids_reports_whole_device(self):
        dev = _dev(16000, 5000, 10.0, [(100, 1200)])
        out = _aggregate([dev], pids=None)
        assert out.server_vram_mib == 5000.0


class TestSampleGpu:
    def test_empty_when_nvml_unavailable(self, monkeypatch):
        monkeypatch.setattr(gpu, "_read_devices", lambda: None)
        assert sample_gpu({1, 2}) == GpuSample()

    def test_empty_when_no_devices(self, monkeypatch):
        monkeypatch.setattr(gpu, "_read_devices", lambda: [])
        assert sample_gpu({1}).server_vram_mib == ""

    def test_attributes_via_read_devices(self, monkeypatch):
        dev = _dev(16000, 5000, 25.0, [(7, 2048), (8, 1000)])
        monkeypatch.setattr(gpu, "_read_devices", lambda: [dev])
        out = sample_gpu({7})
        assert out.server_vram_mib == 2048.0
        assert out.total_gpu_used_mib == 5000.0
        assert out.gpu_util_pct == 25.0
