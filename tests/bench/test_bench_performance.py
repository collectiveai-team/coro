"""Tests for the performance benchmark: Sampler, per-rep summary, aggregation, full run."""

from __future__ import annotations

import csv
import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from coro.bench.models.gpu import GpuSample
from coro.bench.models.performance import PerRepSummary
from coro.bench.models.resource import ProcessTreeSample
from coro.bench.sampling import Sampler
from coro.bench.schema import RESOURCE_FIELDNAMES
from support.bench_http import mock_sample_fn, placeholder_audio, results_dir
from support.network import LOOPBACK


_SAMPLE_WAIT_S = 5.0


def _sampled(pid: int, samples: int) -> Sampler:
    """A Sampler stopped once it has taken ``samples`` samples of the mock process tree."""
    taken = 0
    reached = threading.Event()

    def sample_fn(root_pid: int) -> ProcessTreeSample:
        nonlocal taken
        taken += 1
        if taken >= samples:
            reached.set()
        return mock_sample_fn(root_pid)

    sampler = Sampler(pid=pid, interval=0.01, sample_fn=sample_fn)
    sampler.start()
    assert reached.wait(_SAMPLE_WAIT_S)
    sampler.stop()
    return sampler


def _read_csv(path: Path) -> tuple[list[str] | None, list[dict[str, str]]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return (list(reader.fieldnames) if reader.fieldnames else None), rows


def _names_by_mtime(directory: Path, pattern: str) -> list[str]:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return [f.stem for f in files]


class TestSampler:
    def test_collects_samples_when_started(self):
        with patch("coro.bench.sampling.sample_gpu", return_value=GpuSample(), autospec=True):
            sampler = _sampled(pid=123, samples=2)

        assert len(sampler.samples) >= 2
        assert sampler.samples[0]["root_pid"] == 123
        assert sampler.samples[0]["pss_kb"] == 100123

    def test_write_csv_writes_valid_resource_csv(self, tmp_path: Path):
        sampler = _sampled(pid=123, samples=2)

        csv_path = tmp_path / "perf" / "resource_item1_rep1.csv"
        sampler.write_csv(csv_path)

        assert csv_path.exists()
        headers, rows = _read_csv(csv_path)
        assert headers is not None
        assert headers == RESOURCE_FIELDNAMES
        assert len(rows) == len(sampler.samples)
        assert rows[0]["root_pid"] == "123"
        assert "time_to_first_delta_s" in headers

    def test_backfill_sets_fields_on_all_rows(self):
        sampler = _sampled(pid=1, samples=2)

        sampler.backfill(wall_seconds=1.234, audio_seconds=5.0, transcription_throughput=4.05)

        backfilled = [
            (row["wall_seconds"], row["audio_seconds"], row["transcription_throughput"])
            for row in sampler.samples
        ]
        assert len(backfilled) >= 2
        assert backfilled == [pytest.approx((1.234, 5.0, 4.05), abs=1e-12)] * len(backfilled)


def _write_resource_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESOURCE_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _make_csv_rows(
    *,
    pss_kb: float = 100000,
    cpu_pct: float = 50.0,
    server_vram_mib: float | str = "",
    gpu_util_pct: float | str = "",
    baseline_pss_kb: float | str = "",
    baseline_vram_mib: float | str = "",
    wall_seconds: float = 2.0,
    audio_seconds: float = 10.0,
    observed_hardware_profile: str = "cpu-only",
) -> list[dict[str, Any]]:
    base: dict[str, Any] = dict.fromkeys(RESOURCE_FIELDNAMES, "")
    base["pss_kb"] = pss_kb
    base["cpu_pct"] = cpu_pct
    base["server_vram_mib"] = server_vram_mib
    base["gpu_util_pct"] = gpu_util_pct
    base["baseline_pss_kb"] = baseline_pss_kb
    base["baseline_vram_mib"] = baseline_vram_mib
    base["wall_seconds"] = wall_seconds
    base["audio_seconds"] = audio_seconds
    base["observed_hardware_profile"] = observed_hardware_profile
    base["transcription_throughput"] = (
        round(audio_seconds / wall_seconds, 6) if wall_seconds > 0 else ""
    )
    base["ts_epoch"] = 1000.0
    base["elapsed_s"] = 0.5
    base["sample_dt_s"] = 0.25
    base["root_pid"] = 1
    base["process_count"] = 2
    return [dict(base) for _ in range(3)]


class TestComputePerRepSummary:
    def test_extracts_peaks_from_csv(self, tmp_path: Path):
        from coro.bench.performance import compute_per_rep_summary

        rows = _make_csv_rows(pss_kb=150000, cpu_pct=75.5, wall_seconds=2.0, audio_seconds=10.0)
        csv_path = tmp_path / "resource_item1_rep1.csv"
        _write_resource_csv(csv_path, rows)

        summary = compute_per_rep_summary(csv_path)
        assert summary.observed_hardware_profile == "cpu-only"
        assert (
            summary.peak_pss_kb,
            summary.peak_cpu_pct,
            summary.wall_seconds,
            summary.transcription_throughput,
            summary.audio_seconds,
        ) == pytest.approx((150000, 75.5, 2, 5, 10), abs=1e-9)

    def test_handles_multiple_rows_with_varying_peaks(self, tmp_path: Path):
        from coro.bench.performance import compute_per_rep_summary

        r1 = _make_csv_rows(pss_kb=100000, cpu_pct=30.0)
        r2 = _make_csv_rows(pss_kb=200000, cpu_pct=80.0)
        r3 = _make_csv_rows(pss_kb=150000, cpu_pct=50.0)
        all_rows = r1 + r2 + r3
        csv_path = tmp_path / "resource_item1_rep1.csv"
        _write_resource_csv(csv_path, all_rows)

        summary = compute_per_rep_summary(csv_path)
        assert (summary.peak_pss_kb, summary.peak_cpu_pct) == pytest.approx((200000, 80), abs=1e-9)

    def test_extracts_gpu_and_baseline_adjusted_memory(self, tmp_path: Path):
        from coro.bench.performance import compute_per_rep_summary

        rows = _make_csv_rows(
            pss_kb=180000,
            server_vram_mib=2048.0,
            gpu_util_pct=91.0,
            baseline_pss_kb=100000,
            baseline_vram_mib=1536.0,
        )
        csv_path = tmp_path / "resource_item1_rep1.csv"
        _write_resource_csv(csv_path, rows)

        summary = compute_per_rep_summary(csv_path)
        assert (
            summary.peak_pss_kb,
            summary.baseline_pss_kb,
            summary.peak_pss_delta_kb,
            summary.peak_vram_mib,
            summary.baseline_vram_mib,
            summary.peak_vram_delta_mib,
            summary.peak_gpu_util_pct,
        ) == pytest.approx((180000, 100000, 80000, 2048, 1536, 512, 91), abs=1e-9)


class TestAggregateAcrossReps:
    def test_computes_median_min_max_mean_stddev(self):
        from coro.bench.performance import aggregate_across_reps

        summaries = [
            PerRepSummary(peak_pss_kb=100, peak_cpu_pct=30, transcription_throughput=5.0),
            PerRepSummary(peak_pss_kb=200, peak_cpu_pct=60, transcription_throughput=10.0),
            PerRepSummary(peak_pss_kb=300, peak_cpu_pct=90, transcription_throughput=15.0),
        ]
        agg = aggregate_across_reps(summaries)
        assert agg.peak_pss_kb is not None
        assert agg.peak_pss_kb.median == 200
        assert agg.peak_pss_kb.min == 100
        assert agg.peak_pss_kb.max == 300
        assert agg.peak_pss_kb.mean == 200
        assert agg.peak_pss_kb.stddev > 0
        assert agg.peak_cpu_pct is not None and agg.peak_cpu_pct.median == 60
        assert (
            agg.transcription_throughput is not None and agg.transcription_throughput.median == 10.0
        )

    def test_single_rep(self):
        from coro.bench.performance import aggregate_across_reps

        summaries = [
            PerRepSummary(peak_pss_kb=100, peak_cpu_pct=30, transcription_throughput=5.0),
        ]
        agg = aggregate_across_reps(summaries)
        assert agg.peak_pss_kb is not None
        assert agg.peak_pss_kb.median == 100
        assert agg.peak_pss_kb.stddev == 0


_PER_REP_KEYS = {
    "wall_seconds",
    "transcription_throughput",
    "peak_pss_kb",
    "peak_pss_delta_kb",
    "peak_cpu_pct",
    "observed_hardware_profile",
}


class TestFullPerformanceRun:
    pytestmark = LOOPBACK

    def test_produces_artifacts_per_item_rep(self, stub_server, tmp_path: Path):
        from coro.bench.orchestrate import run_performance_workload

        audio1 = placeholder_audio(tmp_path, "meeting1")
        audio2 = placeholder_audio(tmp_path, "meeting2")

        out_dir = results_dir(tmp_path)

        items = [
            {"item_id": "meeting1", "audio_path": audio1, "ref_stm_path": None},
            {"item_id": "meeting2", "audio_path": audio2, "ref_stm_path": None},
        ]

        run_performance_workload(
            items=items,
            base_url=stub_server,
            out_dir=out_dir,
            reps=2,
            server_pid=1,
            sample_fn=mock_sample_fn,
            sample_interval=0.05,
        )

        perf_dir = out_dir / "performance"
        assert (perf_dir / "resource_meeting1_rep1.csv").exists()
        assert (perf_dir / "resource_meeting1_rep2.csv").exists()
        assert (perf_dir / "resource_meeting2_rep1.csv").exists()
        assert (perf_dir / "resource_meeting2_rep2.csv").exists()
        assert (perf_dir / "summary.json").exists()

    def test_summary_has_correct_structure(self, stub_server, tmp_path: Path):
        from coro.bench.orchestrate import run_performance_workload

        audio1 = placeholder_audio(tmp_path, "meeting1")
        audio2 = placeholder_audio(tmp_path, "meeting2")

        out_dir = results_dir(tmp_path)

        items = [
            {"item_id": "meeting1", "audio_path": audio1, "ref_stm_path": None},
            {"item_id": "meeting2", "audio_path": audio2, "ref_stm_path": None},
        ]

        run_performance_workload(
            items=items,
            base_url=stub_server,
            out_dir=out_dir,
            reps=2,
            server_pid=1,
            sample_fn=mock_sample_fn,
            sample_interval=0.05,
        )

        perf_dir = out_dir / "performance"
        summary = json.loads((perf_dir / "summary.json").read_text())

        assert len(summary["per_rep"]) == 4
        item_ids = [r["item_id"] for r in summary["per_rep"]]
        assert item_ids == ["meeting1", "meeting1", "meeting2", "meeting2"]

        missing = [{key for key in _PER_REP_KEYS if key not in row} for row in summary["per_rep"]]
        assert missing == [set()] * 4

        assert "meeting1" in summary["per_item_aggregation"]
        assert "meeting2" in summary["per_item_aggregation"]
        m1_agg = summary["per_item_aggregation"]["meeting1"]
        assert "peak_pss_kb" in m1_agg
        assert "median" in m1_agg["peak_pss_kb"]
        assert "min" in m1_agg["peak_pss_kb"]
        assert "max" in m1_agg["peak_pss_kb"]
        assert "mean" in m1_agg["peak_pss_kb"]
        assert "stddev" in m1_agg["peak_pss_kb"]

        assert summary["run_totals"]["workload_set_size"] == 2
        assert summary["run_totals"]["total_wall_seconds"] > 0

    def test_csv_has_no_legacy_columns(self, stub_server, tmp_path: Path):
        from coro.bench.orchestrate import run_performance_workload

        audio1 = placeholder_audio(tmp_path, "meeting1")

        out_dir = results_dir(tmp_path)

        items = [
            {"item_id": "meeting1", "audio_path": audio1, "ref_stm_path": None},
        ]

        run_performance_workload(
            items=items,
            base_url=stub_server,
            out_dir=out_dir,
            reps=1,
            server_pid=1,
            sample_fn=mock_sample_fn,
            sample_interval=0.05,
        )

        csv_path = out_dir / "performance" / "resource_meeting1_rep1.csv"
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            assert headers is not None
            assert headers == RESOURCE_FIELDNAMES
            assert "time_to_first_delta_s" in headers

    def test_reps_grouped_by_item_order(self, stub_server, tmp_path: Path):
        from coro.bench.orchestrate import run_performance_workload

        audio1 = placeholder_audio(tmp_path, "a1", size=100)
        audio2 = placeholder_audio(tmp_path, "a2", size=100)
        out_dir = results_dir(tmp_path)

        items = [
            {"item_id": "a1", "audio_path": audio1, "ref_stm_path": None},
            {"item_id": "a2", "audio_path": audio2, "ref_stm_path": None},
        ]

        run_performance_workload(
            items=items,
            base_url=stub_server,
            out_dir=out_dir,
            reps=2,
            server_pid=1,
            sample_fn=mock_sample_fn,
            sample_interval=0.05,
        )

        names = _names_by_mtime(out_dir / "performance", "resource_*.csv")
        assert names.index("resource_a1_rep1") < names.index("resource_a1_rep2")
        assert names.index("resource_a1_rep2") < names.index("resource_a2_rep1")
        assert names.index("resource_a2_rep1") < names.index("resource_a2_rep2")
