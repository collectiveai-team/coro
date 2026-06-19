"""Tests for performance benchmark: Sampler, aggregation, full run."""

from __future__ import annotations

import csv
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from coro.bench.gpu import GpuSample
from coro.bench.performance import PerRepSummary
from coro.bench.run import ProcessTreeSample
from coro.bench.schema import RESOURCE_FIELDNAMES


CANNED_RESPONSE = {
    "task": "transcribe",
    "duration": 3.5,
    "text": "hello world",
    "segments": [],
    "usage": {"type": "duration", "seconds": 4},
}

CANNED_HEALTH = {
    "status": "ok",
    "ready": True,
    "warmup_ready": True,
    "startup_selection": {},
    "capability_readiness": {"asr": True},
}


class _StubHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            body = json.dumps(CANNED_HEALTH).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/v1/audio/transcriptions":
            content_length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(content_length)
            body = json.dumps(CANNED_RESPONSE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


@pytest.fixture()
def stub_server():
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    thread.join()


def _mock_sample_fn(pid: int) -> ProcessTreeSample:
    return ProcessTreeSample(
        pids={pid, pid + 1},
        pss_kb=100000 + pid,
        uss_kb=50000,
        rss_kb=120000,
        vsz_kb=200000,
        cpu_user_s=1.5,
        cpu_system_s=0.3,
        rchar=1000000,
        wchar=500000,
        read_bytes=300000,
        write_bytes=200000,
        thread_count=8,
    )


class TestSampler:
    def test_collects_samples_when_started(self):
        from coro.bench.sampling import Sampler

        sampler = Sampler(pid=123, interval=0.05, sample_fn=_mock_sample_fn)
        with patch("coro.bench.sampling.sample_gpu", return_value=GpuSample()):
            sampler.start()
            time.sleep(0.2)
            sampler.stop()

        assert len(sampler.samples) >= 2
        assert sampler.samples[0]["root_pid"] == 123
        assert sampler.samples[0]["pss_kb"] == 100123

    def test_write_csv_writes_valid_resource_csv(self, tmp_path: Path):
        from coro.bench.sampling import Sampler

        sampler = Sampler(pid=123, interval=0.05, sample_fn=_mock_sample_fn)
        sampler.start()
        time.sleep(0.15)
        sampler.stop()

        csv_path = tmp_path / "perf" / "resource_item1_rep1.csv"
        sampler.write_csv(csv_path)

        assert csv_path.exists()
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            assert headers is not None
            assert headers == RESOURCE_FIELDNAMES
            rows = list(reader)
            assert len(rows) == len(sampler.samples)
            assert rows[0]["root_pid"] == "123"
            assert "time_to_first_delta_s" in headers

    def test_backfill_sets_fields_on_all_rows(self):
        from coro.bench.sampling import Sampler

        sampler = Sampler(pid=1, interval=0.05, sample_fn=_mock_sample_fn)
        sampler.start()
        time.sleep(0.15)
        sampler.stop()

        sampler.backfill(wall_seconds=1.234, audio_seconds=5.0, transcription_throughput=4.05)

        for row in sampler.samples:
            assert row["wall_seconds"] == 1.234
            assert row["audio_seconds"] == 5.0
            assert row["transcription_throughput"] == 4.05


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
        assert summary.peak_pss_kb == 150000.0
        assert summary.peak_cpu_pct == 75.5
        assert summary.wall_seconds == 2.0
        assert summary.transcription_throughput == 5.0
        assert summary.observed_hardware_profile == "cpu-only"
        assert summary.audio_seconds == 10.0

    def test_handles_multiple_rows_with_varying_peaks(self, tmp_path: Path):
        from coro.bench.performance import compute_per_rep_summary

        r1 = _make_csv_rows(pss_kb=100000, cpu_pct=30.0)
        r2 = _make_csv_rows(pss_kb=200000, cpu_pct=80.0)
        r3 = _make_csv_rows(pss_kb=150000, cpu_pct=50.0)
        all_rows = r1 + r2 + r3
        csv_path = tmp_path / "resource_item1_rep1.csv"
        _write_resource_csv(csv_path, all_rows)

        summary = compute_per_rep_summary(csv_path)
        assert summary.peak_pss_kb == 200000.0
        assert summary.peak_cpu_pct == 80.0

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
        assert summary.peak_pss_kb == 180000.0
        assert summary.baseline_pss_kb == 100000.0
        assert summary.peak_pss_delta_kb == 80000.0
        assert summary.peak_vram_mib == 2048.0
        assert summary.baseline_vram_mib == 1536.0
        assert summary.peak_vram_delta_mib == 512.0
        assert summary.peak_gpu_util_pct == 91.0


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


class TestFullPerformanceRun:
    def test_produces_artifacts_per_item_rep(self, stub_server, tmp_path: Path):
        from coro.bench.orchestrate import run_performance_workload

        audio1 = tmp_path / "meeting1.wav"
        audio1.write_bytes(b"RIFF" + b"\x00" * 200)
        audio2 = tmp_path / "meeting2.wav"
        audio2.write_bytes(b"RIFF" + b"\x00" * 200)

        out_dir = tmp_path / "results"
        out_dir.mkdir()

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
            sample_fn=_mock_sample_fn,
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

        audio1 = tmp_path / "meeting1.wav"
        audio1.write_bytes(b"RIFF" + b"\x00" * 200)
        audio2 = tmp_path / "meeting2.wav"
        audio2.write_bytes(b"RIFF" + b"\x00" * 200)

        out_dir = tmp_path / "results"
        out_dir.mkdir()

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
            sample_fn=_mock_sample_fn,
            sample_interval=0.05,
        )

        perf_dir = out_dir / "performance"
        summary = json.loads((perf_dir / "summary.json").read_text())

        assert len(summary["per_rep"]) == 4
        item_ids = [r["item_id"] for r in summary["per_rep"]]
        assert item_ids == ["meeting1", "meeting1", "meeting2", "meeting2"]

        for row in summary["per_rep"]:
            assert "wall_seconds" in row
            assert "transcription_throughput" in row
            assert "peak_pss_kb" in row
            assert "peak_pss_delta_kb" in row
            assert "peak_cpu_pct" in row
            assert "observed_hardware_profile" in row

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

        audio1 = tmp_path / "meeting1.wav"
        audio1.write_bytes(b"RIFF" + b"\x00" * 200)

        out_dir = tmp_path / "results"
        out_dir.mkdir()

        items = [
            {"item_id": "meeting1", "audio_path": audio1, "ref_stm_path": None},
        ]

        run_performance_workload(
            items=items,
            base_url=stub_server,
            out_dir=out_dir,
            reps=1,
            server_pid=1,
            sample_fn=_mock_sample_fn,
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

        audio1 = tmp_path / "a1.wav"
        audio1.write_bytes(b"RIFF" + b"\x00" * 100)
        audio2 = tmp_path / "a2.wav"
        audio2.write_bytes(b"RIFF" + b"\x00" * 100)

        out_dir = tmp_path / "results"
        out_dir.mkdir()

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
            sample_fn=_mock_sample_fn,
            sample_interval=0.05,
        )

        perf_dir = out_dir / "performance"
        mtimes = []
        for item_id in ["a1", "a1", "a2", "a2"]:
            rep = len(mtimes) % 2 + 1 if item_id == "a1" else len(mtimes) - 2 + 1
            p = perf_dir / f"resource_{item_id}_rep{rep}.csv"
            mtimes.append(p.stat().st_mtime)

        csv_files = sorted(
            perf_dir.glob("resource_*.csv"),
            key=lambda p: p.stat().st_mtime,
        )
        names = [f.stem for f in csv_files]
        assert names.index("resource_a1_rep1") < names.index("resource_a1_rep2")
        assert names.index("resource_a1_rep2") < names.index("resource_a2_rep1")
        assert names.index("resource_a2_rep1") < names.index("resource_a2_rep2")


SSE_SEGMENTS = [{"start": 0.0, "end": 1.5, "text": "hello world", "speaker": "A"}]

SSE_RESPONSE_BODY = (
    b"event: transcript.text.delta\r\n"
    b'data: {"type": "transcript.text.delta", "delta": "hello"}\r\n'
    b"\r\n"
    b"event: transcript.text.done\r\n"
    + b'data: {"type": "transcript.text.done", "text": '
    + json.dumps(json.dumps({"segments": SSE_SEGMENTS})).encode()
    + b"}\r\n"
    b"\r\n"
)


class _SSEStubHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            body = json.dumps(CANNED_HEALTH).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/v1/audio/transcriptions":
            content_length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(content_length)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(SSE_RESPONSE_BODY)
            self.wfile.flush()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


@pytest.fixture()
def sse_stub_server():
    server = HTTPServer(("127.0.0.1", 0), _SSEStubHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    thread.join()


class TestStreamingPerformanceRun:
    def test_ttft_column_non_empty_in_streaming_run(self, sse_stub_server, tmp_path: Path):
        from coro.bench.orchestrate import run_performance_workload

        audio = tmp_path / "meeting1.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 200)
        out_dir = tmp_path / "results"
        out_dir.mkdir()
        items = [{"item_id": "meeting1", "audio_path": audio, "ref_stm_path": None}]

        run_performance_workload(
            items=items,
            base_url=sse_stub_server,
            out_dir=out_dir,
            reps=1,
            server_pid=1,
            sample_fn=_mock_sample_fn,
            sample_interval=0.05,
            stream=True,
        )

        csv_path = out_dir / "performance" / "resource_meeting1_rep1.csv"
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) > 0
        ttft_values = [r["time_to_first_delta_s"] for r in rows]
        assert all(v != "" for v in ttft_values), f"Expected non-empty TTFT, got: {ttft_values}"

    def test_ttft_column_empty_in_non_streaming_run(self, stub_server, tmp_path: Path):
        from coro.bench.orchestrate import run_performance_workload

        audio = tmp_path / "meeting1.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 200)
        out_dir = tmp_path / "results"
        out_dir.mkdir()
        items = [{"item_id": "meeting1", "audio_path": audio, "ref_stm_path": None}]

        run_performance_workload(
            items=items,
            base_url=stub_server,
            out_dir=out_dir,
            reps=1,
            server_pid=1,
            sample_fn=_mock_sample_fn,
            sample_interval=0.05,
            stream=False,
        )

        csv_path = out_dir / "performance" / "resource_meeting1_rep1.csv"
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) > 0
        ttft_values = [r["time_to_first_delta_s"] for r in rows]
        assert all(v == "" for v in ttft_values), f"Expected empty TTFT, got: {ttft_values}"

    def test_summary_includes_ttft_aggregates_when_streaming(self, sse_stub_server, tmp_path: Path):
        from coro.bench.orchestrate import run_performance_workload

        audio = tmp_path / "meeting1.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 200)
        out_dir = tmp_path / "results"
        out_dir.mkdir()
        items = [{"item_id": "meeting1", "audio_path": audio, "ref_stm_path": None}]

        run_performance_workload(
            items=items,
            base_url=sse_stub_server,
            out_dir=out_dir,
            reps=2,
            server_pid=1,
            sample_fn=_mock_sample_fn,
            sample_interval=0.05,
            stream=True,
        )

        summary = json.loads((out_dir / "performance" / "summary.json").read_text())
        m1_agg = summary["per_item_aggregation"]["meeting1"]
        assert "time_to_first_delta_s" in m1_agg, (
            f"Expected TTFT aggregates, got: {list(m1_agg.keys())}"
        )
        ttft_agg = m1_agg["time_to_first_delta_s"]
        for key in ("median", "min", "max", "mean", "stddev"):
            assert key in ttft_agg, f"Missing {key} in TTFT aggregates"

    def test_summary_excludes_ttft_when_not_streaming(self, stub_server, tmp_path: Path):
        from coro.bench.orchestrate import run_performance_workload

        audio = tmp_path / "meeting1.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 200)
        out_dir = tmp_path / "results"
        out_dir.mkdir()
        items = [{"item_id": "meeting1", "audio_path": audio, "ref_stm_path": None}]

        run_performance_workload(
            items=items,
            base_url=stub_server,
            out_dir=out_dir,
            reps=1,
            server_pid=1,
            sample_fn=_mock_sample_fn,
            sample_interval=0.05,
            stream=False,
        )

        summary = json.loads((out_dir / "performance" / "summary.json").read_text())
        m1_agg = summary["per_item_aggregation"]["meeting1"]
        assert m1_agg["time_to_first_delta_s"] is None

    def test_manifest_records_stream_true(self, sse_stub_server, tmp_path: Path):
        from coro.bench.orchestrate import run_performance_workload

        audio = tmp_path / "meeting1.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 200)
        out_dir = tmp_path / "results"
        out_dir.mkdir()
        items = [{"item_id": "meeting1", "audio_path": audio, "ref_stm_path": None}]

        run_performance_workload(
            items=items,
            base_url=sse_stub_server,
            out_dir=out_dir,
            reps=1,
            server_pid=1,
            sample_fn=_mock_sample_fn,
            sample_interval=0.05,
            stream=True,
        )

        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["stream"] is True

    def test_manifest_records_stream_false(self, stub_server, tmp_path: Path):
        from coro.bench.orchestrate import run_performance_workload

        audio = tmp_path / "meeting1.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 200)
        out_dir = tmp_path / "results"
        out_dir.mkdir()
        items = [{"item_id": "meeting1", "audio_path": audio, "ref_stm_path": None}]

        run_performance_workload(
            items=items,
            base_url=stub_server,
            out_dir=out_dir,
            reps=1,
            server_pid=1,
            sample_fn=_mock_sample_fn,
            sample_interval=0.05,
            stream=False,
        )

        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["stream"] is False
