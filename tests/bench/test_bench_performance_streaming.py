"""Tests for streaming performance runs: TTFT columns, summaries and the manifest."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from support.bench_http import (
    StubHandler,
    mock_sample_fn,
    placeholder_audio,
    results_dir,
    serve,
)
from support.network import LOOPBACK

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


class _SSEStubHandler(StubHandler):
    def send_transcription(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(SSE_RESPONSE_BODY)
        self.wfile.flush()


@pytest.fixture()
def sse_stub_server() -> Iterator[str]:
    with serve(_SSEStubHandler) as url:
        yield url


class TestStreamingPerformanceRun:
    pytestmark = LOOPBACK

    def test_ttft_column_non_empty_in_streaming_run(self, sse_stub_server, tmp_path: Path):
        from coro.bench.orchestrate import run_performance_workload

        audio = placeholder_audio(tmp_path, "meeting1")
        out_dir = results_dir(tmp_path)
        items = [{"item_id": "meeting1", "audio_path": audio, "ref_stm_path": None}]

        run_performance_workload(
            items=items,
            base_url=sse_stub_server,
            out_dir=out_dir,
            reps=1,
            server_pid=1,
            sample_fn=mock_sample_fn,
            sample_interval=0.05,
            stream=True,
        )

        csv_path = out_dir / "performance" / "resource_meeting1_rep1.csv"
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        ttft_values = [float(r["time_to_first_delta_s"]) for r in rows]
        assert len(ttft_values) == len(rows) >= 1
        assert all(v >= 0 for v in ttft_values)

    def test_ttft_column_empty_in_non_streaming_run(self, stub_server, tmp_path: Path):
        from coro.bench.orchestrate import run_performance_workload

        audio = placeholder_audio(tmp_path, "meeting1")
        out_dir = results_dir(tmp_path)
        items = [{"item_id": "meeting1", "audio_path": audio, "ref_stm_path": None}]

        run_performance_workload(
            items=items,
            base_url=stub_server,
            out_dir=out_dir,
            reps=1,
            server_pid=1,
            sample_fn=mock_sample_fn,
            sample_interval=0.05,
            stream=False,
        )

        csv_path = out_dir / "performance" / "resource_meeting1_rep1.csv"
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert {r["root_pid"] for r in rows} == {"1"}
        assert {r["time_to_first_delta_s"] for r in rows} == {""}

    def test_summary_includes_ttft_aggregates_when_streaming(self, sse_stub_server, tmp_path: Path):
        from coro.bench.orchestrate import run_performance_workload

        audio = placeholder_audio(tmp_path, "meeting1")
        out_dir = results_dir(tmp_path)
        items = [{"item_id": "meeting1", "audio_path": audio, "ref_stm_path": None}]

        run_performance_workload(
            items=items,
            base_url=sse_stub_server,
            out_dir=out_dir,
            reps=2,
            server_pid=1,
            sample_fn=mock_sample_fn,
            sample_interval=0.05,
            stream=True,
        )

        summary = json.loads((out_dir / "performance" / "summary.json").read_text())
        m1_agg = summary["per_item_aggregation"]["meeting1"]
        assert "time_to_first_delta_s" in m1_agg, f"Expected TTFT aggregates, got: {list(m1_agg)}"
        ttft_agg = m1_agg["time_to_first_delta_s"]
        for key in ("median", "min", "max", "mean", "stddev"):
            assert key in ttft_agg, f"Missing {key} in TTFT aggregates"

    def test_summary_excludes_ttft_when_not_streaming(self, stub_server, tmp_path: Path):
        from coro.bench.orchestrate import run_performance_workload

        audio = placeholder_audio(tmp_path, "meeting1")
        out_dir = results_dir(tmp_path)
        items = [{"item_id": "meeting1", "audio_path": audio, "ref_stm_path": None}]

        run_performance_workload(
            items=items,
            base_url=stub_server,
            out_dir=out_dir,
            reps=1,
            server_pid=1,
            sample_fn=mock_sample_fn,
            sample_interval=0.05,
            stream=False,
        )

        summary = json.loads((out_dir / "performance" / "summary.json").read_text())
        m1_agg = summary["per_item_aggregation"]["meeting1"]
        assert m1_agg["time_to_first_delta_s"] is None

    def test_manifest_records_stream_true(self, sse_stub_server, tmp_path: Path):
        from coro.bench.orchestrate import run_performance_workload

        audio = placeholder_audio(tmp_path, "meeting1")
        out_dir = results_dir(tmp_path)
        items = [{"item_id": "meeting1", "audio_path": audio, "ref_stm_path": None}]

        run_performance_workload(
            items=items,
            base_url=sse_stub_server,
            out_dir=out_dir,
            reps=1,
            server_pid=1,
            sample_fn=mock_sample_fn,
            sample_interval=0.05,
            stream=True,
        )

        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["stream"] is True

    def test_manifest_records_stream_false(self, stub_server, tmp_path: Path):
        from coro.bench.orchestrate import run_performance_workload

        audio = placeholder_audio(tmp_path, "meeting1")
        out_dir = results_dir(tmp_path)
        items = [{"item_id": "meeting1", "audio_path": audio, "ref_stm_path": None}]

        run_performance_workload(
            items=items,
            base_url=stub_server,
            out_dir=out_dir,
            reps=1,
            server_pid=1,
            sample_fn=mock_sample_fn,
            sample_interval=0.05,
            stream=False,
        )

        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["stream"] is False

    def test_manifest_records_a_forced_language(self, stub_server, tmp_path: Path):
        from coro.bench.orchestrate import run_performance_workload

        audio = placeholder_audio(tmp_path, "meeting1")
        out_dir = results_dir(tmp_path)
        items = [{"item_id": "meeting1", "audio_path": audio, "ref_stm_path": None}]

        run_performance_workload(
            items=items,
            base_url=stub_server,
            out_dir=out_dir,
            reps=1,
            server_pid=1,
            sample_fn=mock_sample_fn,
            sample_interval=0.05,
            language="es-US",
        )

        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["language"] == "es-US"

    def test_manifest_records_none_for_auto_language_detection(self, stub_server, tmp_path: Path):
        from coro.bench.orchestrate import run_performance_workload

        audio = placeholder_audio(tmp_path, "meeting1")
        out_dir = results_dir(tmp_path)
        items = [{"item_id": "meeting1", "audio_path": audio, "ref_stm_path": None}]

        run_performance_workload(
            items=items,
            base_url=stub_server,
            out_dir=out_dir,
            reps=1,
            server_pid=1,
            sample_fn=mock_sample_fn,
            sample_interval=0.05,
        )

        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["language"] is None
