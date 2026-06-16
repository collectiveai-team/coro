"""Tests for Quality Benchmark: MeetEval Metric Set and run-level summary."""

from __future__ import annotations

import json
import threading
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


CANNED_DIARIZED_JSON = {
    "task": "transcribe",
    "duration": 3.5,
    "text": "hello world from test",
    "segments": [
        {"type": "transcript.text.segment", "id": "seg_001", "start": 0.0, "end": 1.5, "text": "hello world", "speaker": "SPEAKER_00"},
        {"type": "transcript.text.segment", "id": "seg_002", "start": 1.5, "end": 3.5, "text": "from test", "speaker": "SPEAKER_01"},
    ],
    "usage": {"type": "duration", "seconds": 4},
}

CANNED_HEALTH = {"status": "ok", "ready": True}


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
            body = json.dumps(CANNED_DIARIZED_JSON).encode()
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
def e2e_server():
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    thread.join()


class TestPyprojectBenchExtra:
    def test_pyproject_declares_bench_optional_extra(self):
        toml_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        text = toml_path.read_text()
        assert "[project.optional-dependencies]" in text
        assert 'bench = ["meeteval"' in text or "bench = ['meeteval'" in text
        assert "rich" in text


class TestRequireMeeteval:
    def test_require_meeteval_exits_with_helpful_message(self):
        with patch.dict(sys.modules, {"meeteval": None}):
            from coro.bench.quality import _require_meeteval

            with pytest.raises(SystemExit) as exc_info:
                _require_meeteval()
            assert exc_info.value.code == 1

    def test_require_meeteval_returns_module_when_available(self):
        mock_meeteval = MagicMock()
        with patch.dict(sys.modules, {"meeteval": mock_meeteval}):
            from coro.bench.quality import _require_meeteval

            result = _require_meeteval()
            assert result is mock_meeteval


def _make_mock_meeteval():
    mock_meeteval = MagicMock()

    def _wer_result(wer_val=0.1, errors=5, length=50, ins=2, dele=1, sub=2):
        r = MagicMock()
        r.wer = wer_val
        r.errors = errors
        r.length = length
        r.insertions = ins
        r.deletions = dele
        r.substitutions = sub
        return r

    mock_meeteval.wer.siwer.return_value = _wer_result(0.12, 6, 50, 2, 2, 2)
    mock_meeteval.wer.cpwer.return_value = _wer_result(0.15, 8, 53, 3, 3, 2)
    mock_meeteval.wer.greedy_orcwer.return_value = _wer_result(0.13, 7, 54, 2, 3, 2)
    mock_meeteval.wer.greedy_dicpwer.return_value = _wer_result(0.14, 7, 50, 2, 3, 2)

    der_result = MagicMock()
    der_result.der = 0.08
    der_result.false_alarm = 1.0
    der_result.missed_detection = 2.0
    der_result.speaker_error = 3.0
    der_result.total_speech = 50.0
    mock_meeteval.der.md_eval_22.return_value = der_result

    return mock_meeteval


class TestScoreItem:
    def test_normalize_transcript_text_removes_punctuation_and_extra_spaces(self):
        from coro.bench.quality import _normalize_transcript_text

        assert _normalize_transcript_text("Hello,   world!!") == "Hello world"

    def test_write_normalized_stm_preserves_metadata_and_normalizes_text(
        self, tmp_path: Path,
    ):
        from coro.bench.quality import _write_normalized_stm

        src = tmp_path / "in.stm"
        dst = tmp_path / "out.stm"
        src.write_text("meeting 1 A 0.000 1.500 Hello,   world!!\n")

        _write_normalized_stm(src, dst)

        assert dst.read_text() == "meeting 1 A 0.000 1.500 Hello world\n"

    def test_score_item_returns_metrics_with_all_five(self, tmp_path: Path):
        ref_stm = tmp_path / "meeting1.ref.stm"
        ref_stm.write_text("meeting1 1 A 0.000 1.500 hello world\n")
        hyp_stm = tmp_path / "meeting1.hyp.stm"
        hyp_stm.write_text("meeting1 1 A 0.000 1.500 hello world\n")

        mock_meeteval = _make_mock_meeteval()
        with patch.dict(sys.modules, {"meeteval": mock_meeteval}):
            from coro.bench.quality import score_item

            result = score_item(ref_stm, hyp_stm)

        assert result["metrics"] is not None
        metrics = result["metrics"]
        assert "siwer" in metrics
        assert "cpwer" in metrics
        assert "orcwer" in metrics
        assert "dicpwer" in metrics
        assert "der" in metrics
        assert metrics["siwer"]["wer"] == 0.12
        assert metrics["cpwer"]["wer"] == 0.15
        assert metrics["orcwer"]["wer"] == 0.13
        assert metrics["dicpwer"]["wer"] == 0.14
        assert metrics["der"]["der"] == 0.08

    def test_score_item_wer_metrics_have_full_breakdown(self, tmp_path: Path):
        ref_stm = tmp_path / "m.ref.stm"
        ref_stm.write_text("m 1 A 0.0 1.0 test\n")
        hyp_stm = tmp_path / "m.hyp.stm"
        hyp_stm.write_text("m 1 A 0.0 1.0 test\n")

        mock_meeteval = _make_mock_meeteval()
        with patch.dict(sys.modules, {"meeteval": mock_meeteval}):
            from coro.bench.quality import score_item

            result = score_item(ref_stm, hyp_stm)

        siwer = result["metrics"]["siwer"]
        for key in ("wer", "errors", "length", "insertions", "deletions", "substitutions"):
            assert key in siwer

    def test_score_item_der_has_full_breakdown(self, tmp_path: Path):
        ref_stm = tmp_path / "m.ref.stm"
        ref_stm.write_text("m 1 A 0.0 1.0 test\n")
        hyp_stm = tmp_path / "m.hyp.stm"
        hyp_stm.write_text("m 1 A 0.0 1.0 test\n")

        mock_meeteval = _make_mock_meeteval()
        with patch.dict(sys.modules, {"meeteval": mock_meeteval}):
            from coro.bench.quality import score_item

            result = score_item(ref_stm, hyp_stm)

        der = result["metrics"]["der"]
        for key in ("der", "false_alarm", "missed_detection", "speaker_error", "total_speech"):
            assert key in der

    def test_score_item_returns_error_when_meeteval_raises(self, tmp_path: Path):
        ref_stm = tmp_path / "m.ref.stm"
        ref_stm.write_text("m 1 A 0.0 1.0 test\n")
        hyp_stm = tmp_path / "m.hyp.stm"
        hyp_stm.write_text("m 1 A 0.0 1.0 test\n")

        mock_meeteval = MagicMock()
        mock_meeteval.wer.siwer.side_effect = RuntimeError("scoring failed")
        with patch.dict(sys.modules, {"meeteval": mock_meeteval}):
            from coro.bench.quality import score_item

            result = score_item(ref_stm, hyp_stm)

        assert result["metrics"] is None
        assert "error" in result
        assert result["error"]["type"] == "RuntimeError"
        assert "scoring failed" in result["error"]["message"]


class TestCombineItems:
    def test_combine_items_produces_combined_metrics(self):
        mock_meeteval = _make_mock_meeteval()

        combined_er = MagicMock()
        combined_er.wer = 0.125
        combined_er.errors = 12
        combined_er.length = 96
        combined_er.insertions = 4
        combined_er.deletions = 4
        combined_er.substitutions = 4
        mock_meeteval.wer.combine_error_rates.return_value = combined_er

        raw_mock = MagicMock()
        item_results = [
            {"session_id": "A", "metrics": {"siwer": {"wer": 0.1}, "cpwer": {"wer": 0.1}, "orcwer": {"wer": 0.1}, "dicpwer": {"wer": 0.1}, "der": {"der": 0.1}}, "_raw": {"siwer": raw_mock, "cpwer": raw_mock, "orcwer": raw_mock, "dicpwer": raw_mock, "der": raw_mock}},
            {"session_id": "B", "metrics": {"siwer": {"wer": 0.15}, "cpwer": {"wer": 0.15}, "orcwer": {"wer": 0.15}, "dicpwer": {"wer": 0.15}, "der": {"der": 0.15}}, "_raw": {"siwer": raw_mock, "cpwer": raw_mock, "orcwer": raw_mock, "dicpwer": raw_mock, "der": raw_mock}},
        ]

        with patch.dict(sys.modules, {"meeteval": mock_meeteval}):
            from coro.bench.quality import combine_items

            summary = combine_items(item_results)

        assert summary["n_succeeded"] == 2
        assert summary["n_failed"] == 0
        assert "combined" in summary
        assert summary["per_item"][0]["session_id"] == "A"
        assert summary["per_item"][1]["session_id"] == "B"

    def test_combine_items_counts_failures(self):
        mock_meeteval = _make_mock_meeteval()
        mock_meeteval.wer.combine_error_rates.return_value = MagicMock(
            wer=0.1, errors=5, length=50, insertions=2, deletions=1, substitutions=2
        )

        raw_mock = MagicMock()
        item_results = [
            {"session_id": "A", "metrics": {"siwer": {"wer": 0.1}, "cpwer": {"wer": 0.1}, "orcwer": {"wer": 0.1}, "dicpwer": {"wer": 0.1}, "der": {"der": 0.1}}, "_raw": {"siwer": raw_mock, "cpwer": raw_mock, "orcwer": raw_mock, "dicpwer": raw_mock, "der": raw_mock}},
            {"session_id": "B", "metrics": None, "error": {"type": "RuntimeError", "message": "fail"}},
        ]

        with patch.dict(sys.modules, {"meeteval": mock_meeteval}):
            from coro.bench.quality import combine_items

            summary = combine_items(item_results)

        assert summary["n_succeeded"] == 1
        assert summary["n_failed"] == 1
        assert len(summary["per_item"]) == 2
        assert summary["per_item"][1]["session_id"] == "B"


class TestCLIFlags:
    def test_der_collar_flag_accepted(self):
        from coro.bench.cli import parse_args

        args = parse_args(["quality", "--der-collar", "0.25"])
        assert args.der_collar == 0.25

    def test_der_regions_flag_accepted(self):
        from coro.bench.cli import parse_args

        args = parse_args(["quality", "--der-regions", "nooverlap"])
        assert args.der_regions == "nooverlap"

    def test_der_collar_default_is_zero(self):
        from coro.bench.cli import parse_args

        args = parse_args(["quality"])
        assert args.der_collar == 0.0

    def test_der_regions_default_is_all(self):
        from coro.bench.cli import parse_args

        args = parse_args(["quality"])
        assert args.der_regions == "all"

    def test_der_regions_invalid_rejected(self):
        from coro.bench.cli import parse_args

        with pytest.raises(SystemExit):
            parse_args(["quality", "--der-regions", "invalid"])


class TestQualityRun:
    def test_quality_run_produces_artifacts(self, e2e_server, tmp_path: Path):
        from coro.bench.orchestrate import run_workload

        audio = tmp_path / "meeting1.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 200)

        ref_stm = tmp_path / "meeting1.ref.stm"
        ref_stm.write_text("meeting1 1 SPEAKER_00 0.000 1.500 hello world\n")

        out_dir = tmp_path / "results"
        out_dir.mkdir()

        items = [
            {
                "item_id": "meeting1",
                "audio_path": audio,
                "ref_stm_path": ref_stm,
                "audio_seconds": 3.5,
            }
        ]

        mock_meeteval = _make_mock_meeteval()
        combined_er = MagicMock(
            wer=0.12, errors=6, length=50, insertions=2, deletions=2, substitutions=2
        )
        mock_meeteval.wer.combine_error_rates.return_value = combined_er

        with patch.dict(sys.modules, {"meeteval": mock_meeteval}):
            run_workload(
                items=items,
                base_url=e2e_server,
                out_dir=out_dir,
                reps=1,
                subcommand="quality",
                der_collar=0.0,
                der_regions="all",
            )

        quality_dir = out_dir / "quality"
        assert (quality_dir / "meeting1.json").exists()
        assert (quality_dir / "summary.json").exists()

        item_data = json.loads((quality_dir / "meeting1.json").read_text())
        assert item_data["session_id"] == "meeting1"
        assert item_data["audio_seconds"] == 3.5
        assert item_data["metrics"] is not None
        assert "siwer" in item_data["metrics"]
        assert "der" in item_data["metrics"]

        summary = json.loads((quality_dir / "summary.json").read_text())
        assert summary["n_succeeded"] == 1
        assert summary["n_failed"] == 0
        assert "combined" in summary
        assert "per_item" in summary
        assert summary["per_item"][0]["session_id"] == "meeting1"

    def test_quality_run_isolates_failures(self, e2e_server, tmp_path: Path):
        from coro.bench.orchestrate import run_workload

        audio1 = tmp_path / "meeting1.wav"
        audio1.write_bytes(b"RIFF" + b"\x00" * 200)
        audio2 = tmp_path / "meeting2.wav"
        audio2.write_bytes(b"RIFF" + b"\x00" * 200)

        ref1 = tmp_path / "meeting1.ref.stm"
        ref1.write_text("meeting1 1 A 0.0 1.0 hello\n")
        ref2 = tmp_path / "meeting2.ref.stm"
        ref2.write_text("meeting2 1 A 0.0 1.0 world\n")

        out_dir = tmp_path / "results"
        out_dir.mkdir()

        items = [
            {"item_id": "meeting1", "audio_path": audio1, "ref_stm_path": ref1, "audio_seconds": 1.0},
            {"item_id": "meeting2", "audio_path": audio2, "ref_stm_path": ref2, "audio_seconds": 1.0},
        ]

        mock_meeteval = _make_mock_meeteval()
        call_count = [0]
        original_siwer = mock_meeteval.wer.siwer

        def fail_on_second(ref, hyp):
            call_count[0] += 1
            if call_count[0] > 1:
                raise RuntimeError("scoring failed for meeting2")
            return original_siwer.return_value

        mock_meeteval.wer.siwer.side_effect = fail_on_second
        combined_er = MagicMock(
            wer=0.12, errors=6, length=50, insertions=2, deletions=2, substitutions=2
        )
        mock_meeteval.wer.combine_error_rates.return_value = combined_er

        with patch.dict(sys.modules, {"meeteval": mock_meeteval}):
            run_workload(
                items=items,
                base_url=e2e_server,
                out_dir=out_dir,
                reps=1,
                subcommand="quality",
                der_collar=0.0,
                der_regions="all",
            )

        quality_dir = out_dir / "quality"
        item2 = json.loads((quality_dir / "meeting2.json").read_text())
        assert item2["metrics"] is None
        assert "error" in item2

        summary = json.loads((quality_dir / "summary.json").read_text())
        assert summary["n_succeeded"] == 1
        assert summary["n_failed"] == 1
