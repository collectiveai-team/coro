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
        assert "bench = [" in text
        assert "meeteval" in text
        assert "rich" in text


class TestRequireMeeteval:
    def test_require_meeteval_exits_with_helpful_message(self):
        with patch.dict(sys.modules, {"meeteval": None}):
            from asr_diar_server.bench.quality import _require_meeteval

            with pytest.raises(SystemExit) as exc_info:
                _require_meeteval()
            assert exc_info.value.code == 1

    def test_require_meeteval_returns_module_when_available(self):
        mock_meeteval = MagicMock()
        with patch.dict(sys.modules, {"meeteval": mock_meeteval}):
            from asr_diar_server.bench.quality import _require_meeteval

            result = _require_meeteval()
            assert result is mock_meeteval


def _write_stm_pair(tmp_path: Path, name: str, ref: str, hyp: str) -> tuple[Path, Path]:
    """Write a (ref, hyp) STM pair and return their paths."""
    ref_stm = tmp_path / f"{name}.ref.stm"
    hyp_stm = tmp_path / f"{name}.hyp.stm"
    ref_stm.write_text(ref)
    hyp_stm.write_text(hyp)
    return ref_stm, hyp_stm


class TestScoreItem:
    def test_normalize_transcript_text_removes_punctuation_and_extra_spaces(self):
        from asr_diar_server.bench.quality import _normalize_transcript_text

        assert _normalize_transcript_text("Hello,   world!!") == "Hello world"

    def test_write_normalized_stm_preserves_metadata_and_normalizes_text(
        self, tmp_path: Path,
    ):
        from asr_diar_server.bench.quality import _write_normalized_stm

        src = tmp_path / "in.stm"
        dst = tmp_path / "out.stm"
        src.write_text("meeting 1 A 0.000 1.500 Hello,   world!!\n")

        _write_normalized_stm(src, dst)

        assert dst.read_text() == "meeting 1 A 0.000 1.500 Hello world\n"

    def test_score_item_returns_all_wer_and_der_metrics(self, tmp_path: Path):
        ref_stm, hyp_stm = _write_stm_pair(
            tmp_path,
            "meeting1",
            "meeting1 1 A 0.000 1.500 hello world\n",
            "meeting1 1 A 0.000 1.500 hello world\n",
        )

        from asr_diar_server.bench.quality import score_item

        result = score_item(ref_stm, hyp_stm)

        assert result["metrics"] is not None
        metrics = result["metrics"]
        assert "cpwer" in metrics
        assert "orcwer" in metrics
        assert "dicpwer" in metrics
        assert "der" in metrics
        # Perfect match -> zero WER on every speaker-attributed metric.
        assert metrics["cpwer"]["wer"] == 0.0
        assert metrics["orcwer"]["wer"] == 0.0
        assert metrics["dicpwer"]["wer"] == 0.0

    def test_score_item_wer_metrics_have_full_breakdown(self, tmp_path: Path):
        ref_stm, hyp_stm = _write_stm_pair(
            tmp_path, "m", "m 1 A 0.0 1.0 test\n", "m 1 A 0.0 1.0 test\n",
        )

        from asr_diar_server.bench.quality import score_item

        result = score_item(ref_stm, hyp_stm)

        cpwer = result["metrics"]["cpwer"]
        for key in ("wer", "errors", "length", "insertions", "deletions", "substitutions"):
            assert key in cpwer

    def test_score_item_der_has_full_breakdown(self, tmp_path: Path):
        ref_stm, hyp_stm = _write_stm_pair(
            tmp_path, "m", "m 1 A 0.0 1.0 test\n", "m 1 A 0.0 1.0 test\n",
        )

        from asr_diar_server.bench.quality import score_item

        result = score_item(ref_stm, hyp_stm)

        der = result["metrics"]["der"]
        for key in ("der", "false_alarm", "missed_detection", "speaker_error", "total_speech"):
            assert key in der

    def test_score_item_returns_error_when_meeteval_raises(self, tmp_path: Path):
        ref_stm, hyp_stm = _write_stm_pair(
            tmp_path, "m", "m 1 A 0.0 1.0 test\n", "m 1 A 0.0 1.0 test\n",
        )

        mock_meeteval = MagicMock()
        mock_meeteval.wer.cpwer.side_effect = RuntimeError("scoring failed")
        with patch.dict(sys.modules, {"meeteval": mock_meeteval}):
            from asr_diar_server.bench.quality import score_item

            result = score_item(ref_stm, hyp_stm)

        assert result["metrics"] is None
        assert "error" in result
        assert result["error"]["type"] == "RuntimeError"
        assert "scoring failed" in result["error"]["message"]

    def test_score_item_reports_diarization_sanity(self, tmp_path: Path):
        # Single hyp speaker against a two-speaker reference is degenerate.
        ref_stm, hyp_stm = _write_stm_pair(
            tmp_path,
            "m",
            "m 1 A 0.0 2.0 hello world\nm 1 B 2.0 4.0 foo bar\n",
            "m 1 1 0.0 4.0 hello world foo bar\n",
        )

        from asr_diar_server.bench.quality import score_item

        result = score_item(ref_stm, hyp_stm)

        diar = result["diarization"]
        assert diar["ref_speakers"] == 2
        assert diar["hyp_speakers"] == 1
        assert diar["degenerate"] is True


def _scored(tmp_path: Path, name: str, ref: str, hyp: str, seconds: float) -> dict:
    """Score a tiny STM pair with real meeteval and tag it like the orchestrator."""
    ref_stm, hyp_stm = _write_stm_pair(tmp_path, name, ref, hyp)
    from asr_diar_server.bench.quality import score_item

    result = score_item(ref_stm, hyp_stm)
    result["session_id"] = name
    result["audio_seconds"] = seconds
    return result


class TestCombineItems:
    def test_combine_items_produces_combined_metrics(self, tmp_path: Path):
        from asr_diar_server.bench.quality import combine_items

        item_results = [
            _scored(tmp_path, "A", "A 1 X 0.0 2.0 hello world\n", "A 1 X 0.0 2.0 hello world\n", 2.0),
            _scored(tmp_path, "B", "B 1 X 0.0 2.0 foo bar\n", "B 1 X 0.0 2.0 foo bar\n", 2.0),
        ]

        summary = combine_items(item_results)

        assert summary["n_succeeded"] == 2
        assert summary["n_failed"] == 0
        assert "combined" in summary
        assert summary["combined"]["cpwer"] is not None
        assert summary["combined"]["der"] is not None
        assert summary["per_item"][0]["session_id"] == "A"
        assert summary["per_item"][1]["session_id"] == "B"

    def test_combine_items_aggregates_der_across_all_items(self, tmp_path: Path):
        """Regression: combined DER must reflect every item, not just the first.

        Item A is a perfect single-speaker match (DER 0.0); item B collapses two
        reference speakers onto one hypothesis speaker (DER > 0). The old code
        reported only item A's DER (0.0); the aggregate must be > 0.
        """
        from asr_diar_server.bench.quality import combine_items

        item_results = [
            _scored(tmp_path, "A", "A 1 X 0.0 2.0 hello world\n", "A 1 X 0.0 2.0 hello world\n", 2.0),
            _scored(
                tmp_path,
                "B",
                "B 1 P 0.0 2.0 foo bar\nB 1 Q 2.0 4.0 baz qux\n",
                "B 1 1 0.0 4.0 foo bar baz qux\n",
                4.0,
            ),
        ]

        summary = combine_items(item_results)

        first_item_der = item_results[0]["metrics"]["der"]["der"]
        assert first_item_der == 0.0
        assert summary["combined"]["der"]["der"] > 0.0
        assert summary["n_degenerate_diarization"] == 1

    def test_combine_items_counts_failures(self, tmp_path: Path):
        from asr_diar_server.bench.quality import combine_items

        good = _scored(tmp_path, "A", "A 1 X 0.0 2.0 hello\n", "A 1 X 0.0 2.0 hello\n", 2.0)
        item_results = [
            good,
            {"session_id": "B", "metrics": None, "error": {"type": "RuntimeError", "message": "fail"}},
        ]

        summary = combine_items(item_results)

        assert summary["n_succeeded"] == 1
        assert summary["n_failed"] == 1
        assert len(summary["per_item"]) == 2
        assert summary["per_item"][1]["session_id"] == "B"


class TestCLIFlags:
    def test_der_collar_flag_accepted(self):
        from asr_diar_server.bench.cli import parse_args

        args = parse_args(["quality", "--der-collar", "0.25"])
        assert args.der_collar == 0.25

    def test_der_regions_flag_accepted(self):
        from asr_diar_server.bench.cli import parse_args

        args = parse_args(["quality", "--der-regions", "nooverlap"])
        assert args.der_regions == "nooverlap"

    def test_der_collar_default_is_zero(self):
        from asr_diar_server.bench.cli import parse_args

        args = parse_args(["quality"])
        assert args.der_collar == 0.0

    def test_der_regions_default_is_all(self):
        from asr_diar_server.bench.cli import parse_args

        args = parse_args(["quality"])
        assert args.der_regions == "all"

    def test_der_regions_invalid_rejected(self):
        from asr_diar_server.bench.cli import parse_args

        with pytest.raises(SystemExit):
            parse_args(["quality", "--der-regions", "invalid"])


class TestQualityRun:
    def test_quality_run_produces_artifacts(self, e2e_server, tmp_path: Path):
        from asr_diar_server.bench.orchestrate import run_workload

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
        assert "cpwer" in item_data["metrics"]
        assert "der" in item_data["metrics"]

        summary = json.loads((quality_dir / "summary.json").read_text())
        assert summary["n_succeeded"] == 1
        assert summary["n_failed"] == 0
        assert "combined" in summary
        assert "per_item" in summary
        assert summary["per_item"][0]["session_id"] == "meeting1"

    def test_quality_run_isolates_failures(self, e2e_server, tmp_path: Path):
        from asr_diar_server.bench.orchestrate import run_workload

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

        # Force the second item's scoring to fail while the first succeeds,
        # mimicking score_item's own error-result contract.
        from asr_diar_server.bench import quality as quality_mod

        real_score_item = quality_mod.score_item
        call_count = [0]

        def fail_on_second(ref_path, hyp_path, **kwargs):
            call_count[0] += 1
            if call_count[0] > 1:
                return {
                    "metrics": None,
                    "error": {"type": "RuntimeError", "message": "scoring failed for meeting2"},
                }
            return real_score_item(ref_path, hyp_path, **kwargs)

        with patch.object(quality_mod, "score_item", side_effect=fail_on_second):
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
