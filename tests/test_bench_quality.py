"""Tests for Quality Benchmark: MeetEval Metric Set and run-level summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


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
    def test_score_item_returns_metrics_with_all_five(self, tmp_path: Path):
        ref_stm = tmp_path / "meeting1.ref.stm"
        ref_stm.write_text("meeting1 1 A 0.000 1.500 hello world\n")
        hyp_stm = tmp_path / "meeting1.hyp.stm"
        hyp_stm.write_text("meeting1 1 A 0.000 1.500 hello world\n")

        mock_meeteval = _make_mock_meeteval()
        with patch.dict(sys.modules, {"meeteval": mock_meeteval}):
            from asr_diar_server.bench.quality import score_item

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
            from asr_diar_server.bench.quality import score_item

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
            from asr_diar_server.bench.quality import score_item

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
            from asr_diar_server.bench.quality import score_item

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
            from asr_diar_server.bench.quality import combine_items

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
            from asr_diar_server.bench.quality import combine_items

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
