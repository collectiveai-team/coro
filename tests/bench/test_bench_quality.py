"""Tests for Quality Benchmark: MeetEval Metric Set and run-level summary."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from coro.bench.models.quality import ScoreError, ScoreResult


def _write_stm_pair(tmp_path: Path, name: str, ref: str, hyp: str) -> tuple[Path, Path]:
    """Write a (ref, hyp) STM pair and return their paths."""
    ref_stm = tmp_path / f"{name}.ref.stm"
    hyp_stm = tmp_path / f"{name}.hyp.stm"
    ref_stm.write_text(ref)
    hyp_stm.write_text(hyp)
    return ref_stm, hyp_stm


class TestScoreItem:
    def test_normalized_lane_applies_identically_to_ref_and_hyp(self, tmp_path: Path):
        """Casing/punctuation-only differences vanish in the Normalized Metric Lane."""
        from coro.bench.quality import score_item

        ref_stm, hyp_stm = _write_stm_pair(
            tmp_path,
            "sesion",
            "sesion 1 A 0.000 1.500 ¿Está el señor Ibáñez?\n",
            "sesion 1 A 0.000 1.500 Está el señor Ibáñez\n",
        )

        result = score_item(ref_stm, hyp_stm)

        assert result.metrics is not None
        assert result.metrics.normalized is not None
        assert result.metrics.normalized.cpwer is not None
        assert result.metrics.normalized.cpwer.wer == 0.0

    def test_dropped_diacritic_still_counts_as_an_error(self, tmp_path: Path):
        """The whole point of preserving diacritics: `esta` != `está`."""
        from coro.bench.quality import score_item

        ref_stm, hyp_stm = _write_stm_pair(
            tmp_path,
            "sesion",
            "sesion 1 A 0.000 1.500 está el año pasado\n",
            "sesion 1 A 0.000 1.500 esta el ano pasado\n",
        )

        result = score_item(ref_stm, hyp_stm)

        assert result.metrics is not None
        assert result.metrics.normalized is not None
        assert result.metrics.normalized.cpwer is not None
        assert result.metrics.normalized.cpwer.wer > 0.0

    def test_normalized_lane_skipped_when_nothing_survives_normalization(
        self,
        tmp_path: Path,
        caplog,
    ):
        """A degenerate normalized lane must not take the Raw Metric Lane down."""
        from coro.bench.quality import score_item

        ref_stm, hyp_stm = _write_stm_pair(
            tmp_path,
            "sesion",
            "sesion 1 A 0.000 1.500 hola mundo\n",
            "sesion 1 A 0.000 1.500 [tos]\n",
        )

        result = score_item(ref_stm, hyp_stm)

        assert result.metrics is not None
        assert result.metrics.cpwer is not None
        assert result.metrics.der is not None
        assert result.metrics.normalized is None
        assert "normalized lane skipped" in caplog.text

    def test_both_lanes_reported_per_item(self, tmp_path: Path):
        from coro.bench.quality import score_item

        ref_stm, hyp_stm = _write_stm_pair(
            tmp_path,
            "sesion",
            "sesion 1 A 0.000 1.500 Hola, mundo.\n",
            "sesion 1 A 0.000 1.500 hola mundo\n",
        )

        result = score_item(ref_stm, hyp_stm)

        assert result.metrics is not None
        # Raw Metric Lane sees the punctuation/casing difference...
        assert result.metrics.cpwer is not None
        assert result.metrics.cpwer.wer > 0.0
        # ...the Normalized Metric Lane does not.
        assert result.metrics.normalized is not None
        assert result.metrics.normalized.cpwer is not None
        assert result.metrics.normalized.cpwer.wer == 0.0

    def test_score_item_returns_all_wer_and_der_metrics(self, tmp_path: Path):
        ref_stm, hyp_stm = _write_stm_pair(
            tmp_path,
            "meeting1",
            "meeting1 1 A 0.000 1.500 hello world\n",
            "meeting1 1 A 0.000 1.500 hello world\n",
        )

        from coro.bench.quality import score_item

        result = score_item(ref_stm, hyp_stm)

        assert result.metrics is not None
        metrics = result.metrics
        assert metrics.cpwer is not None
        assert metrics.orcwer is not None
        assert metrics.dicpwer is not None
        assert metrics.der is not None
        # Perfect match -> zero WER on every speaker-attributed metric.
        assert metrics.cpwer.wer == 0.0
        assert metrics.orcwer.wer == 0.0
        assert metrics.dicpwer.wer == 0.0

    def test_score_item_wer_metrics_have_full_breakdown(self, tmp_path: Path):
        ref_stm, hyp_stm = _write_stm_pair(
            tmp_path,
            "m",
            "m 1 A 0.0 1.0 test\n",
            "m 1 A 0.0 1.0 test\n",
        )

        from coro.bench.quality import score_item

        result = score_item(ref_stm, hyp_stm)

        assert result.metrics is not None and result.metrics.cpwer is not None
        cpwer = result.metrics.cpwer
        for key in ("wer", "errors", "length", "insertions", "deletions", "substitutions"):
            assert hasattr(cpwer, key)

    def test_score_item_der_has_full_breakdown(self, tmp_path: Path):
        ref_stm, hyp_stm = _write_stm_pair(
            tmp_path,
            "m",
            "m 1 A 0.0 1.0 test\n",
            "m 1 A 0.0 1.0 test\n",
        )

        from coro.bench.quality import score_item

        result = score_item(ref_stm, hyp_stm)

        assert result.metrics is not None and result.metrics.der is not None
        der = result.metrics.der
        for key in ("der", "false_alarm", "missed_detection", "speaker_error", "total_speech"):
            assert hasattr(der, key)

    def test_score_item_returns_error_when_meeteval_raises(self, tmp_path: Path):
        ref_stm, hyp_stm = _write_stm_pair(
            tmp_path,
            "m",
            "m 1 A 0.0 1.0 test\n",
            "m 1 A 0.0 1.0 test\n",
        )

        mock_meeteval = MagicMock()
        mock_meeteval.wer.cpwer.side_effect = RuntimeError("scoring failed")
        with patch.dict(sys.modules, {"meeteval": mock_meeteval}):
            from coro.bench.quality import score_item

            result = score_item(ref_stm, hyp_stm)

        assert result.metrics is None
        assert result.error is not None
        assert result.error.type == "RuntimeError"
        assert "scoring failed" in result.error.message

    def test_score_item_reports_diarization_sanity(self, tmp_path: Path):
        # Single hyp speaker against a two-speaker reference is degenerate.
        ref_stm, hyp_stm = _write_stm_pair(
            tmp_path,
            "m",
            "m 1 A 0.0 2.0 hello world\nm 1 B 2.0 4.0 foo bar\n",
            "m 1 1 0.0 4.0 hello world foo bar\n",
        )

        from coro.bench.quality import score_item

        result = score_item(ref_stm, hyp_stm)

        diar = result.diarization
        assert diar is not None
        assert diar.ref_speakers == 2
        assert diar.hyp_speakers == 1
        assert diar.degenerate is True


def _scored(tmp_path: Path, name: str, ref: str, hyp: str, seconds: float) -> ScoreResult:
    """Score a tiny STM pair with real meeteval and tag it like the orchestrator."""
    ref_stm, hyp_stm = _write_stm_pair(tmp_path, name, ref, hyp)
    from coro.bench.quality import score_item

    result = score_item(ref_stm, hyp_stm)
    result.session_id = name
    result.audio_seconds = seconds
    return result


class TestCombineItems:
    def test_combine_items_produces_combined_metrics(self, tmp_path: Path):
        from coro.bench.quality import combine_items

        item_results = [
            _scored(
                tmp_path, "A", "A 1 X 0.0 2.0 hello world\n", "A 1 X 0.0 2.0 hello world\n", 2.0
            ),
            _scored(tmp_path, "B", "B 1 X 0.0 2.0 foo bar\n", "B 1 X 0.0 2.0 foo bar\n", 2.0),
        ]

        summary = combine_items(item_results)

        assert summary.n_succeeded == 2
        assert summary.n_failed == 0
        assert summary.combined is not None
        assert summary.combined.cpwer is not None
        assert summary.combined.der is not None
        assert summary.per_item[0].session_id == "A"
        assert summary.per_item[1].session_id == "B"

    def test_combine_items_aggregates_der_across_all_items(self, tmp_path: Path):
        """Regression: combined DER must reflect every item, not just the first.

        Item A is a perfect single-speaker match (DER 0.0); item B collapses two
        reference speakers onto one hypothesis speaker (DER > 0). The old code
        reported only item A's DER (0.0); the aggregate must be > 0.
        """
        from coro.bench.quality import combine_items

        item_results = [
            _scored(
                tmp_path, "A", "A 1 X 0.0 2.0 hello world\n", "A 1 X 0.0 2.0 hello world\n", 2.0
            ),
            _scored(
                tmp_path,
                "B",
                "B 1 P 0.0 2.0 foo bar\nB 1 Q 2.0 4.0 baz qux\n",
                "B 1 1 0.0 4.0 foo bar baz qux\n",
                4.0,
            ),
        ]

        summary = combine_items(item_results)

        assert item_results[0].metrics is not None and item_results[0].metrics.der is not None
        first_item_der = item_results[0].metrics.der.der
        assert first_item_der == 0.0
        assert summary.combined is not None and summary.combined.der is not None
        assert summary.combined.der.der > 0.0
        assert summary.n_degenerate_diarization == 1

    def test_combine_items_reports_both_lanes_in_the_aggregate(self, tmp_path: Path):
        """Run-level aggregate carries the Raw and Normalized Metric Lanes."""
        from coro.bench.quality import combine_items

        item_results = [
            _scored(
                tmp_path,
                "A",
                "A 1 X 0.0 2.0 Hola, mundo.\n",
                "A 1 X 0.0 2.0 hola mundo\n",
                2.0,
            ),
            _scored(
                tmp_path,
                "B",
                "B 1 X 0.0 2.0 ¿Está el señor?\n",
                "B 1 X 0.0 2.0 Está el señor\n",
                2.0,
            ),
        ]

        summary = combine_items(item_results)

        assert summary.combined is not None
        combined = summary.combined
        # Raw Metric Lane sees the punctuation and casing differences.
        assert combined.cpwer is not None
        assert combined.cpwer.wer > 0.0
        assert combined.orcwer is not None
        assert combined.dicpwer is not None
        # Normalized Metric Lane does not.
        assert combined.normalized is not None
        assert combined.normalized.cpwer is not None
        assert combined.normalized.cpwer.wer == 0.0
        assert combined.normalized.orcwer is not None
        assert combined.normalized.dicpwer is not None
        # ...and both lanes are present on every per-item row.
        assert [entry.session_id for entry in summary.per_item] == ["A", "B"]
        assert [entry.cpwer is not None for entry in summary.per_item] == [True, True]
        assert [entry.normalized_cpwer is not None for entry in summary.per_item] == [True, True]

    def test_combine_items_counts_failures(self, tmp_path: Path):
        from coro.bench.quality import combine_items

        good = _scored(tmp_path, "A", "A 1 X 0.0 2.0 hello\n", "A 1 X 0.0 2.0 hello\n", 2.0)
        item_results = [
            good,
            ScoreResult(
                session_id="B",
                metrics=None,
                error=ScoreError(type="RuntimeError", message="fail"),
            ),
        ]

        summary = combine_items(item_results)

        assert summary.n_succeeded == 1
        assert summary.n_failed == 1
        assert len(summary.per_item) == 2
        assert summary.per_item[1].session_id == "B"


class TestDiarizationOnly:
    """Diarization-only references (VoxConverse-style) score DER but omit WER."""

    _REF = "rec 1 spkA 0.000 2.000 <sd>\nrec 1 spkB 2.500 4.000 <sd>\n"
    _HYP = "rec 1 0 0.100 2.100 hola mundo\nrec 1 1 2.600 3.900 que tal\n"

    def test_score_item_skips_wer_keeps_der(self, tmp_path: Path):
        from coro.bench.quality import score_item

        ref, hyp = _write_stm_pair(tmp_path, "rec", self._REF, self._HYP)
        result = score_item(ref, hyp)

        assert result.diarization_only is True
        assert result.metrics is not None
        # Only DER is computed for diarization-only references.
        assert result.metrics.der is not None
        assert result.metrics.cpwer is None
        assert result.metrics.orcwer is None
        assert result.metrics.dicpwer is None
        assert result.metrics.normalized is None
        assert result.metrics.der.der >= 0.0

    def test_combine_items_aggregates_der_without_wer(self, tmp_path: Path):
        from coro.bench.quality import combine_items

        result = _scored(tmp_path, "rec", self._REF, self._HYP, 4.0)
        summary = combine_items([result])

        assert summary.n_succeeded == 1
        assert summary.n_failed == 0
        assert summary.combined is not None
        assert summary.combined.cpwer is None
        assert summary.combined.der is not None
        entry = summary.per_item[0]
        assert entry.diarization_only is True
        assert entry.cpwer is None
        assert entry.der is not None


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
