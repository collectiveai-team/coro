"""Tests for published-WER calibration of the Spanish Workload Set."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coro.bench import calibration


def _write_item(quality_dir: Path, item_id: str, errors: int, length: int) -> None:
    quality_dir.mkdir(parents=True, exist_ok=True)
    (quality_dir / f"{item_id}.json").write_text(
        json.dumps(
            {
                "session_id": item_id,
                "audio_seconds": 10.0,
                "metrics": {
                    "normalized": {
                        "orcwer": {
                            "wer": errors / length,
                            "errors": errors,
                            "length": length,
                            "insertions": 0,
                            "deletions": 0,
                            "substitutions": errors,
                        }
                    }
                },
            }
        )
    )


def _write_manifest(out_dir: Path, model_id: str | None) -> None:
    health = {"startup_selection": {"asr_model": model_id}} if model_id else {}
    (out_dir / "manifest.json").write_text(json.dumps({"server_health": health}))


class TestFindTarget:
    def test_whisper_medium_has_published_spanish_figures(self):
        fleurs = calibration.find_target("openai/whisper-medium", "fleurs")
        mls = calibration.find_target("openai/whisper-medium", "mls")

        assert fleurs is not None and fleurs.published_wer == 0.036
        assert mls is not None and mls.published_wer == 0.053
        assert "arxiv.org" in fleurs.source_url

    def test_parakeet_v3_has_published_spanish_figures(self):
        target = calibration.find_target("nvidia/parakeet-tdt-0.6b-v3", "fleurs")

        assert target is not None
        assert target.published_wer == pytest.approx(0.0345, abs=1e-12)

    def test_lookup_is_case_insensitive(self):
        assert calibration.find_target("OpenAI/Whisper-Medium", "mls") is not None

    def test_unknown_model_has_no_target(self):
        assert calibration.find_target("openai/whisper-large-v3-turbo", "fleurs") is None

    def test_no_target_for_the_primary_corpus(self):
        assert calibration.find_target("openai/whisper-medium", "voxpopuli") is None


class TestCalibrateRun:
    def test_passes_within_the_margin(self, tmp_path: Path):
        _write_item(tmp_path / "quality", "fleurs-1", errors=5, length=100)
        _write_manifest(tmp_path, "openai/whisper-medium")

        report = calibration.calibrate_run(tmp_path, model_id="openai/whisper-medium")

        assert report.failed is False
        outcome = report.outcomes[0]
        assert outcome.corpus == "fleurs"
        assert outcome.status == "pass"
        assert outcome.scored_wer == pytest.approx(0.05, abs=1e-12)
        assert outcome.published_wer == pytest.approx(0.036, abs=1e-12)

    def test_fails_loudly_beyond_the_margin(self, tmp_path: Path):
        _write_item(tmp_path / "quality", "fleurs-1", errors=60, length=100)

        report = calibration.calibrate_run(tmp_path, model_id="openai/whisper-medium")

        assert report.failed is True
        assert report.outcomes[0].status == "fail"
        assert "harness fault" in report.outcomes[0].detail

    def test_fails_when_scored_far_below_published(self, tmp_path: Path):
        _write_item(tmp_path / "quality", "mls-1", errors=0, length=100)

        report = calibration.calibrate_run(
            tmp_path,
            model_id="openai/whisper-tiny",
            margin=0.05,
        )

        assert report.failed is True
        assert report.outcomes[0].deviation is not None
        assert report.outcomes[0].deviation < 0

    def test_aggregates_errors_and_length_across_items(self, tmp_path: Path):
        _write_item(tmp_path / "quality", "mls-1", errors=1, length=10)
        _write_item(tmp_path / "quality", "mls-2", errors=9, length=90)

        report = calibration.calibrate_run(tmp_path, model_id="openai/whisper-medium")

        assert report.outcomes[0].n_items == 2
        assert report.outcomes[0].scored_wer == pytest.approx(0.1, abs=1e-12)

    def test_unregistered_model_does_not_fail_the_run(self, tmp_path: Path):
        _write_item(tmp_path / "quality", "fleurs-1", errors=90, length=100)

        report = calibration.calibrate_run(tmp_path, model_id="openai/whisper-large-v3-turbo")

        assert report.failed is False
        assert report.outcomes[0].status == "unregistered"

    def test_ami_items_are_ignored(self, tmp_path: Path):
        _write_item(tmp_path / "quality", "IB4001", errors=90, length=100)

        report = calibration.calibrate_run(tmp_path, model_id="openai/whisper-medium")

        assert report.outcomes == []

    def test_summary_and_calibration_artifacts_are_skipped(self, tmp_path: Path):
        quality = tmp_path / "quality"
        _write_item(quality, "mls-1", errors=5, length=100)
        (quality / "summary.json").write_text(json.dumps({"session_id": "mls-x"}))
        (quality / "calibration.json").write_text(json.dumps({"session_id": "mls-y"}))

        report = calibration.calibrate_run(tmp_path, model_id="openai/whisper-medium")

        assert report.outcomes[0].n_items == 1

    def test_zero_length_reference_reports_no_score(self, tmp_path: Path):
        quality = tmp_path / "quality"
        quality.mkdir(parents=True)
        (quality / "mls-1.json").write_text(
            json.dumps(
                {
                    "session_id": "mls-1",
                    "metrics": {"normalized": {"orcwer": {"errors": 0, "length": 0}}},
                }
            )
        )

        report = calibration.calibrate_run(tmp_path, model_id="openai/whisper-medium")

        assert report.outcomes[0].status == "no-score"
        assert report.failed is False

    def test_missing_quality_dir_is_not_a_failure(self, tmp_path: Path):
        report = calibration.calibrate_run(tmp_path, model_id="openai/whisper-medium")

        assert report.outcomes == []
        assert report.failed is False


class TestModelIdFromManifest:
    def test_reads_the_startup_selection(self, tmp_path: Path):
        _write_manifest(tmp_path, "openai/whisper-medium")

        assert calibration.model_id_from_manifest(tmp_path) == "openai/whisper-medium"

    def test_missing_manifest_returns_none(self, tmp_path: Path):
        assert calibration.model_id_from_manifest(tmp_path) is None

    def test_missing_health_returns_none(self, tmp_path: Path):
        _write_manifest(tmp_path, None)

        assert calibration.model_id_from_manifest(tmp_path) is None


class TestRenderCalibration:
    def test_renders_status_and_source(self, tmp_path: Path):
        _write_item(tmp_path / "quality", "fleurs-1", errors=5, length=100)
        report = calibration.calibrate_run(tmp_path, model_id="openai/whisper-medium")

        text = calibration.render_calibration(report)

        assert "PASS" in text
        assert "fleurs" in text
        assert "arxiv.org" in text

    def test_empty_report_renders_nothing(self):
        from coro.bench.models.spanish import CalibrationReport

        assert calibration.render_calibration(CalibrationReport()) == ""
