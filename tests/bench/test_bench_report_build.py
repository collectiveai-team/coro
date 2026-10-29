"""Tests for coro.bench.report.build_report: reading a run directory into a report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coro.bench.report import build_report
from coro.bench.report_markdown import (
    NORMALIZED_LANE_LABEL,
    RAW_LANE_LABEL,
    render_markdown,
)


def test_build_report_diarization_only_item_shows_der_not_error(tmp_path: Path):
    """A diarization-only item (no WER) renders DER without an ERROR footnote."""
    der = {
        "der": 0.0943,
        "false_alarm": 0.1,
        "missed_detection": 0.1,
        "speaker_error": 0.0,
        "total_speech": 5.3,
    }
    summary = {
        "workload_set": ["voxc_clip"],
        "n_succeeded": 1,
        "n_failed": 0,
        "n_degenerate_diarization": 0,
        "combined": {
            "cpwer": None,
            "orcwer": None,
            "dicpwer": None,
            "normalized": {"cpwer": None, "orcwer": None, "dicpwer": None},
            "der": der,
        },
        "per_item": [
            {
                "session_id": "voxc_clip",
                "audio_seconds": 6.0,
                "diarization_only": True,
                "der": 0.0943,
                "diarization": {"ref_speakers": 2, "hyp_speakers": 2, "degenerate": False},
            }
        ],
    }
    quality_dir = tmp_path / "quality"
    quality_dir.mkdir()
    (quality_dir / "summary.json").write_text(json.dumps(summary))

    md = render_markdown(build_report(tmp_path))

    assert "ERROR for voxc_clip" not in md
    assert "0.0943" in md
    assert "| voxc_clip | 6.0 | - | - | - | 0.0943 |" in md


_FULL_MANIFEST = {
    "timestamp": "2026-05-04T10:00:00+00:00",
    "git_sha": "deadbeef",
    "subcommand": "quality",
    "reps": 1,
    "workload_set": [
        {"item_id": "IB4001", "audio_path": "/data/IB4001.wav", "ref_stm_path": None},
    ],
    "server_health": {
        "startup_selection": {
            "asr_backend": "whisperlivekit",
            "asr_model": "openai/whisper-medium",
            "diar_backend": "whisperlivekit",
            "diar_model": "nvidia/diar_streaming_sortformer_4spk-v2",
            "pipeline": "full-memory",
        },
    },
    "versions": {"coro": "0.1.0"},
    "cli_args": ["quality", "--ami-meetings", "IB4001"],
    "hostname": "testhost",
}

_FULL_QUALITY_SUMMARY = {
    "workload_set": ["IB4001"],
    "n_succeeded": 1,
    "n_failed": 0,
    "combined": {
        "siwer": {"wer": 0.12, "errors": 10, "length": 83},
        "cpwer": {"wer": 0.15, "errors": 12, "length": 80},
        "orcwer": {"wer": 0.13, "errors": 11, "length": 84},
        "dicpwer": {"wer": 0.14, "errors": 11, "length": 78},
        "der": {
            "der": 0.08,
            "false_alarm": 0.01,
            "missed_detection": 0.03,
            "speaker_error": 0.04,
            "total_speech": 100.0,
        },
    },
    "per_item": [
        {
            "session_id": "IB4001",
            "siwer": 0.12,
            "cpwer": 0.15,
            "orcwer": 0.13,
            "dicpwer": 0.14,
            "der": 0.08,
            "audio_seconds": 1837.4,
        },
    ],
}


def _write_run(run_dir: Path, manifest: dict, quality_summary: dict) -> None:
    """Write a run directory holding ``manifest`` and a quality ``summary.json``."""
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    quality_dir = run_dir / "quality"
    quality_dir.mkdir()
    (quality_dir / "summary.json").write_text(json.dumps(quality_summary))


def test_build_report_reads_manifest_and_summaries(tmp_path):
    _write_run(tmp_path, _FULL_MANIFEST, _FULL_QUALITY_SUMMARY)

    report = build_report(tmp_path)

    assert report.subcommand == "quality"
    assert report.git_sha == "deadbeef"
    assert len(report.quality_rows) == 1
    assert report.quality_rows[0].session_id == "IB4001"
    assert report.quality_combined is not None
    assert report.quality_combined.session_id == "COMBINED"


_LANES_MANIFEST = {
    "timestamp": "2026-05-04T10:00:00+00:00",
    "git_sha": "deadbeef",
    "subcommand": "quality",
    "workload_set": [
        {"item_id": "IB4001", "audio_path": "/data/IB4001.wav"},
    ],
    "server_health": {"startup_selection": {}},
}

_LANES_QUALITY_SUMMARY = {
    "combined": {
        "cpwer": {"wer": 0.30},
        "orcwer": {"wer": 0.25},
        "dicpwer": {"wer": 0.20},
        "der": {"der": 0.10},
        "normalized": {
            "cpwer": {"wer": 0.21},
            "orcwer": {"wer": 0.16},
            "dicpwer": {"wer": 0.11},
        },
    },
    "per_item": [
        {
            "session_id": "IB4001",
            "audio_seconds": 10.0,
            "cpwer": 0.30,
            "orcwer": 0.25,
            "dicpwer": 0.20,
            "der": 0.10,
            "normalized_cpwer": 0.21,
            "normalized_orcwer": 0.16,
            "normalized_dicpwer": 0.11,
        },
    ],
}


def test_build_report_reads_normalized_quality_summaries(tmp_path):
    _write_run(tmp_path, _LANES_MANIFEST, _LANES_QUALITY_SUMMARY)

    report = build_report(tmp_path)
    md = render_markdown(report)

    assert len(report.normalized_quality_rows) == 1
    assert report.normalized_quality_combined is not None
    assert report.normalized_quality_rows[0].cpwer == pytest.approx(0.21, abs=1e-9)
    assert "## Quality Results — Raw Metric Lane" in md
    assert "## Quality Results — Normalized Metric Lane" in md
    assert "| IB4001 | 10.0 | 0.2100 | 0.1600 | 0.1100 |" in md


def test_rendered_report_labels_each_metric_lane(tmp_path):
    """Every quality figure must be attributable to the lane that produced it."""
    manifest = {
        "timestamp": "2026-05-04T10:00:00+00:00",
        "git_sha": "deadbeef",
        "subcommand": "quality",
        "workload_set": [{"item_id": "IB4001", "audio_path": "/data/IB4001.wav"}],
        "server_health": {"startup_selection": {}},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    quality_dir = tmp_path / "quality"
    quality_dir.mkdir()
    (quality_dir / "summary.json").write_text(
        json.dumps(
            {
                "combined": {
                    "cpwer": {"wer": 0.30},
                    "normalized": {"cpwer": {"wer": 0.21}},
                },
                "per_item": [
                    {
                        "session_id": "IB4001",
                        "audio_seconds": 10.0,
                        "cpwer": 0.30,
                        "normalized_cpwer": 0.21,
                    }
                ],
            }
        )
    )

    md = render_markdown(build_report(tmp_path))

    assert RAW_LANE_LABEL in md
    assert NORMALIZED_LANE_LABEL in md
    # The normalized lane states its protocol, including diacritic preservation.
    assert "diacritics preserved" in md
    assert "verbatim" in md
    # The legacy, locally-invented protocol is no longer advertised.
    assert "removing punctuation and collapsing whitespace" not in md


def test_build_report_surfaces_degenerate_diarization_warning(tmp_path):
    manifest = {
        "timestamp": "2026-05-04T10:00:00+00:00",
        "git_sha": "deadbeef",
        "subcommand": "quality",
        "workload_set": [{"item_id": "IB4001", "audio_path": "/data/IB4001.wav"}],
        "server_health": {"startup_selection": {}},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    quality_dir = tmp_path / "quality"
    quality_dir.mkdir()
    quality_summary = {
        "n_succeeded": 1,
        "n_failed": 0,
        "n_degenerate_diarization": 1,
        "combined": {
            "cpwer": {"wer": 0.30},
            "orcwer": {"wer": 0.05},
            "dicpwer": {"wer": 0.20},
            "der": {"der": 0.55},
        },
        "per_item": [
            {
                "session_id": "IB4001",
                "audio_seconds": 1837.4,
                "cpwer": 0.30,
                "orcwer": 0.05,
                "dicpwer": 0.20,
                "der": 0.55,
                "diarization": {"ref_speakers": 4, "hyp_speakers": 1, "degenerate": True},
            },
        ],
    }
    (quality_dir / "summary.json").write_text(json.dumps(quality_summary))

    report = build_report(tmp_path)
    md = render_markdown(report)

    assert any("degenerate diarization" in note for note in report.quality_footnotes)
    assert "degenerate diarization" in md
    assert "IB4001" in md


def test_build_report_reads_provider_keys_from_health(tmp_path):
    """/health uses *_provider keys; the report must still record the config."""
    manifest = {
        "subcommand": "quality",
        "server_health": {
            "startup_selection": {
                "pipeline": "full-memory",
                "asr_provider": "faster-whisper",
                "asr_model": "openai/whisper-medium",
                "diarization_provider": "nemo",
                "diarization_model": "nvidia/diar_streaming_sortformer_4spk-v2",
            },
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    report = build_report(tmp_path)

    assert report.server_config["asr_backend"] == "faster-whisper"
    assert report.server_config["diar_backend"] == "nemo"
    assert report.server_config["diar_model"] == "nvidia/diar_streaming_sortformer_4spk-v2"
    md = render_markdown(report)
    assert "faster-whisper" in md
    assert "nemo" in md
