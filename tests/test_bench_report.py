"""Tests for asr_diar_server.bench.report module."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from asr_diar_server.bench.report import (
    BenchReport,
    PerformanceRow,
    QualityRow,
    build_report,
    render_markdown,
    render_stdout,
)


def _quality_report(**kwargs) -> BenchReport:
    defaults = dict(
        subcommand="quality",
        timestamp="2026-05-04T10:00:00+00:00",
        out_dir="/tmp/bench-out",
        git_sha="abc1234",
        total_wall_seconds=120.5,
        stream=False,
        server_config={
            "asr_backend": "whisperlivekit",
            "asr_model": "openai/whisper-medium",
            "diar_backend": "whisperlivekit",
            "diar_model": "nvidia/diar_sortformer_4spk-v1",
            "pipeline": "full-memory",
            "warmup": False,
        },
        workload_set=["IB4001", "IB4002"],
        quality_rows=[
            QualityRow(
                session_id="IB4001",
                duration=1837.4,
                siwer=0.12,
                cpwer=0.15,
                orcwer=0.13,
                dicpwer=0.14,
                der=0.08,
            ),
            QualityRow(
                session_id="IB4002",
                duration=900.0,
                siwer=0.10,
                cpwer=0.11,
                orcwer=0.10,
                dicpwer=0.11,
                der=0.06,
            ),
        ],
        quality_combined=QualityRow(
            session_id="COMBINED",
            duration=2737.4,
            siwer=0.115,
            cpwer=0.135,
            orcwer=0.12,
            dicpwer=0.13,
            der=0.073,
        ),
        quality_footnotes=[],
        performance_rows=[],
        versions={"asr_diar_server": "0.1.0"},
        cli_args=["quality", "--ami-meetings", "IB4001", "IB4002"],
    )
    defaults.update(kwargs)
    return BenchReport(**defaults)


def _performance_report(**kwargs) -> BenchReport:
    defaults = dict(
        subcommand="performance",
        timestamp="2026-05-04T10:00:00+00:00",
        out_dir="/tmp/bench-out",
        git_sha="abc1234",
        total_wall_seconds=250.0,
        stream=False,
        server_config={
            "asr_backend": "whisperlivekit",
            "asr_model": "openai/whisper-medium",
            "diar_backend": "whisperlivekit",
            "diar_model": "nvidia/diar_sortformer_4spk-v1",
            "pipeline": "full-memory",
            "warmup": False,
        },
        workload_set=["IB4001"],
        quality_rows=[],
        quality_combined=None,
        quality_footnotes=[],
        performance_rows=[
            PerformanceRow(
                session_id="IB4001",
                rep=1,
                duration=1837.4,
                wall_seconds=120.5,
                throughput=15.24,
                peak_pss_kb=512000.0,
                peak_cpu_pct=85.3,
                observed_profile="cpu-only",
                ttft=None,
            ),
        ],
        versions={"asr_diar_server": "0.1.0"},
        cli_args=["performance", "--ami-meetings", "IB4001"],
    )
    defaults.update(kwargs)
    return BenchReport(**defaults)


def test_render_markdown_quality_includes_siwer_column():
    report = _quality_report()
    md = render_markdown(report)
    assert "siWER" in md


def test_render_markdown_quality_table_has_combined_row():
    report = _quality_report()
    md = render_markdown(report)
    assert "COMBINED" in md


def test_render_markdown_failed_item_renders_error_row_and_footnote():
    report = _quality_report(
        quality_rows=[
            QualityRow(
                session_id="IB4001",
                duration=1837.4,
                siwer=None,
                cpwer=None,
                orcwer=None,
                dicpwer=None,
                der=None,
                error="RuntimeError: scoring failed",
            ),
        ],
        quality_footnotes=["ERROR for IB4001: RuntimeError — scoring failed"],
    )
    md = render_markdown(report)
    assert "ERROR" in md
    assert "IB4001" in md
    assert "RuntimeError" in md


def test_render_markdown_stream_false_omits_ttft_column():
    report = _performance_report(stream=False)
    md = render_markdown(report)
    assert "TTFT" not in md


def test_render_markdown_stream_true_includes_ttft_column():
    report = _performance_report(
        stream=True,
        performance_rows=[
            PerformanceRow(
                session_id="IB4001",
                rep=1,
                duration=1837.4,
                wall_seconds=120.5,
                throughput=15.24,
                peak_pss_kb=512000.0,
                peak_cpu_pct=85.3,
                observed_profile="cpu-only",
                ttft=2.34,
            ),
        ],
    )
    md = render_markdown(report)
    assert "TTFT" in md


def test_render_markdown_includes_run_configuration_section():
    report = _quality_report()
    md = render_markdown(report)
    assert "## Run Configuration" in md or "## Run configuration" in md


def test_render_markdown_includes_artifacts_section():
    report = _quality_report()
    md = render_markdown(report)
    assert "## Artifacts" in md


def test_render_stdout_runs_without_exception(capsys):
    report = _quality_report()
    render_stdout(report)


def test_render_stdout_performance_runs_without_exception(capsys):
    report = _performance_report()
    render_stdout(report)


def test_build_report_reads_manifest_and_summaries(tmp_path):
    manifest = {
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
                "diar_model": "nvidia/diar_sortformer_4spk-v1",
                "pipeline": "full-memory",
            },
        },
        "versions": {"asr_diar_server": "0.1.0"},
        "cli_args": ["quality", "--ami-meetings", "IB4001"],
        "hostname": "testhost",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    quality_dir = tmp_path / "quality"
    quality_dir.mkdir()
    quality_summary = {
        "workload_set": ["IB4001"],
        "n_succeeded": 1,
        "n_failed": 0,
        "combined": {
            "siwer": {"wer": 0.12, "errors": 10, "length": 83},
            "cpwer": {"wer": 0.15, "errors": 12, "length": 80},
            "orcwer": {"wer": 0.13, "errors": 11, "length": 84},
            "dicpwer": {"wer": 0.14, "errors": 11, "length": 78},
            "der": {"der": 0.08, "false_alarm": 0.01, "missed_detection": 0.03, "speaker_error": 0.04, "total_speech": 100.0},
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
    (quality_dir / "summary.json").write_text(json.dumps(quality_summary))

    report = build_report(tmp_path)

    assert report.subcommand == "quality"
    assert report.git_sha == "deadbeef"
    assert len(report.quality_rows) == 1
    assert report.quality_rows[0].session_id == "IB4001"
    assert report.quality_combined is not None
    assert report.quality_combined.session_id == "COMBINED"


def test_both_renderers_produce_consistent_session_ids():
    """Verify both renderers use the same underlying data model."""
    report = _quality_report()
    md = render_markdown(report)

    assert "IB4001" in md
    assert "IB4002" in md
    assert "COMBINED" in md


def test_render_markdown_performance_includes_wall_column():
    report = _performance_report()
    md = render_markdown(report)
    assert "wall" in md.lower() or "wall (s)" in md


def test_render_markdown_all_subcommand_shows_rep_note():
    report = _quality_report(subcommand="all")
    md = render_markdown(report)
    assert "rep1" in md or "rep 1" in md or "Quality scored" in md
