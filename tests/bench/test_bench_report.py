"""Tests for the markdown and stdout renderers of a bench report."""

from __future__ import annotations

import pytest

from coro.bench.models.report import BenchReport, PerformanceRow, QualityRow
from coro.bench.report_markdown import render_markdown
from coro.bench.report_stdout import render_stdout


def _quality_report(**kwargs) -> BenchReport:
    defaults = {
        "subcommand": "quality",
        "timestamp": "2026-05-04T10:00:00+00:00",
        "out_dir": "/tmp/bench-out",
        "git_sha": "abc1234",
        "total_wall_seconds": 120.5,
        "stream": False,
        "server_config": {
            "asr_backend": "whisperlivekit",
            "asr_model": "openai/whisper-medium",
            "diar_backend": "whisperlivekit",
            "diar_model": "nvidia/diar_streaming_sortformer_4spk-v2",
            "pipeline": "full-memory",
            "warmup": False,
        },
        "workload_set": ["IB4001", "IB4002"],
        "quality_rows": [
            QualityRow(
                session_id="IB4001",
                duration=1837.4,
                cpwer=0.15,
                orcwer=0.13,
                dicpwer=0.14,
                der=0.08,
            ),
            QualityRow(
                session_id="IB4002",
                duration=900.0,
                cpwer=0.11,
                orcwer=0.10,
                dicpwer=0.11,
                der=0.06,
            ),
        ],
        "quality_combined": QualityRow(
            session_id="COMBINED",
            duration=2737.4,
            cpwer=0.135,
            orcwer=0.12,
            dicpwer=0.13,
            der=0.073,
        ),
        "quality_footnotes": [],
        "performance_rows": [],
        "versions": {"coro": "0.1.0"},
        "cli_args": ["quality", "--ami-meetings", "IB4001", "IB4002"],
    }
    defaults.update(kwargs)
    return BenchReport(**defaults)


def _performance_report(**kwargs) -> BenchReport:
    defaults = {
        "subcommand": "performance",
        "timestamp": "2026-05-04T10:00:00+00:00",
        "out_dir": "/tmp/bench-out",
        "git_sha": "abc1234",
        "total_wall_seconds": 250.0,
        "stream": False,
        "server_config": {
            "asr_backend": "whisperlivekit",
            "asr_model": "openai/whisper-medium",
            "diar_backend": "whisperlivekit",
            "diar_model": "nvidia/diar_streaming_sortformer_4spk-v2",
            "pipeline": "full-memory",
            "warmup": False,
        },
        "workload_set": ["IB4001"],
        "quality_rows": [],
        "quality_combined": None,
        "quality_footnotes": [],
        "performance_rows": [
            PerformanceRow(
                session_id="IB4001",
                rep=1,
                duration=1837.4,
                wall_seconds=120.5,
                throughput=15.24,
                peak_pss_kb=512000.0,
                peak_pss_delta_kb=64000.0,
                peak_vram_mib=4096.0,
                peak_vram_delta_mib=512.0,
                peak_gpu_util_pct=75.0,
                peak_cpu_pct=85.3,
                observed_profile="cpu-only",
                ttft=None,
            ),
        ],
        "versions": {"coro": "0.1.0"},
        "cli_args": ["performance", "--ami-meetings", "IB4001"],
    }
    defaults.update(kwargs)
    return BenchReport(**defaults)


def test_render_markdown_quality_includes_cpwer_column():
    report = _quality_report()
    md = render_markdown(report)
    assert "cpWER" in md


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
                peak_pss_delta_kb=64000.0,
                peak_vram_mib=4096.0,
                peak_vram_delta_mib=512.0,
                peak_gpu_util_pct=75.0,
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


@pytest.fixture
def wide_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rich sizes its tables to COLUMNS; a wide terminal keeps every cell on one line."""
    monkeypatch.setenv("COLUMNS", "240")


@pytest.mark.usefixtures("wide_terminal")
def test_render_stdout_prints_header_and_quality_rows(capsys):
    render_stdout(_quality_report())

    out = capsys.readouterr().out
    assert "Git SHA   : abc1234" in out
    assert "Quality Results" in out
    assert "IB4001" in out
    assert "IB4002" in out
    assert "COMBINED" in out
    assert "Performance Results" not in out


@pytest.mark.usefixtures("wide_terminal")
def test_render_stdout_prints_performance_table(capsys):
    render_stdout(_performance_report())

    out = capsys.readouterr().out
    assert "Performance Results" in out
    assert "cpu-only" in out
    assert "Quality Results" not in out


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


def test_render_markdown_performance_includes_memory_and_gpu_columns():
    report = _performance_report()
    md = render_markdown(report)
    assert "peak VRAM" in md
    assert "pred VRAM" in md
    assert "peak GPU" in md
    assert "4096 MiB" in md
    assert "512 MiB" in md


def test_render_markdown_performance_uses_dash_for_missing_gpu():
    report = _performance_report(
        performance_rows=[
            PerformanceRow(
                session_id="IB4001",
                rep=1,
                duration=1837.4,
                wall_seconds=120.5,
                throughput=15.24,
                peak_pss_kb=512000.0,
                peak_pss_delta_kb=64000.0,
                peak_vram_mib=None,
                peak_vram_delta_mib=None,
                peak_gpu_util_pct=None,
                peak_cpu_pct=85.3,
                observed_profile="cpu-only",
                ttft=None,
            ),
        ]
    )
    md = render_markdown(report)
    assert (
        "| IB4001 | 1 | 1837.4 | 120.50 | 15.24x | 500 MB | 62 MB | - | - | - | 85.3% | cpu-only |"
        in md
    )


def test_render_markdown_all_subcommand_shows_rep_note():
    report = _quality_report(subcommand="all")
    md = render_markdown(report)
    assert "rep1" in md or "rep 1" in md or "Quality scored" in md
