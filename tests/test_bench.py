"""Benchmark package: Resource CSV schema and subcommand CLI."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from coro.bench import RESOURCE_FIELDNAMES
from coro.bench.cli import parse_args


_LEGACY_QUALITY_FIELDS = {
    "wer",
    "der",
    "der_collar_s",
    "der_skip_overlap",
    "wer_normalization",
}

_REQUIRED_FIELDS = {
    "ts_epoch",
    "elapsed_s",
    "sample_dt_s",
    "root_pid",
    "process_count",
    "rss_kb",
    "pss_kb",
    "uss_kb",
    "vsz_kb",
    "cpu_pct",
    "io_rchar_bytes",
    "io_wchar_bytes",
    "io_read_bytes",
    "io_write_bytes",
    "io_rchar_bps",
    "io_wchar_bps",
    "io_read_bps",
    "io_write_bps",
    "server_vram_mib",
    "observed_hardware_profile",
    "audio_seconds",
    "wall_seconds",
    "transcription_throughput",
    "sampling_warning",
    "time_to_first_delta_s",
}


def test_resource_fieldnames_excludes_legacy_quality():
    """RESOURCE_FIELDNAMES no longer contains legacy quality columns."""
    for field in _LEGACY_QUALITY_FIELDS:
        assert field not in RESOURCE_FIELDNAMES, f"Legacy field {field!r} still present"


def test_resource_fieldnames_contains_required_schema():
    """RESOURCE_FIELDNAMES preserves the required resource columns."""
    assert _REQUIRED_FIELDS.issubset(set(RESOURCE_FIELDNAMES))


def test_resource_fieldnames_is_list():
    assert isinstance(RESOURCE_FIELDNAMES, list)


def test_resource_fieldnames_no_duplicates():
    assert len(RESOURCE_FIELDNAMES) == len(set(RESOURCE_FIELDNAMES))


def test_parse_args_accepts_quality():
    args = parse_args(["quality"])
    assert args.subcommand == "quality"


def test_parse_args_accepts_performance():
    args = parse_args(["performance"])
    assert args.subcommand == "performance"


def test_parse_args_accepts_all():
    args = parse_args(["all"])
    assert args.subcommand == "all"


def test_parse_args_rejects_unknown_subcommand():
    with pytest.raises(SystemExit):
        parse_args(["foobar"])


def test_parse_args_requires_subcommand():
    with pytest.raises(SystemExit):
        parse_args([])


def test_parse_args_audio_defaults_none():
    args = parse_args(["quality"])
    assert args.audio is None


def test_main_quality_calls_run_quality(capsys):
    from coro.bench.cli import main

    with (
        patch.object(sys, "argv", ["coro-bench", "quality"]),
        patch("coro.bench.cli.ensure_audio_and_annotations"),
        patch("coro.bench.cli.materialize_reference_stms"),
        patch("coro.bench.cli._run_quality") as mock_quality,
    ):
        main()
    mock_quality.assert_called_once()


def test_main_performance_runs_and_outputs_summary(capsys):
    from coro.bench.cli import main

    with (
        patch.object(sys, "argv", ["coro-bench", "performance"]),
        patch("coro.bench.cli.ensure_audio_and_annotations"),
        patch("coro.bench.cli.materialize_reference_stms"),
        patch("coro.bench.cli._run_performance") as mock_perf,
    ):
        main()
    mock_perf.assert_called_once()


def test_main_all_calls_run_all(capsys):
    from coro.bench.cli import main

    with (
        patch.object(sys, "argv", ["coro-bench", "all"]),
        patch("coro.bench.cli.ensure_audio_and_annotations"),
        patch("coro.bench.cli.materialize_reference_stms"),
        patch("coro.bench.cli._run_all") as mock_all,
    ):
        main()
    mock_all.assert_called_once()


def test_parse_args_accepts_warmup():
    args = parse_args(["all", "--warmup"])
    assert args.warmup is True


def test_parse_args_warmup_defaults_false():
    args = parse_args(["all"])
    assert args.warmup is False


def test_parse_args_warmup_audio_implies_warmup(tmp_path):
    audio = tmp_path / "warmup.wav"
    audio.touch()
    args = parse_args(["all", "--warmup-audio", str(audio)])
    assert args.warmup is True
    assert args.warmup_audio == audio


def test_legacy_tool_files_deleted():
    assert not Path("tools/bench_asr.py").exists()
    assert not Path("tools/whisperx_to_rttm.py").exists()


def test_parse_args_stream_rejected_on_quality():
    with pytest.raises(SystemExit) as exc_info:
        parse_args(["quality", "--stream"])
    assert exc_info.value.code != 0


def test_parse_args_stream_accepted_on_performance():
    args = parse_args(["performance", "--stream"])
    assert args.stream is True


def test_parse_args_stream_accepted_on_all():
    args = parse_args(["all", "--stream"])
    assert args.stream is True


def test_parse_args_stream_defaults_false_on_performance():
    args = parse_args(["performance"])
    assert args.stream is False


def test_parse_args_stream_defaults_false_on_all():
    args = parse_args(["all"])
    assert args.stream is False
