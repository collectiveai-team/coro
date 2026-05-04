"""Benchmark package: Resource CSV schema and subcommand CLI."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

from asr_diar_server.bench import RESOURCE_FIELDNAMES
from asr_diar_server.bench.cli import parse_args


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


def test_parse_args_no_audio_positional():
    args = parse_args(["quality"])
    assert not hasattr(args, "audio")


def test_main_quality_prints_not_implemented(capsys):
    from asr_diar_server.bench.cli import main

    with patch.object(sys, "argv", ["asr-diar-bench", "quality"]), \
         patch("asr_diar_server.bench.cli.ensure_audio_and_annotations"), \
         patch("asr_diar_server.bench.cli.materialize_reference_stms"):
        main()
    captured = capsys.readouterr()
    assert "quality not yet implemented" in captured.out


def test_main_performance_prints_not_implemented(capsys):
    from asr_diar_server.bench.cli import main

    with patch.object(sys, "argv", ["asr-diar-bench", "performance"]), \
         patch("asr_diar_server.bench.cli.ensure_audio_and_annotations"), \
         patch("asr_diar_server.bench.cli.materialize_reference_stms"):
        main()
    captured = capsys.readouterr()
    assert "performance not yet implemented" in captured.out


def test_main_all_prints_not_implemented(capsys):
    from asr_diar_server.bench.cli import main

    with patch.object(sys, "argv", ["asr-diar-bench", "all"]), \
         patch("asr_diar_server.bench.cli.ensure_audio_and_annotations"), \
         patch("asr_diar_server.bench.cli.materialize_reference_stms"):
        main()
    captured = capsys.readouterr()
    assert "all not yet implemented" in captured.out


def test_legacy_tool_files_deleted():
    assert not os.path.exists("tools/bench_asr.py")
    assert not os.path.exists("tools/whisperx_to_rttm.py")
