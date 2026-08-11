"""Benchmark package: Resource CSV schema and subcommand CLI."""

from __future__ import annotations

import sys
from contextlib import contextmanager
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


@contextmanager
def _stubbed_main(subcommand: str, runner: str, handle):
    """Run `coro-bench <subcommand>` with AMI IO and the server handle stubbed out."""
    with (
        patch.object(sys, "argv", ["coro-bench", subcommand]),
        patch("coro.bench.cli.ensure_audio_and_annotations"),
        patch("coro.bench.cli.materialize_reference_stms"),
        patch("coro.bench.cli.build_server_handle", return_value=handle),
        patch(f"coro.bench.cli.{runner}") as mock_runner,
    ):
        yield mock_runner


def test_main_quality_calls_run_quality(stub_server_handle):
    from coro.bench.cli import main

    with _stubbed_main("quality", "_run_quality", stub_server_handle) as mock_quality:
        main()
    mock_quality.assert_called_once()


def test_main_performance_runs_and_outputs_summary(stub_server_handle):
    from coro.bench.cli import main

    with _stubbed_main("performance", "_run_performance", stub_server_handle) as mock_perf:
        main()
    mock_perf.assert_called_once()


def test_main_all_calls_run_all(stub_server_handle):
    from coro.bench.cli import main

    with _stubbed_main("all", "_run_all", stub_server_handle) as mock_all:
        main()
    mock_all.assert_called_once()


def test_main_enters_and_exits_the_server_handle(stub_server_handle):
    """The server handle's lifecycle brackets the workload, teardown included."""
    from coro.bench.cli import main

    with _stubbed_main("all", "_run_all", stub_server_handle):
        main()
    stub_server_handle.__enter__.assert_called_once()
    stub_server_handle.__exit__.assert_called_once()


def test_main_passes_the_server_handle_to_the_runner(stub_server_handle):
    """The runner receives the live handle, not a URL rebuilt from flag defaults."""
    from coro.bench.cli import main

    with _stubbed_main("all", "_run_all", stub_server_handle) as mock_all:
        main()
    assert mock_all.call_args.args[2] is stub_server_handle


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
