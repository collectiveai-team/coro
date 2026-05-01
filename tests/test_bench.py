"""Cycle 12: benchmark package migration — Resource CSV schema and CLI arg parsing.

Tests verify that:
- RESOURCE_FIELDNAMES contains all expected Stable Resource Schema columns.
- The parse_args function accepts the expected arguments without error.
- The ``asr-diar-bench`` entry point is importable via asr_diar_server.bench.cli.

No real benchmark runs are executed.
"""

from __future__ import annotations

import pytest

from asr_diar_server.bench import RESOURCE_FIELDNAMES
from asr_diar_server.bench.cli import parse_args


# ---------------------------------------------------------------------------
# Stable Resource Schema
# ---------------------------------------------------------------------------

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
    "wer",
    "der",
    "der_collar_s",
    "der_skip_overlap",
    "wer_normalization",
    "sampling_warning",
}


def test_resource_fieldnames_contains_required_schema():
    """RESOURCE_FIELDNAMES preserves the Stable Resource Schema columns."""
    assert _REQUIRED_FIELDS.issubset(set(RESOURCE_FIELDNAMES))


def test_resource_fieldnames_is_list():
    """RESOURCE_FIELDNAMES is a list (preserves column ordering)."""
    assert isinstance(RESOURCE_FIELDNAMES, list)


def test_resource_fieldnames_no_duplicates():
    """RESOURCE_FIELDNAMES has no duplicate column names."""
    assert len(RESOURCE_FIELDNAMES) == len(set(RESOURCE_FIELDNAMES))


# ---------------------------------------------------------------------------
# CLI arg parsing
# ---------------------------------------------------------------------------


def test_parse_args_requires_audio(tmp_path):
    """parse_args exits cleanly when required audio argument is missing."""
    import sys
    from unittest.mock import patch

    with patch.object(sys, "argv", ["asr-diar-bench"]), pytest.raises(SystemExit):
        parse_args()


def test_parse_args_accepts_audio_path(tmp_path):
    """parse_args accepts a valid audio path and returns a Namespace."""
    import sys
    from unittest.mock import patch

    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"\x00" * 100)

    with patch.object(sys, "argv", ["asr-diar-bench", str(audio_file)]):
        args = parse_args()
    assert str(args.audio) == str(audio_file)


def test_parse_args_default_url_targets_packaged_server():
    """Default --url points to the packaged server endpoint (not a legacy path)."""
    import sys
    from unittest.mock import patch

    with patch.object(sys, "argv", ["asr-diar-bench", "/fake/audio.wav"]):
        args = parse_args()
    # The URL should reference /v1/audio/transcriptions (or /v2/...)
    assert "audio/transcriptions" in args.url
