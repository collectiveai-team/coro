"""Tests for the Common Voice WER clip maker."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

_TSV = (
    "client_id\tpath\tsentence\tup_votes\n"
    "c1\tcommon_voice_es_1.mp3\tHola qué tal\t2\n"
    "c2\tcommon_voice_es_2.mp3\t\t1\n"  # empty sentence -> skipped
    "c3\t\tFrase sin audio\t1\n"  # empty path -> skipped
    "c4\tcommon_voice_es_3.mp3\tBuenos días a todos\t3\n"
)


def test_read_cv_rows_skips_empty_and_respects_limit(tmp_path: Path):
    from coro.bench.utils.make_common_voice_clips import read_cv_rows

    tsv = tmp_path / "test.tsv"
    tsv.write_text(_TSV, encoding="utf-8")

    rows = read_cv_rows(tsv)
    assert rows == [
        ("common_voice_es_1.mp3", "Hola qué tal"),
        ("common_voice_es_3.mp3", "Buenos días a todos"),
    ]
    assert read_cv_rows(tsv, limit=1) == [("common_voice_es_1.mp3", "Hola qué tal")]


def test_main_writes_single_speaker_stm_per_clip(tmp_path: Path):
    from coro.bench.utils import make_common_voice_clips as mod

    cv_dir = tmp_path / "es"
    (cv_dir / "clips").mkdir(parents=True)
    (cv_dir / "test.tsv").write_text(_TSV, encoding="utf-8")
    out_dir = tmp_path / "cv-clips"

    argv = [
        "make_common_voice_clips",
        "--cv-dir", str(cv_dir),
        "--split", "test",
        "--out-dir", str(out_dir),
    ]
    with patch.object(sys, "argv", argv), \
        patch.object(mod, "transcode_to_wav") as transcode, \
        patch.object(mod, "wav_duration_seconds", return_value=3.5):
        # Ensure the STM write does not fail on a missing out-dir.
        transcode.side_effect = lambda src, dst: dst.parent.mkdir(parents=True, exist_ok=True)
        mod.main()

    stm = (out_dir / "common_voice_es_1.ref.stm").read_text().strip()
    assert stm == "common_voice_es_1 1 1 0.000 3.500 Hola qué tal"
    # Two valid rows -> two references; the empty-sentence/path rows are skipped.
    assert (out_dir / "common_voice_es_3.ref.stm").exists()
    assert not (out_dir / "common_voice_es_2.ref.stm").exists()
