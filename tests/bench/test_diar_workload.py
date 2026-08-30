"""Workload and reference resolution for the standalone diarization comparison."""

from __future__ import annotations

from pathlib import Path

import pytest

from coro.bench.diar_workload import (
    items_from_clips_dir,
    items_from_meetings,
    materialize_rttm_references,
    parse_clip_id,
)

RTTM = """\
SPEAKER m1 1 10.000 2.000 <NA> <NA> A <NA> <NA>
SPEAKER m1 1 305.000 4.000 <NA> <NA> B <NA> <NA>
SPEAKER m1 1 950.000 3.000 <NA> <NA> A <NA> <NA>
"""
"""Three turns: one before a [300, 900) window, one inside it, one after."""


def _clip(dir_path: Path, clip_id: str, *, with_ref: bool = True) -> None:
    (dir_path / f"{clip_id}.wav").write_bytes(b"")
    if with_ref:
        (dir_path / f"{clip_id}.ref.stm").write_text(f"{clip_id} 1 A 0.000 1.000 hi\n")


def test_parse_clip_id_splits_meeting_and_window():
    assert parse_clip_id("IB4001_300_600") == ("IB4001", (300.0, 600.0))


def test_parse_clip_id_treats_a_plain_meeting_as_unwindowed():
    assert parse_clip_id("IS1009a") == ("IS1009a", None)


def test_parse_clip_id_does_not_mistake_non_numeric_suffixes_for_a_window():
    assert parse_clip_id("EN2002a_part_two") == ("EN2002a_part_two", None)


def test_items_from_clips_dir_pairs_audio_with_its_reference(tmp_path: Path):
    _clip(tmp_path, "IB4001_300_600")
    _clip(tmp_path, "TS3003a_0_600")

    items = items_from_clips_dir(tmp_path)

    assert [i.item_id for i in items] == ["IB4001_300_600", "TS3003a_0_600"]
    assert items[0].meeting_id == "IB4001"
    assert items[0].window == (300.0, 600.0)
    assert items[0].ref_stm_path == tmp_path / "IB4001_300_600.ref.stm"


def test_items_from_clips_dir_refuses_a_clip_without_a_reference(tmp_path: Path):
    """A partial workload yields a combined DER that looks valid and is not."""
    _clip(tmp_path, "IB4001_300_600")
    _clip(tmp_path, "TS3003a_0_600", with_ref=False)

    with pytest.raises(FileNotFoundError, match="TS3003a_0_600"):
        items_from_clips_dir(tmp_path)


def test_items_from_clips_dir_refuses_an_empty_directory(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match=r"No \.wav clips"):
        items_from_clips_dir(tmp_path)


def test_items_from_meetings_uses_the_ami_root_layout(tmp_path: Path):
    (item,) = items_from_meetings(["IS1009a"], tmp_path)

    assert item.window is None
    assert item.audio_path == tmp_path / "IS1009a" / "audio" / "IS1009a.Mix-Headset.wav"
    assert item.ref_stm_path == tmp_path / "stm" / "IS1009a.ref.stm"


def test_materialize_rttm_references_windows_and_rebases_per_clip(tmp_path: Path):
    """A clip must be scored only against the turns its audio contains."""
    clips = tmp_path / "clips"
    clips.mkdir()
    _clip(clips, "m1_300_600")
    rttm_dir = tmp_path / "rttm"
    rttm_dir.mkdir()
    (rttm_dir / "m1.rttm").write_text(RTTM)

    (item,) = materialize_rttm_references(items_from_clips_dir(clips), rttm_dir, tmp_path / "out")

    lines = item.ref_stm_path.read_text().splitlines()
    # The 10 s turn falls before the window and the 950 s turn after it.
    assert len(lines) == 1
    session, _, speaker, start, end, _ = lines[0].split(maxsplit=5)
    assert session == "m1_300_600"
    assert speaker == "B"
    assert (float(start), float(end)) == (5.0, 9.0)


def test_materialize_rttm_references_finds_rttms_inside_split_subdirectories(tmp_path: Path):
    """Published RTTM sets ship partitioned into train/dev/test."""
    clips = tmp_path / "clips"
    clips.mkdir()
    _clip(clips, "m1_300_600")
    split = tmp_path / "rttm" / "test"
    split.mkdir(parents=True)
    (split / "m1.rttm").write_text(RTTM)

    (item,) = materialize_rttm_references(
        items_from_clips_dir(clips), tmp_path / "rttm", tmp_path / "out"
    )

    assert item.ref_stm_path.read_text().strip() != ""


def test_materialize_rttm_references_keeps_a_whole_meeting_unwindowed(tmp_path: Path):
    rttm_dir = tmp_path / "rttm"
    rttm_dir.mkdir()
    (rttm_dir / "m1.rttm").write_text(RTTM)

    (item,) = materialize_rttm_references(
        items_from_meetings(["m1"], tmp_path), rttm_dir, tmp_path / "out"
    )

    assert len(item.ref_stm_path.read_text().splitlines()) == 3


def test_materialize_rttm_references_names_the_meetings_it_cannot_find(tmp_path: Path):
    clips = tmp_path / "clips"
    clips.mkdir()
    _clip(clips, "absent_0_600")
    rttm_dir = tmp_path / "rttm"
    rttm_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="absent"):
        materialize_rttm_references(items_from_clips_dir(clips), rttm_dir, tmp_path / "out")
