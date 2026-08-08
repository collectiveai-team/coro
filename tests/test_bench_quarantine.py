"""Tests for the circular-reference quarantine."""

from __future__ import annotations

from pathlib import Path

import pytest

from coro.bench.clips import resolve_clip_items
from coro.bench.quarantine import (
    CircularReferenceError,
    assert_scorable_reference,
    reference_quarantine_reason,
)


class TestReferenceQuarantineReason:
    def test_legacy_ground_truth_tree_is_quarantined(self):
        reason = reference_quarantine_reason(Path("benchmark/groundtruth/es/session.ref.stm"))

        assert reason is not None
        assert "quarantined" in reason

    def test_quarantine_is_case_insensitive(self):
        assert reference_quarantine_reason(Path("/srv/Benchmark/GroundTruth/a.stm")) is not None

    def test_hypothesis_stm_is_quarantined(self):
        reason = reference_quarantine_reason(Path("run/hyp/IB4001.hyp.stm"))

        assert reason is not None
        assert ".hyp.stm" in reason

    def test_corpus_reference_is_scorable(self):
        assert reference_quarantine_reason(Path("spanish-corpora/mls/mls-1.ref.stm")) is None


class TestAssertScorableReference:
    def test_raises_for_self_generated_reference(self):
        with pytest.raises(CircularReferenceError) as excinfo:
            assert_scorable_reference(Path("out/hyp/x.hyp.stm"))

        assert "Refusing to score" in str(excinfo.value)

    def test_passes_for_public_corpus_reference(self, tmp_path: Path):
        assert_scorable_reference(tmp_path / "fleurs-1.ref.stm")


class TestClipsDirQuarantine:
    def test_clips_dir_under_quarantine_is_rejected(self, tmp_path: Path):
        clips = tmp_path / "benchmark" / "groundtruth"
        clips.mkdir(parents=True)
        (clips / "a.wav").write_text("")

        with pytest.raises(CircularReferenceError):
            resolve_clip_items(clips)

    def test_ordinary_clips_dir_is_accepted(self, tmp_path: Path):
        (tmp_path / "a.wav").write_text("")

        assert [it["item_id"] for it in resolve_clip_items(tmp_path)] == ["a"]
