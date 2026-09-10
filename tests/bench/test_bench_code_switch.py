"""Tests for the synthetic code-switch corpus registry and materialisation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coro.bench import code_switch
from coro.bench.clips import resolve_clip_items


class TestRegistry:
    def test_every_source_corpus_records_a_licence(self):
        assert {key: c.licence for key, c in code_switch.CODE_SWITCH_CORPORA.items()} == {
            "fleurs_es": "CC-BY-4.0",
            "fleurs_en": "CC-BY-4.0",
        }

    def test_carrier_and_switch_corpora_speak_different_languages(self):
        assert code_switch.CODE_SWITCH_CORPORA["fleurs_es"].language == "es"
        assert code_switch.CODE_SWITCH_CORPORA["fleurs_en"].language == "en"

    def test_presets_only_reference_known_corpora(self):
        known = set(code_switch.CODE_SWITCH_CORPORA)
        presets = list(code_switch.CODE_SWITCH_PRESETS.values())

        assert presets != []
        assert all(p.carrier_corpus in known and p.switch_corpus in known for p in presets)

    def test_presets_cover_both_splice_patterns(self):
        assert {p.pattern for p in code_switch.CODE_SWITCH_PRESETS.values()} == {
            "sparse",
            "paragraph",
        }

    def test_unknown_preset_names_the_known_ones(self):
        with pytest.raises(ValueError, match="sparse-splice"):
            code_switch.resolve_code_switch_preset("nope")


class TestBuildSequence:
    def test_paragraph_pattern_alternates_every_clip(self):
        sequence = code_switch.build_sequence("paragraph", ["c0", "c1"], ["s0", "s1"])

        assert sequence == [
            ("carrier", "c0"),
            ("switch", "s0"),
            ("carrier", "c1"),
            ("switch", "s1"),
        ]

    def test_paragraph_pattern_appends_leftover_carrier_clips(self):
        sequence = code_switch.build_sequence("paragraph", ["c0", "c1", "c2"], ["s0"])

        assert sequence == [
            ("carrier", "c0"),
            ("switch", "s0"),
            ("carrier", "c1"),
            ("carrier", "c2"),
        ]

    def test_sparse_pattern_inserts_one_switch_clip_near_the_middle(self):
        sequence = code_switch.build_sequence("sparse", ["c0", "c1", "c2", "c3"], ["s0"])

        assert sequence == [
            ("carrier", "c0"),
            ("carrier", "c1"),
            ("switch", "s0"),
            ("carrier", "c2"),
            ("carrier", "c3"),
        ]

    def test_sparse_pattern_never_drops_a_switch_row_on_rounding_collisions(self):
        # 3 carrier clips for 3 switch clips is a case a naive round()-based
        # index scheme collides on (see the code_switch.build_sequence
        # docstring) -- every switch row must still appear exactly once.
        sequence = code_switch.build_sequence("sparse", ["c0", "c1", "c2"], ["s0", "s1", "s2"])

        assert [row for role, row in sequence if role == "switch"] == ["s0", "s1", "s2"]
        assert [row for role, row in sequence if role == "carrier"] == ["c0", "c1", "c2"]

    def test_sparse_pattern_with_no_switch_rows_is_carrier_only(self):
        assert code_switch.build_sequence("sparse", ["c0", "c1"], []) == [
            ("carrier", "c0"),
            ("carrier", "c1"),
        ]

    def test_unknown_pattern_raises(self):
        with pytest.raises(ValueError, match="sparse, paragraph"):
            code_switch.build_sequence("bogus", ["c0"], ["s0"])


class TestMaterializeCodeSwitchCorpus:
    def test_writes_clip_pairs_and_manifest(self, tmp_path: Path, fake_code_switch_corpus):
        out_dir = code_switch.materialize_code_switch_corpus("sparse-splice", tmp_path, items=2)

        assert (out_dir / "sparse-splice-000.wav").exists()
        assert (out_dir / "sparse-splice-000.ref.stm").exists()
        assert (out_dir / "sparse-splice-001.wav").exists()
        assert (out_dir / code_switch.MANIFEST_NAME).exists()
        assert (out_dir / code_switch.LICENCES_NAME).exists()

    def test_manifest_records_exact_per_segment_ground_truth(
        self, tmp_path: Path, fake_code_switch_corpus
    ):
        out_dir = code_switch.materialize_code_switch_corpus("sparse-splice", tmp_path, items=1)

        manifest = json.loads((out_dir / code_switch.MANIFEST_NAME).read_text(encoding="utf-8"))
        item = manifest["items"][0]
        assert item["item_id"] == "sparse-splice-000"
        # 4 carrier clips + 1 switch clip per item for this preset.
        assert len(item["segments"]) == 5
        languages = [seg["language"] for seg in item["segments"]]
        assert languages.count("es") == 4
        assert languages.count("en") == 1
        # Segments are contiguous and non-overlapping, in order.
        starts = [seg["start_seconds"] for seg in item["segments"]]
        ends = [seg["end_seconds"] for seg in item["segments"]]
        assert all(next_start >= end for next_start, end in zip(starts[1:], ends[:-1], strict=True))

    def test_reference_stm_concatenates_every_segment_text(
        self, tmp_path: Path, fake_code_switch_corpus
    ):
        out_dir = code_switch.materialize_code_switch_corpus("paragraph-level", tmp_path, items=1)

        stm_text = (out_dir / "paragraph-level-000.ref.stm").read_text(encoding="utf-8")
        assert "es sentence number 0" in stm_text
        assert "en sentence number 0" in stm_text
        assert "es sentence number 1" in stm_text
        assert "en sentence number 1" in stm_text

    def test_manifest_records_source_corpus_licences(self, tmp_path: Path, fake_code_switch_corpus):
        out_dir = code_switch.materialize_code_switch_corpus("sparse-splice", tmp_path, items=2)

        manifest = json.loads((out_dir / code_switch.MANIFEST_NAME).read_text(encoding="utf-8"))
        blocks = {block["key"]: block for block in manifest["source_corpora"]}
        assert blocks["fleurs_es"]["licence"] == "CC-BY-4.0"
        assert blocks["fleurs_es"]["materialised"] == 8  # 2 items * 4 carrier clips
        assert blocks["fleurs_en"]["materialised"] == 2  # 2 items * 1 switch clip

    def test_licences_file_lists_both_source_corpora(self, tmp_path: Path, fake_code_switch_corpus):
        out_dir = code_switch.materialize_code_switch_corpus("sparse-splice", tmp_path, items=1)

        text = (out_dir / code_switch.LICENCES_NAME).read_text(encoding="utf-8")
        assert "google/fleurs" in text
        assert "CC-BY-4.0" in text
        assert "no human language annotation" in text

    def test_is_idempotent_and_does_not_refetch(
        self, tmp_path: Path, fake_code_switch_corpus, monkeypatch
    ):
        code_switch.materialize_code_switch_corpus("sparse-splice", tmp_path, items=1)

        def explode(*args, **kwargs):
            raise AssertionError("should not refetch an already-materialised preset")

        monkeypatch.setattr(code_switch, "resolve_shard_urls", explode)
        out_dir = code_switch.materialize_code_switch_corpus("sparse-splice", tmp_path, items=1)

        assert (out_dir / "sparse-splice-000.ref.stm").exists()

    def test_changing_the_item_count_refetches(self, tmp_path: Path, fake_code_switch_corpus):
        code_switch.materialize_code_switch_corpus("sparse-splice", tmp_path, items=1)
        out_dir = code_switch.materialize_code_switch_corpus("sparse-splice", tmp_path, items=2)

        manifest = json.loads((out_dir / code_switch.MANIFEST_NAME).read_text(encoding="utf-8"))
        assert manifest["items_requested"] == 2
        assert (out_dir / "sparse-splice-001.wav").exists()

    def test_no_download_fails_when_not_materialised(self, tmp_path: Path):
        with pytest.raises(RuntimeError, match="--no-download"):
            code_switch.materialize_code_switch_corpus("sparse-splice", tmp_path, no_download=True)

    def test_temporary_per_clip_parts_are_cleaned_up(self, tmp_path: Path, fake_code_switch_corpus):
        out_dir = code_switch.materialize_code_switch_corpus("sparse-splice", tmp_path, items=1)

        leftovers = list(out_dir.glob(".sparse-splice-*.parts"))
        assert leftovers == []

    def test_output_feeds_the_existing_clips_workload_path(
        self, tmp_path: Path, fake_code_switch_corpus
    ):
        out_dir = code_switch.materialize_code_switch_corpus("paragraph-level", tmp_path, items=2)

        items = resolve_clip_items(out_dir)

        assert {item["item_id"] for item in items} == {
            "paragraph-level-000",
            "paragraph-level-001",
        }
        assert all(item["ref_stm_path"] is not None for item in items)
