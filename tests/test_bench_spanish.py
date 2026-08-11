"""Tests for the Spanish Workload Set registry and materialisation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coro.bench import spanish
from coro.bench.clips import resolve_clip_items


class TestRegistry:
    def test_every_corpus_records_a_licence(self):
        """Pinned by name: a new corpus must have its licence reviewed, not inferred."""
        assert {key: c.licence for key, c in spanish.SPANISH_CORPORA.items()} == {
            "voxpopuli": "CC0-1.0",
            "fleurs": "CC-BY-4.0",
            "mls": "CC-BY-4.0",
        }
        assert [
            key
            for key, c in spanish.SPANISH_CORPORA.items()
            if not (c.licence_url.startswith("https://") and c.homepage.startswith("https://"))
        ] == []

    def test_corpus_keys_are_item_id_prefix_safe(self):
        assert [key for key in spanish.SPANISH_CORPORA if "-" in key] == []

    def test_presets_only_reference_known_corpora(self):
        known = set(spanish.SPANISH_CORPORA)
        assert (
            sorted(
                key
                for preset in spanish.SPANISH_PRESETS.values()
                for key in set(preset.corpora) - known
            )
            == []
        )
        assert {name: list(p.corpora) for name, p in spanish.SPANISH_PRESETS.items()} == {
            "voxpopuli": ["voxpopuli"],
            "fleurs": ["fleurs"],
            "mls": ["mls"],
            "calibration": ["fleurs", "mls"],
            "all": ["voxpopuli", "fleurs", "mls"],
        }

    def test_voxpopuli_is_the_primary_corpus(self):
        assert spanish.SPANISH_CORPORA["voxpopuli"].role == "primary"
        assert spanish.SPANISH_CORPORA["voxpopuli"].licence == "CC0-1.0"

    def test_calibration_preset_covers_both_calibration_sets(self):
        assert set(spanish.SPANISH_PRESETS["calibration"].corpora) == {"fleurs", "mls"}

    def test_every_corpus_is_single_speaker(self):
        assert all(c.single_speaker for c in spanish.SPANISH_CORPORA.values())

    def test_unknown_preset_names_the_known_ones(self):
        with pytest.raises(ValueError, match="calibration"):
            spanish.resolve_spanish_preset("nope")


class TestCorpusOfItem:
    def test_recovers_corpus_from_item_id(self):
        assert spanish.corpus_of_item("fleurs-101") == "fleurs"
        assert spanish.corpus_of_item("mls-10446_10446_000000") == "mls"

    def test_ami_item_ids_do_not_map_to_a_spanish_corpus(self):
        assert spanish.corpus_of_item("IB4001") not in spanish.SPANISH_CORPORA


class TestMaterializeSpanishWorkloadSet:
    def test_writes_clip_pairs_manifest_and_licences(self, tmp_path: Path, fake_spanish_corpus):
        out_dir = spanish.materialize_spanish_workload_set("calibration", tmp_path)

        assert (out_dir / "fleurs-101.wav").exists()
        assert (out_dir / "fleurs-101.ref.stm").exists()
        assert (out_dir / "mls-10446_10446_000000.wav").exists()
        assert (out_dir / spanish.MANIFEST_NAME).exists()
        assert (out_dir / spanish.LICENCES_NAME).exists()

    def test_reference_stm_carries_the_raw_transcript(self, tmp_path: Path, fake_spanish_corpus):
        out_dir = spanish.materialize_spanish_workload_set("fleurs", tmp_path)

        line = (out_dir / "fleurs-101.ref.stm").read_text(encoding="utf-8").strip()
        assert line.startswith("fleurs-101 1 1 0.000 1.000 ")
        assert "¿cómo está el año?" in line

    def test_items_without_a_transcript_are_skipped(self, tmp_path: Path, fake_spanish_corpus):
        out_dir = spanish.materialize_spanish_workload_set("fleurs", tmp_path)

        assert not (out_dir / "fleurs-103.wav").exists()

    def test_manifest_records_licence_and_provenance(self, tmp_path: Path, fake_spanish_corpus):
        out_dir = spanish.materialize_spanish_workload_set("calibration", tmp_path)

        manifest = json.loads((out_dir / spanish.MANIFEST_NAME).read_text(encoding="utf-8"))
        blocks = {block["key"]: block for block in manifest["corpora"]}
        assert blocks["fleurs"]["licence"] == "CC-BY-4.0"
        assert blocks["fleurs"]["hf_dataset"] == "google/fleurs"
        assert blocks["mls"]["single_speaker"] is True
        assert blocks["fleurs"]["materialised"] == 2

    def test_licences_file_lists_every_corpus(self, tmp_path: Path, fake_spanish_corpus):
        out_dir = spanish.materialize_spanish_workload_set("calibration", tmp_path)

        text = (out_dir / spanish.LICENCES_NAME).read_text(encoding="utf-8")
        assert "CC-BY-4.0" in text
        assert "google/fleurs" in text
        assert "yields no meaningful DER" in text

    def test_is_idempotent_and_does_not_refetch(
        self, tmp_path: Path, fake_spanish_corpus, monkeypatch
    ):
        spanish.materialize_spanish_workload_set("mls", tmp_path)

        def explode(*args, **kwargs):
            raise AssertionError("should not refetch an already-materialised preset")

        monkeypatch.setattr(spanish, "resolve_shard_urls", explode)
        out_dir = spanish.materialize_spanish_workload_set("mls", tmp_path)

        assert (out_dir / "mls-10446_10446_000000.ref.stm").exists()

    def test_changing_the_item_count_refetches(self, tmp_path: Path, fake_spanish_corpus):
        spanish.materialize_spanish_workload_set("fleurs", tmp_path, items_per_corpus=1)
        out_dir = spanish.materialize_spanish_workload_set("fleurs", tmp_path, items_per_corpus=2)

        manifest = json.loads((out_dir / spanish.MANIFEST_NAME).read_text(encoding="utf-8"))
        assert manifest["items_per_corpus"] == 2

    def test_no_download_fails_when_not_materialised(self, tmp_path: Path):
        with pytest.raises(RuntimeError, match="--no-download"):
            spanish.materialize_spanish_workload_set("mls", tmp_path, no_download=True)

    def test_output_feeds_the_existing_clips_workload_path(
        self, tmp_path: Path, fake_spanish_corpus
    ):
        out_dir = spanish.materialize_spanish_workload_set("calibration", tmp_path)

        items = resolve_clip_items(out_dir)

        assert {item["item_id"] for item in items} == {
            "fleurs-101",
            "fleurs-102",
            "mls-10446_10446_000000",
        }
        assert all(item["ref_stm_path"] is not None for item in items)
