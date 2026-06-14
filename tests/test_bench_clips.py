"""Tests for clip-directory workload resolution and the curated Spanish reference."""

from __future__ import annotations

from pathlib import Path

from asr_diar_server.bench.clips import resolve_clip_items


def _touch(path: Path, content: str = "") -> None:
    path.write_text(content)


class TestResolveClipItems:
    def test_pairs_audio_with_sibling_ref_stm(self, tmp_path: Path):
        _touch(tmp_path / "clipA.wav")
        _touch(tmp_path / "clipA.ref.stm", "clipA 1 A 0.0 1.0 hola\n")
        _touch(tmp_path / "clipB.wav")
        _touch(tmp_path / "clipB.ref.stm", "clipB 1 A 0.0 1.0 mundo\n")

        items = resolve_clip_items(tmp_path)

        assert [it["item_id"] for it in items] == ["clipA", "clipB"]
        assert items[0]["ref_stm_path"] == tmp_path / "clipA.ref.stm"
        assert items[0]["audio_path"] == tmp_path / "clipA.wav"

    def test_missing_ref_stm_yields_none(self, tmp_path: Path):
        _touch(tmp_path / "solo.wav")

        items = resolve_clip_items(tmp_path)

        assert len(items) == 1
        assert items[0]["ref_stm_path"] is None

    def test_ignores_non_audio_files(self, tmp_path: Path):
        _touch(tmp_path / "notes.txt")
        _touch(tmp_path / "clip.ref.stm")

        assert resolve_clip_items(tmp_path) == []

    def test_accepts_multiple_audio_extensions(self, tmp_path: Path):
        _touch(tmp_path / "a.wav")
        _touch(tmp_path / "b.mp3")

        items = resolve_clip_items(tmp_path)

        assert {it["item_id"] for it in items} == {"a", "b"}

    def test_missing_directory_returns_empty(self, tmp_path: Path):
        assert resolve_clip_items(tmp_path / "does-not-exist") == []


class TestSpanishReference:
    def test_bundled_spanish_reference_exists_and_is_valid_stm(self):
        from asr_diar_server.bench.data import SPANISH_REFERENCE_STMS

        path = SPANISH_REFERENCE_STMS["RNE14-agosto-13"]
        assert path.exists()
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines, "reference STM must be non-empty"
        speakers = set()
        for line in lines:
            parts = line.split(maxsplit=5)
            assert len(parts) == 6, f"malformed STM line: {line!r}"
            assert parts[0] == "RNE14-agosto-13"
            float(parts[3])
            float(parts[4])
            speakers.add(parts[2])
        # Curated reference is multi-speaker (diarization is meaningful).
        assert len(speakers) >= 2
