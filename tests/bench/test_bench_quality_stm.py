"""Tests for the STM text helpers of the Quality Benchmark."""

from __future__ import annotations

from pathlib import Path


class TestNormalizedStm:
    def test_normalize_transcript_text_applies_basic_normalizer(self):
        from coro.bench.quality_stm import normalize_transcript_text

        assert normalize_transcript_text("Hello,   world!! [cough]") == "hello world"

    def test_normalize_transcript_text_preserves_spanish_diacritics(self):
        from coro.bench.quality_stm import normalize_transcript_text

        assert normalize_transcript_text("¿Está el señor Ibáñez?") == "está el señor ibáñez"

    def test_write_normalized_stm_preserves_metadata_and_normalizes_text(
        self,
        tmp_path: Path,
    ):
        from coro.bench.quality_stm import write_normalized_stm

        src = tmp_path / "in.stm"
        dst = tmp_path / "out.stm"
        src.write_text("meeting 1 A 0.000 1.500 Hello,   world!!\n")

        write_normalized_stm(src, dst)

        assert dst.read_text() == "meeting 1 A 0.000 1.500 hello world\n"

    def test_write_normalized_stm_keeps_diacritics_in_text_field(self, tmp_path: Path):
        from coro.bench.quality_stm import write_normalized_stm

        src = tmp_path / "in.stm"
        dst = tmp_path / "out.stm"
        src.write_text("sesion 1 A 0.000 1.500 ¿Cuántos años más?\n")

        write_normalized_stm(src, dst)

        assert dst.read_text() == "sesion 1 A 0.000 1.500 cuántos años más\n"

    def test_write_normalized_stm_leaves_source_untouched(self, tmp_path: Path):
        """Normalization happens at scoring time; STM artifacts stay intact."""
        from coro.bench.quality_stm import write_normalized_stm

        src = tmp_path / "in.stm"
        dst = tmp_path / "out.stm"
        original = "meeting 1 A 0.000 1.500 Hello,   world!!\n"
        src.write_text(original)

        write_normalized_stm(src, dst)

        assert src.read_text() == original


class TestDiarizationOnlyStm:
    def test_is_diarization_only_stm_detects_sentinel(self, tmp_path: Path):
        from coro.bench.quality_stm import is_diarization_only_stm

        ref = tmp_path / "rec.ref.stm"
        hyp = tmp_path / "rec.hyp.stm"
        ref.write_text("rec 1 spkA 0.000 2.000 <sd>\nrec 1 spkB 2.500 4.000 <sd>\n")
        hyp.write_text("rec 1 0 0.100 2.100 hola mundo\nrec 1 1 2.600 3.900 que tal\n")
        assert is_diarization_only_stm(ref) is True
        assert is_diarization_only_stm(hyp) is False
