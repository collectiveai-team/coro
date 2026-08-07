"""Text schemas used by Quality Benchmark scoring.

Covers both schemas the MeetEval Metric Set is reported under: the
punctuation-stripping one coro has always used, and the Leaderboard Text Schema
that matches how published ASR results are scored.
"""

from __future__ import annotations

from pathlib import Path

from coro.bench.text import (
    TEXT_SCHEMAS,
    english_normalizer,
    whisper_english_text,
    strip_punctuation,
    write_schema_stm,
)


class TestStripPunctuation:
    def test_removes_punctuation_and_collapses_whitespace(self):
        assert strip_punctuation("Hello,   world!!") == "Hello world"

    def test_preserves_case(self):
        """Case survives — the reason this schema is not comparable to published WERs."""
        assert strip_punctuation("Okay .") == "Okay"


class TestLeaderboardText:
    def test_lowercases(self):
        assert whisper_english_text("Okay , Right .") == "okay right"

    def test_expands_contractions(self):
        assert whisper_english_text("I'm sure we're done .") == "i am sure we are done"

    def test_removes_filler_words(self):
        assert whisper_english_text("Um so uh we start .") == "so we start"

    def test_empties_a_backchannel_only_segment(self):
        """A reference segment of pure backchannel normalizes away entirely."""
        assert whisper_english_text("Mm-hmm .") == ""

    def test_normalizer_is_constructed_once(self):
        assert english_normalizer() is english_normalizer()


class TestWriteSchemaStm:
    def test_preserves_metadata_and_normalizes_only_the_text(self, tmp_path: Path):
        src = tmp_path / "in.stm"
        dst = tmp_path / "out.stm"
        src.write_text("meeting 1 A 0.000 1.500 Hello,   world!!\n")

        write_schema_stm(src, dst, strip_punctuation)

        assert dst.read_text() == "meeting 1 A 0.000 1.500 Hello world\n"

    def test_drops_segments_emptied_by_normalization(self, tmp_path: Path):
        src = tmp_path / "in.stm"
        dst = tmp_path / "out.stm"
        src.write_text(
            "meeting 1 A 0.000 1.500 Mm-hmm .\nmeeting 1 A 2.000 3.000 Real words here .\n"
        )

        write_schema_stm(src, dst, whisper_english_text)

        assert dst.read_text() == "meeting 1 A 2.000 3.000 real words here\n"

    def test_writes_nothing_when_every_segment_normalizes_away(self, tmp_path: Path):
        src = tmp_path / "in.stm"
        dst = tmp_path / "out.stm"
        src.write_text("meeting 1 A 0.000 1.500 Mm-hmm .\n")

        write_schema_stm(src, dst, whisper_english_text)

        assert dst.read_text() == ""


class TestSchemaRegistry:
    def test_registry_names_both_schemas_in_report_order(self):
        assert [key for key, _ in TEXT_SCHEMAS] == ["unpunctuated", "whisper_english"]

    def test_schemas_disagree_on_the_same_text(self):
        """The two schemas must stay distinct, or reporting both is pointless."""
        text = "Um , I'm Okay ."
        by_key = dict(TEXT_SCHEMAS)
        assert by_key["unpunctuated"](text) != by_key["whisper_english"](text)
