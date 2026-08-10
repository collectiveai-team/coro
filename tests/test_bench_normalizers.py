"""Vendored Basic Text Normalizer: the Normalized Metric Lane protocol (ADR 0011)."""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from coro.bench.normalizers import BasicTextNormalizer
from coro.bench.normalizers.basic import remove_symbols, remove_symbols_and_diacritics

_DIACRITICS = ("á", "é", "í", "ó", "ú", "ñ", "ü")


@pytest.fixture
def normalizer() -> BasicTextNormalizer:
    """Return the normalizer configured exactly as the Quality Benchmark uses it."""
    return BasicTextNormalizer()


class TestDiacriticPreservation:
    """Diacritic preservation is the load-bearing property for Spanish scoring."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("está", {"á"}),
            ("año", {"ñ"}),
            ("canción", {"ó"}),
            ("¿Qué más dijo él?", {"á", "é"}),
            ("Perdón, señor Ibáñez.", {"á", "ñ", "ó"}),
            ("aún ratifiqué la petición", {"é", "ó", "ú"}),
        ],
    )
    def test_spanish_diacritics_survive_normalization(
        self,
        normalizer: BasicTextNormalizer,
        text: str,
        expected: set[str],
    ):
        # Asserted on the input too, so a sample that carries no diacritic
        # cannot make this case pass vacuously.
        assert {mark for mark in _DIACRITICS if mark in text.lower()} == expected
        normalized = normalizer(text)
        assert {mark for mark in _DIACRITICS if mark in normalized} == expected

    def test_normalizer_does_not_collapse_esta_and_estar_accented(
        self,
        normalizer: BasicTextNormalizer,
    ):
        """`esta` and `está` are distinct words and must stay distinct."""
        assert normalizer("esta") != normalizer("está")

    def test_normalizer_does_not_collapse_ano_and_ene_word(
        self,
        normalizer: BasicTextNormalizer,
    ):
        """`ano` and `año` are distinct words and must stay distinct."""
        assert normalizer("ano") != normalizer("año")

    def test_decomposed_and_composed_accents_normalize_identically(
        self,
        normalizer: BasicTextNormalizer,
    ):
        """NFKC folding means encoding form does not change the score."""
        composed = unicodedata.normalize("NFC", "está")
        decomposed = unicodedata.normalize("NFD", "está")
        assert composed != decomposed
        assert normalizer(composed) == normalizer(decomposed)

    def test_remove_diacritics_true_would_collapse_them(self):
        """Guard the rejected setting: opting in demonstrably destroys the distinction."""
        stripping = BasicTextNormalizer(remove_diacritics=True)
        assert stripping("está") == stripping("esta")
        assert stripping("año") == stripping("ano")


class TestBasicNormalizerProtocol:
    def test_lowercases(self, normalizer: BasicTextNormalizer):
        assert normalizer("Hola MUNDO") == "hola mundo"

    def test_removes_bracketed_content(self, normalizer: BasicTextNormalizer):
        assert normalizer("hola [inaudible] mundo") == "hola mundo"

    def test_removes_angle_bracketed_content(self, normalizer: BasicTextNormalizer):
        assert normalizer("hola <unk> mundo") == "hola mundo"

    def test_removes_parenthesised_content(self, normalizer: BasicTextNormalizer):
        assert normalizer("hola (risas) mundo") == "hola mundo"

    def test_maps_punctuation_to_spaces_and_collapses_whitespace(
        self,
        normalizer: BasicTextNormalizer,
    ):
        assert normalizer("Hola,   mundo!!") == "hola mundo"

    def test_strips_spanish_opening_punctuation(self, normalizer: BasicTextNormalizer):
        assert normalizer("¿Cómo estás?") == "cómo estás"

    def test_punctuation_does_not_join_words(self, normalizer: BasicTextNormalizer):
        """Punctuation becomes a space, not nothing, so token counts stay honest."""
        assert normalizer("uno,dos") == "uno dos"

    def test_strips_leading_and_trailing_whitespace(self, normalizer: BasicTextNormalizer):
        assert normalizer("  hola  ") == "hola"

    def test_is_idempotent(self, normalizer: BasicTextNormalizer):
        once = normalizer("¿Qué tal, señor?  [tos]")
        assert normalizer(once) == once

    def test_empty_string_normalizes_to_empty(self, normalizer: BasicTextNormalizer):
        assert normalizer("") == ""

    def test_split_letters_separates_grapheme_clusters(self):
        splitting = BasicTextNormalizer(split_letters=True)
        assert splitting("año") == "a ñ o"


class TestVendoredHelpers:
    def test_remove_symbols_keeps_diacritics(self):
        assert remove_symbols("está.") == "está "

    def test_remove_symbols_and_diacritics_drops_them(self):
        assert remove_symbols_and_diacritics("está.") == "esta "

    def test_additional_diacritics_expand_only_when_stripping(self):
        assert remove_symbols_and_diacritics("œ") == "oe"
        assert remove_symbols("œ") == "œ"


class TestEnglishNormalizerIsNotUsed:
    """The English-specific normalizer must not be reachable for any language."""

    def test_english_normalizer_is_not_vendored(self):
        normalizers_dir = Path("coro/bench/normalizers")
        assert normalizers_dir.is_dir()
        names = {p.name for p in normalizers_dir.iterdir()}
        assert "english.py" not in names

    def test_no_source_file_references_an_english_normalizer(self):
        offenders = [
            path
            for path in Path("coro").rglob("*.py")
            if "EnglishTextNormalizer" in path.read_text(encoding="utf-8")
        ]
        # The vendored module names it once to record that it is rejected.
        assert offenders == [Path("coro/bench/normalizers/basic.py")]

    def test_quality_scoring_never_strips_diacritics(self):
        from coro.bench.quality import _BASIC_NORMALIZER

        assert _BASIC_NORMALIZER.clean is remove_symbols
        assert _BASIC_NORMALIZER.split_letters is False
