"""Language Drift Metric (issue #64's acceptance criterion).

Fixture sentences are deliberately simple, unambiguous Romance-language text
picked to be far from the langid library's decision boundary, so a
misclassification here would indicate a real regression rather than corpus
noise.
"""

from __future__ import annotations

import pytest

from coro.bench.language_drift import (
    DEFAULT_WINDOW_WORDS,
    classify_windows,
    compute_drift_report,
    foreign_word_rate,
    function_word_hits,
    language_histogram,
    orthographic_marker_hits,
    substitution_words,
)

SPANISH = (
    "no se trata sin embargo de una experimentacion es una prueba que se usa "
    "para descartar una o mas de las posibles hipotesis haciendo preguntas y "
    "observaciones que guian la investigacion cientifica"
)
PORTUGUESE = (
    "bom dia como voce esta hoje meu amigo vamos a praia porque o sol esta "
    "brilhando e nao ha nuvens no ceu esta tarde"
)
ITALIAN = (
    "buongiorno come stai oggi amico mio andiamo alla spiaggia perche il "
    "sole splende e non ci sono nuvole nel cielo questo pomeriggio"
)


class TestClassifyWindows:
    def test_all_spanish_has_no_foreign_windows(self) -> None:
        words = SPANISH.split()
        windows = classify_windows(words)
        assert all(not w.is_foreign for w in windows)
        assert foreign_word_rate(windows, len(words)) == 0.0

    def test_all_portuguese_is_entirely_foreign(self) -> None:
        words = PORTUGUESE.split()
        windows = classify_windows(words)
        assert all(w.is_foreign for w in windows)
        assert foreign_word_rate(windows, len(words)) == 1.0
        hist = language_histogram(windows)
        assert hist["pt"] == len(windows)

    def test_mixed_transcript_shows_positional_drift(self) -> None:
        """The exact scenario issue #64 describes: Spanish that drifts to
        Portuguese partway through, as if a later window auto-detected wrong.
        """
        words = (SPANISH + " " + PORTUGUESE).split()
        windows = classify_windows(words)
        first, last = windows[0], windows[-1]
        assert first.language == "es"
        assert last.language == "pt"
        rate = foreign_word_rate(windows, len(words))
        assert 0.0 < rate < 1.0

    def test_short_trailing_remainder_is_absorbed_not_classified_alone(self) -> None:
        """A window_size=7 split of 8 words would leave a lone trailing word;
        single-word langid is exactly the unreliable case this metric exists
        to avoid, so the tail must merge into the previous window instead.
        """
        words = (SPANISH.split())[:8]
        windows = classify_windows(words, window_size=7)
        assert len(windows) == 1
        assert windows[0].word_count == 8

    def test_window_size_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            classify_windows(["hola"], window_size=0)

    def test_empty_input_yields_no_windows(self) -> None:
        assert classify_windows([]) == []


class TestSubstitutionWords:
    def test_identical_transcripts_have_no_substitutions(self) -> None:
        assert substitution_words(SPANISH, SPANISH) == []

    def test_correctly_transcribed_foreign_proper_noun_is_not_a_substitution(self) -> None:
        """A place name like Sao Paulo appearing identically in both reference
        and hypothesis is a match, not a substitution -- even though it would
        read as foreign to a langid classifier -- so it must not count.
        """
        ref = "el vuelo parte desde sao paulo a las nueve"
        hyp = "el vuelo parte desde sao paulo a las nueve"
        assert substitution_words(ref, hyp) == []

    def test_substituted_word_is_reported_with_its_hypothesis_index(self) -> None:
        ref = "el gato come pescado todos los dias"
        hyp = "el gato come peixe todos los dias"
        subs = substitution_words(ref, hyp)
        assert subs == [(3, "peixe")]

    def test_insertions_and_deletions_are_excluded(self) -> None:
        ref = "el gato come pescado"
        hyp = "el gato negro come pescado extra"
        # "negro" is an insertion, "extra" is a trailing insertion; neither
        # has a reference counterpart to be a substitution against.
        assert substitution_words(ref, hyp) == []


class TestOrthographicMarkers:
    @pytest.mark.parametrize(
        ("text", "lang"),
        [
            ("não há problema, é uma boa decisão", "pt"),
            ("questa e una situazione senza soluzione", "it"),
            ("bon dia company meu, anem-hi", "ca"),
        ],
    )
    def test_marker_language_is_detected(self, text: str, lang: str) -> None:
        hits = orthographic_marker_hits(text)
        assert hits.get(lang, 0) > 0

    def test_clean_spanish_has_no_marker_hits(self) -> None:
        assert orthographic_marker_hits(SPANISH) == {}


class TestFunctionWordHits:
    """Session 9's real-audio finding (mTEDx public reproduction, findings.md):
    a single English function word can replace its Spanish counterpart
    mid-sentence without the surrounding 7-word window's langid verdict
    flipping, since six of seven words are still correct Spanish. These
    fixtures are synthetic constructions of that failure mode, not transcripts
    of any specific audio.
    """

    def test_clean_spanish_has_no_function_word_hits(self) -> None:
        assert function_word_hits(SPANISH.split()) == []

    def test_single_english_word_injected_into_spanish_is_caught(self) -> None:
        words = ["el", "tren", "llega", "a", "las", "diez", "de", "la", "mañana"]
        words[2] = "at"  # should have been "llega"'s complement "a"
        hits = function_word_hits(words)
        assert hits == [(2, "at", "en")]

    def test_common_spanish_words_are_never_flagged_as_english(self) -> None:
        """ "no", "as" (ace), "he" (verb haber), "so" (archaic interjection) and
        "si" are all valid Spanish words and must not collide with the
        English list, even though their English near-equivalents are common
        function words.
        """
        risky_spanish_words = ["no", "as", "he", "so", "si", "la", "un", "mas"]
        assert function_word_hits(risky_spanish_words) == []

    def test_multiple_hits_are_all_reported_with_position(self) -> None:
        words = ["el", "tren", "llega", "a", "las", "diez", "de", "the", "mañana"]
        words[2] = "in"
        hits = function_word_hits(words)
        assert hits == [(2, "in", "en"), (7, "the", "en")]


class TestComputeDriftReport:
    def test_all_spanish_reference_and_hypothesis_yields_zero_drift(self) -> None:
        report = compute_drift_report(SPANISH, reference=SPANISH)
        assert report.foreign_word_rate == 0.0
        assert report.foreign_substitution_rate == 0.0
        assert report.substitution_count == 0
        assert report.orthographic_marker_hits == {}
        assert report.function_word_hits == {}
        assert report.function_word_rate == 0.0

    def test_injected_function_word_is_visible_even_when_window_langid_misses_it(
        self,
    ) -> None:
        """The whole point of function_word_hits: a single-word injection
        inside an otherwise-correct Spanish window should not need the
        window's langid verdict to flip to be detected.
        """
        words = SPANISH.split()
        words[3] = "in"  # single-word injection, rest of the window stays Spanish
        hyp = " ".join(words)
        report = compute_drift_report(hyp)
        assert report.function_word_hits["en"] == 1
        assert report.function_word_rate == 1 / len(words)

    def test_hypothesis_drifting_to_portuguese_is_fully_foreign(self) -> None:
        report = compute_drift_report(PORTUGUESE)
        assert report.foreign_word_rate == 1.0
        assert report.language_histogram["pt"] > 0
        assert report.foreign_substitution_rate is None  # no reference given

    def test_foreign_substitution_rate_isolates_drift_from_generic_errors(self) -> None:
        """A hypothesis with one generic ASR slip (mismo/mismos) and one
        language-drift slip (a chunk replaced by Portuguese) should attribute
        only the latter to language drift.
        """
        ref = " ".join([SPANISH] * 3)
        hyp_words = ref.split()
        hyp_words[5] = "mismos"  # generic substitution, still Spanish
        hyp = " ".join(hyp_words[:20]) + " " + PORTUGUESE + " " + " ".join(hyp_words[40:])
        report = compute_drift_report(hyp, reference=ref, window_size=DEFAULT_WINDOW_WORDS)
        assert report.substitution_count >= 2
        assert report.foreign_substitution_rate is not None
        assert 0.0 < report.foreign_substitution_rate < 1.0

    def test_default_window_size_matches_module_constant(self) -> None:
        report = compute_drift_report(SPANISH)
        assert all(w.word_count <= DEFAULT_WINDOW_WORDS * 2 - 1 for w in report.windows)
