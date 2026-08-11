"""Utterance grouping: maximal same-speaker runs over the per-word truth.

Grouping reads ``word_segments`` rather than ``segments`` so the speaker-turn
view stays correct regardless of what shape segments take (ADR 0015).
"""

from __future__ import annotations

import pytest

from coro.api.schemas import WhisperWord
from coro.api.utterances import group_words_into_utterances, mean_confidence


def _word(text: str, speaker: str, score: float | None = 1.0) -> WhisperWord:
    return WhisperWord(word=text, start=0.0, end=1.0, score=score, speaker=speaker)


class TestUtteranceGrouping:
    def test_consecutive_same_speaker_words_form_one_utterance(self):
        utterances = group_words_into_utterances([_word("a", "1"), _word("b", "1")])
        assert len(utterances) == 1
        assert utterances[0].speaker == "1"
        assert utterances[0].text == "a b"

    def test_speaker_change_starts_a_new_utterance(self):
        words = [_word("a", "1"), _word("b", "2"), _word("c", "1")]
        assert [u.speaker for u in group_words_into_utterances(words)] == ["1", "2", "1"]

    def test_recurring_speaker_is_not_merged_across_a_turn(self):
        words = [_word("a", "1"), _word("b", "2"), _word("c", "1")]
        assert len(group_words_into_utterances(words)) == 3

    def test_empty_input_yields_no_utterances(self):
        assert group_words_into_utterances([]) == []

    def test_unknown_speaker_words_group_like_any_other_label(self):
        words = [_word("a", "-1"), _word("b", "-1"), _word("c", "1")]
        assert [u.speaker for u in group_words_into_utterances(words)] == ["-1", "1"]


class TestUtteranceAggregates:
    def test_confidence_is_the_mean_of_its_words(self):
        words = [_word("a", "1", 0.5), _word("b", "1", 1.0)]
        assert group_words_into_utterances(words)[0].confidence == pytest.approx(0.75, abs=1e-9)

    def test_mean_confidence_of_no_words_is_absent_not_zero(self):
        """An aggregate over no evidence is absent; ``0.0`` claims no confidence."""
        assert mean_confidence([]) is None

    def test_unmeasured_words_are_excluded_from_the_mean_not_counted(self):
        """A word without a probability must not drag the mean in either direction.

        Counting it as ``0.0`` understates the words that *were* measured and
        counting it as ``1.0`` overstates them; the only correct denominator is
        the measured words.
        """
        words = [_word("a", "1", 0.5), _word("b", "1", None), _word("c", "1", 1.0)]
        assert mean_confidence(words) == pytest.approx(0.75, abs=1e-9)

    def test_mean_confidence_is_absent_when_no_word_was_measured(self):
        words = [_word("a", "1", None), _word("b", "1", None)]
        assert mean_confidence(words) is None

    def test_span_covers_first_start_to_last_end(self):
        words = [
            WhisperWord(word="a", start=0.25, end=0.75, score=1.0, speaker="1"),
            WhisperWord(word="b", start=0.75, end=2.5, score=1.0, speaker="1"),
        ]
        utterance = group_words_into_utterances(words)[0]
        assert (utterance.start, utterance.end) == (0.25, 2.5)
