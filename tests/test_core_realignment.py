"""Flicker correction over per-word speaker labels.

The load-bearing distinction under test is between a *sandwiched island*
(``A B A``), which is the definition of flicker and is absorbed, and a *clean
two-way turn* (``A B``), which is what a genuine turn looks like and is left
alone. An earlier implementation used a punctuation-bounded majority vote and
flattened both; it was measured against six AMI meetings, made DER
speaker-error worse than baseline on 6 of 6 clips, and was replaced. See
ADR 0014 and ``coro/core/realignment.py``.
"""

from __future__ import annotations

from coro.core.models import TranscriptToken
from coro.core.realignment import realign_speaker_flicker
from coro.core.speakers import UNKNOWN_SPEAKER, SpeakerAttribution


def _tok(start, end, text):
    return TranscriptToken(start=start, end=end, text=text, probability=1.0)


def _attributed(*rows):
    """Build an attributed-token list from ``(start, end, text, speaker[, overlap])`` rows."""
    result = []
    for row in rows:
        start, end, text, speaker = row[:4]
        overlap = row[4] if len(row) > 4 else False
        result.append(
            (_tok(start, end, text), SpeakerAttribution(speaker=speaker, overlap=overlap))
        )
    return result


def _speakers(attributed):
    return [a.speaker for _, a in attributed]


# ---------------------------------------------------------------------------
# The core distinction: sandwiched island vs clean turn
# ---------------------------------------------------------------------------


def test_isolated_flicker_is_absorbed_into_the_flanking_speaker():
    """A single-word blip flanked by one stable speaker is noise, not a turn."""
    attributed = _attributed(
        (0.0, 1.0, " esto", 1),
        (1.0, 2.0, " es", 1),
        (2.0, 2.2, " una", 2),  # 0.2s island between 2.0s and 0.8s of speaker 1
        (2.2, 3.0, " prueba.", 1),
    )
    assert _speakers(realign_speaker_flicker(attributed)) == [1, 1, 1, 1]


def test_clean_two_way_turn_is_never_flattened():
    """The rule that replaced the majority vote exists precisely for this case.

    An unpunctuated ``A B`` split is not sandwiched, so nothing marks it as an
    error. The previous punctuation-majority rule flattened it, and that is
    what made DER speaker-error regress.
    """
    attributed = _attributed(
        (0.0, 0.4, " hola", 2),
        (0.4, 0.8, " mundo", 2),
        (2.0, 2.4, " adios", 3),
        (2.4, 2.8, " amigo.", 3),
    )
    assert _speakers(realign_speaker_flicker(attributed)) == [2, 2, 3, 3]


def test_lopsided_two_way_turn_is_still_not_flattened():
    """Even a one-word tail is a turn, not flicker, when nothing follows it."""
    attributed = _attributed(
        (0.0, 2.0, " una", 1),
        (2.0, 4.0, " frase", 1),
        (4.0, 4.1, " larga.", 2),
    )
    assert _speakers(realign_speaker_flicker(attributed)) == [1, 1, 2]


def test_island_flanked_by_two_different_speakers_is_not_absorbed():
    """``A B C`` carries no evidence about which neighbour B belongs to."""
    attributed = _attributed(
        (0.0, 1.0, " uno", 1),
        (1.0, 1.1, " dos", 2),
        (1.1, 2.1, " tres.", 3),
    )
    assert _speakers(realign_speaker_flicker(attributed)) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Duration, not word count
# ---------------------------------------------------------------------------


def test_absorption_is_weighted_by_duration_not_word_count():
    """Three short flanking words must not absorb one long island."""
    attributed = _attributed(
        (0.0, 0.05, " a", 1),
        (0.05, 5.05, " largapalabra", 2),  # 5s island, flanks total 0.1s
        (5.05, 5.10, " c.", 1),
    )
    assert _speakers(realign_speaker_flicker(attributed)) == [1, 2, 1]


def test_multi_word_island_is_absorbed_when_the_flanks_outweigh_it():
    attributed = _attributed(
        (0.0, 3.0, " larga", 1),
        (3.0, 3.2, " y", 2),
        (3.2, 3.4, " breve", 2),
        (3.4, 6.4, " interrupcion.", 1),
    )
    assert _speakers(realign_speaker_flicker(attributed)) == [1, 1, 1, 1]


# ---------------------------------------------------------------------------
# Sentence bounds still gate the correction
# ---------------------------------------------------------------------------


def test_sentence_final_punctuation_bounds_the_correction():
    """A blip is only a blip within one sentence; punctuation is turn evidence."""
    attributed = _attributed(
        (0.0, 1.0, " hola.", 1),
        (1.0, 1.1, " si.", 2),
        (1.1, 2.1, " adios.", 1),
    )
    assert _speakers(realign_speaker_flicker(attributed)) == [1, 2, 1]


def test_opening_mark_bounds_the_correction():
    attributed = _attributed(
        (0.0, 1.0, " bien", 1),
        (1.0, 1.1, " gracias", 2),
        (1.1, 2.1, " ¿y", 1),
        (2.1, 3.1, " tu?", 1),
    )
    # The island sits in the first sentence, whose only other member is " bien",
    # so the flanks do not surround it and nothing is absorbed.
    assert _speakers(realign_speaker_flicker(attributed)) == [1, 2, 1, 1]


# ---------------------------------------------------------------------------
# UNKNOWN_SPEAKER is transparent
# ---------------------------------------------------------------------------


def test_unknown_words_neither_break_nor_form_an_island():
    """`-1` is an abstention: it does not split speaker 1's run in two."""
    attributed = _attributed(
        (0.0, 1.0, " esto", 1),
        (1.0, 5.0, " ruido", UNKNOWN_SPEAKER),
        (5.0, 5.2, " una", 2),
        (5.2, 6.2, " prueba.", 1),
    )
    assert _speakers(realign_speaker_flicker(attributed)) == [1, UNKNOWN_SPEAKER, 1, 1]


def test_unknown_words_are_never_relabelled():
    attributed = _attributed(
        (0.0, 1.0, " a", 1),
        (1.0, 1.2, " b", UNKNOWN_SPEAKER),
        (1.2, 1.4, " c", 2),
        (1.4, 2.4, " d.", 1),
    )
    realigned = realign_speaker_flicker(attributed)
    assert _speakers(realigned) == [1, UNKNOWN_SPEAKER, 1, 1]


def test_sentence_that_is_entirely_unknown_is_unchanged():
    attributed = _attributed((0.0, 0.5, " a", UNKNOWN_SPEAKER), (0.5, 1.0, " b.", UNKNOWN_SPEAKER))
    assert _speakers(realign_speaker_flicker(attributed)) == [UNKNOWN_SPEAKER] * 2


# ---------------------------------------------------------------------------
# Structural guarantees
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty():
    assert realign_speaker_flicker([]) == []


def test_homogeneous_sentence_is_unchanged():
    attributed = _attributed((0.0, 0.4, " todo", 1), (0.4, 0.8, " igual.", 1))
    assert _speakers(realign_speaker_flicker(attributed)) == [1, 1]


def test_alternating_run_collapses_without_contradicting_itself():
    """``A B A B A`` must not relabel B to A and A to B in the same pass."""
    attributed = _attributed(
        (0.0, 1.0, " a", 1),
        (1.0, 1.1, " b", 2),
        (1.1, 2.1, " c", 1),
        (2.1, 2.2, " d", 2),
        (2.2, 3.2, " e.", 1),
    )
    assert _speakers(realign_speaker_flicker(attributed)) == [1, 1, 1, 1, 1]


def test_absorption_preserves_the_overlap_flag():
    attributed = _attributed(
        (0.0, 1.0, " esto", 1, False),
        (1.0, 1.2, " una", 2, True),
        (1.2, 2.0, " prueba.", 1, False),
    )
    realigned = realign_speaker_flicker(attributed)
    assert [a.overlap for _, a in realigned] == [False, True, False]
    assert _speakers(realigned) == [1, 1, 1]


def test_input_list_is_not_mutated():
    attributed = _attributed(
        (0.0, 1.0, " esto", 1),
        (1.0, 1.2, " una", 2),
        (1.2, 2.0, " prueba.", 1),
    )
    realign_speaker_flicker(attributed)
    assert _speakers(attributed) == [1, 2, 1]
