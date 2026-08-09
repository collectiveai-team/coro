"""Speaker timeline merging and span attribution rules."""

from __future__ import annotations

from coro.core.models import SpeakerSegment, TranscriptToken
from coro.core.speakers import (
    NO_DIARIZATION_SPEAKER,
    UNKNOWN_SPEAKER,
    SpeakerAttribution,
    attribute_span,
    merge_speaker_timeline,
    realign_speakers_with_punctuation,
)


def _seg(start, end, speaker):
    return SpeakerSegment(start=start, end=end, speaker=speaker)


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
# merge_speaker_timeline
# ---------------------------------------------------------------------------


def test_adjacent_same_speaker_entries_are_coalesced():
    merged = merge_speaker_timeline([_seg(0.0, 1.0, 1), _seg(1.1, 2.0, 1)])
    assert merged == [_seg(0.0, 2.0, 1)]


def test_same_speaker_across_a_long_gap_is_not_coalesced():
    """A long silence must not become one entry that outweighs the gap."""
    merged = merge_speaker_timeline([_seg(0.0, 1.0, 1), _seg(60.0, 61.0, 1)])
    assert merged == [_seg(0.0, 1.0, 1), _seg(60.0, 61.0, 1)]


def test_merge_gap_is_configurable():
    timeline = [_seg(0.0, 1.0, 1), _seg(3.0, 4.0, 1)]
    assert len(merge_speaker_timeline(timeline, max_gap_seconds=0.5)) == 2
    assert len(merge_speaker_timeline(timeline, max_gap_seconds=2.5)) == 1


def test_interleaved_speakers_do_not_block_a_merge():
    """Grouping is per speaker, so an interleaved speaker no longer splits a run."""
    merged = merge_speaker_timeline([_seg(0.0, 1.0, 1), _seg(1.0, 1.2, 2), _seg(1.2, 2.0, 1)])
    assert _seg(0.0, 2.0, 1) in merged
    assert _seg(1.0, 1.2, 2) in merged


def test_zero_length_entries_are_dropped():
    assert merge_speaker_timeline([_seg(1.0, 1.0, 1)]) == []


# ---------------------------------------------------------------------------
# attribute_span
# ---------------------------------------------------------------------------


def test_no_timeline_falls_back_to_the_single_speaker_label():
    """An ASR-Only Server has no diarization at all, not an unknown speaker."""
    attribution = attribute_span(0.0, 1.0, [])
    assert attribution.speaker == NO_DIARIZATION_SPEAKER
    assert attribution.overlap is False


def test_span_in_a_diarization_gap_is_unknown():
    merged = merge_speaker_timeline([_seg(0.0, 1.0, 2), _seg(10.0, 11.0, 3)])
    assert attribute_span(4.0, 5.0, merged).speaker == UNKNOWN_SPEAKER


def test_span_past_the_diarization_horizon_is_unknown():
    merged = merge_speaker_timeline([_seg(0.0, 1.0, 2)])
    assert attribute_span(5.0, 6.0, merged).speaker == UNKNOWN_SPEAKER


def test_span_before_the_timeline_is_unknown():
    merged = merge_speaker_timeline([_seg(10.0, 11.0, 2)])
    assert attribute_span(0.0, 1.0, merged).speaker == UNKNOWN_SPEAKER


def test_maximum_overlap_wins():
    merged = merge_speaker_timeline([_seg(0.0, 1.0, 2), _seg(1.0, 5.0, 3)])
    assert attribute_span(0.8, 2.0, merged).speaker == 3


def test_ties_resolve_to_the_lowest_speaker_id():
    merged = merge_speaker_timeline([_seg(0.0, 1.0, 3), _seg(1.0, 2.0, 2)])
    assert attribute_span(0.0, 2.0, merged).speaker == 2


# ---------------------------------------------------------------------------
# Overlapped speech
# ---------------------------------------------------------------------------


def test_concurrent_speakers_flag_the_span_as_overlapped():
    merged = merge_speaker_timeline([_seg(0.0, 2.0, 1), _seg(1.0, 3.0, 2)])
    attribution = attribute_span(1.2, 1.8, merged)
    assert attribution.overlap is True


def test_clean_speaker_turn_is_not_reported_as_overlap():
    """Straddling a turn boundary is not overlapped speech."""
    merged = merge_speaker_timeline([_seg(0.0, 1.0, 1), _seg(1.0, 2.0, 2)])
    assert attribute_span(0.9, 1.1, merged).overlap is False


def test_overlap_outside_the_span_is_not_reported():
    merged = merge_speaker_timeline([_seg(0.0, 5.0, 1), _seg(4.0, 6.0, 2)])
    assert attribute_span(0.0, 1.0, merged).overlap is False


# ---------------------------------------------------------------------------
# realign_speakers_with_punctuation
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty():
    assert realign_speakers_with_punctuation([]) == []


def test_homogeneous_sentence_is_unchanged():
    attributed = _attributed((0.0, 0.4, " todo", 1), (0.4, 0.8, " igual.", 1))
    assert _speakers(realign_speakers_with_punctuation(attributed)) == [1, 1]


def test_isolated_flicker_is_corrected_to_the_surrounding_majority():
    """A single-word blip surrounded by a stable speaker is noise, not a turn."""
    attributed = _attributed(
        (0.0, 1.0, " esto", 1),
        (1.0, 2.0, " es", 1),
        (2.0, 2.2, " una", 2),  # 0.2s flicker
        (2.2, 3.0, " prueba.", 1),
    )
    realigned = realign_speakers_with_punctuation(attributed)
    assert _speakers(realigned) == [1, 1, 1, 1]


def test_majority_vote_is_weighted_by_duration_not_word_count():
    """Three short words must not outvote one long word from another speaker."""
    attributed = _attributed(
        (0.0, 0.05, " a", 1),
        (0.05, 0.10, " b", 1),
        (0.10, 0.15, " c", 1),
        (0.15, 5.15, " largapalabra.", 2),
    )
    realigned = realign_speakers_with_punctuation(attributed)
    assert _speakers(realigned) == [2, 2, 2, 2]


def test_tie_breaks_to_the_lowest_speaker_id():
    attributed = _attributed((0.0, 0.5, " uno", 3), (0.5, 1.0, " dos.", 2))
    realigned = realign_speakers_with_punctuation(attributed)
    assert _speakers(realigned) == [2, 2]


def test_sentence_final_punctuation_bounds_the_vote():
    """A real turn coinciding with a sentence boundary is never touched."""
    attributed = _attributed((0.0, 0.5, " hola.", 2), (0.5, 1.0, " adios.", 3))
    realigned = realign_speakers_with_punctuation(attributed)
    assert _speakers(realigned) == [2, 3]


def test_opening_mark_bounds_the_vote_and_stops_it_spilling_over():
    """A flicker before the mark must not drag the mark's own sentence along."""
    attributed = _attributed(
        (0.0, 0.4, " bien", 1),
        (0.4, 0.5, " gracias", 9),  # 0.1s flicker inside the first sentence
        (0.5, 0.9, " ¿y", 2),
        (0.9, 1.3, " tú?", 2),
    )
    realigned = realign_speakers_with_punctuation(attributed)
    assert _speakers(realigned) == [1, 1, 2, 2]


def test_unknown_speaker_never_enters_the_vote_and_is_never_relabelled():
    attributed = _attributed(
        (0.0, 0.1, " a", 1),
        (0.1, 5.1, " b", UNKNOWN_SPEAKER),  # huge duration, must not count
        (5.1, 5.2, " c", 2),
        (5.2, 5.4, " d.", 2),
    )
    realigned = realign_speakers_with_punctuation(attributed)
    assert _speakers(realigned) == [2, UNKNOWN_SPEAKER, 2, 2]


def test_sentence_that_is_entirely_unknown_is_unchanged():
    attributed = _attributed((0.0, 0.5, " a", UNKNOWN_SPEAKER), (0.5, 1.0, " b.", UNKNOWN_SPEAKER))
    realigned = realign_speakers_with_punctuation(attributed)
    assert _speakers(realigned) == [UNKNOWN_SPEAKER, UNKNOWN_SPEAKER]


def test_relabelling_preserves_the_overlap_flag():
    attributed = _attributed(
        (0.0, 1.0, " esto", 1, False),
        (1.0, 1.2, " una", 2, True),
        (1.2, 2.0, " prueba.", 1, False),
    )
    realigned = realign_speakers_with_punctuation(attributed)
    assert [a.overlap for _, a in realigned] == [False, True, False]
