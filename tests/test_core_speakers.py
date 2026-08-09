"""Speaker timeline merging and span attribution rules."""

from __future__ import annotations

from coro.core.models import SpeakerSegment
from coro.core.speakers import (
    NO_DIARIZATION_SPEAKER,
    UNKNOWN_SPEAKER,
    attribute_span,
    merge_speaker_timeline,
)


def _seg(start, end, speaker):
    return SpeakerSegment(start=start, end=end, speaker=speaker)


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
