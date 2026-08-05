"""Speaker Boundary Split run detection and the Minimum Turn Threshold."""

from __future__ import annotations

from coro.core.models import SpeakerSegment
from coro.core.response import merge_speaker_timeline
from coro.core.segmentation import (
    MinimumTurnThreshold,
    speaker_at_instant,
    speaker_runs,
)

_DEFAULT = MinimumTurnThreshold()


def _runs(starts, span_end, timeline, threshold=_DEFAULT):
    merged = merge_speaker_timeline(timeline)
    last_end = merged[-1].end if merged else 0.0
    return [(r.speaker, r.start, r.end) for r in speaker_runs(starts, span_end, merged, last_end, threshold)]


class TestSpeakerAtInstant:
    def test_returns_the_speaker_covering_the_instant(self):
        merged = merge_speaker_timeline([SpeakerSegment(start=0.0, end=1.0, speaker=2)])
        assert speaker_at_instant(0.5, merged, 1.0) == 2

    def test_is_half_open_on_the_turn_boundary(self):
        """A word starting exactly at a turn edge belongs to the new turn."""
        merged = merge_speaker_timeline(
            [
                SpeakerSegment(start=0.0, end=1.0, speaker=2),
                SpeakerSegment(start=1.0, end=2.0, speaker=3),
            ]
        )
        assert speaker_at_instant(1.0, merged, 2.0) == 3

    def test_silence_between_turns_is_unlabelled(self):
        merged = merge_speaker_timeline(
            [
                SpeakerSegment(start=0.0, end=1.0, speaker=2),
                SpeakerSegment(start=5.0, end=6.0, speaker=3),
            ]
        )
        assert speaker_at_instant(3.0, merged, 6.0) is None

    def test_beyond_the_horizon_is_unknown(self):
        merged = merge_speaker_timeline([SpeakerSegment(start=0.0, end=1.0, speaker=2)])
        assert speaker_at_instant(9.0, merged, 1.0) == -1


class TestSpeakerRuns:
    def test_no_timeline_means_no_split(self):
        """Diarization is off by default; segmentation must be untouched."""
        assert _runs([0.0, 0.5, 1.0], 1.5, []) == []

    def test_single_speaker_segment_is_not_split(self):
        timeline = [SpeakerSegment(start=0.0, end=3.0, speaker=2)]
        assert _runs([0.0, 0.5, 1.0], 1.5, timeline) == []

    def test_turn_change_mid_segment_splits(self):
        """Six words, speaker change at the fourth."""
        timeline = [
            SpeakerSegment(start=0.0, end=1.5, speaker=2),
            SpeakerSegment(start=1.5, end=3.0, speaker=3),
        ]
        starts = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        assert _runs(starts, 3.0, timeline) == [(2, 0, 3), (3, 3, 6)]

    def test_turn_below_the_word_threshold_is_absorbed(self):
        """A one-word backchannel leaves its word with the surrounding speaker."""
        timeline = [
            SpeakerSegment(start=0.0, end=1.0, speaker=2),
            SpeakerSegment(start=1.0, end=1.6, speaker=3),
            SpeakerSegment(start=1.6, end=3.0, speaker=2),
        ]
        # Word 2 (start 1.0) is speaker 3 alone — one word, so not a turn.
        starts = [0.0, 0.5, 1.0, 1.6, 2.0, 2.5]
        assert _runs(starts, 3.0, timeline) == []

    def test_turn_below_the_duration_threshold_is_absorbed(self):
        """Two words spanning under 0.4 s is still not a turn."""
        timeline = [
            SpeakerSegment(start=0.0, end=1.0, speaker=2),
            SpeakerSegment(start=1.0, end=1.2, speaker=3),
            SpeakerSegment(start=1.2, end=3.0, speaker=2),
        ]
        starts = [0.0, 0.5, 1.0, 1.1, 1.2, 2.5]
        assert _runs(starts, 3.0, timeline) == []

    def test_segment_can_split_into_three(self):
        timeline = [
            SpeakerSegment(start=0.0, end=1.0, speaker=2),
            SpeakerSegment(start=1.0, end=2.0, speaker=3),
            SpeakerSegment(start=2.0, end=3.0, speaker=4),
        ]
        starts = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        assert _runs(starts, 3.0, timeline) == [(2, 0, 2), (3, 2, 4), (4, 4, 6)]

    def test_silence_inside_a_turn_does_not_split_it(self):
        """Words in a diarization gap continue the turn they follow."""
        timeline = [
            SpeakerSegment(start=0.0, end=1.0, speaker=2),
            SpeakerSegment(start=4.0, end=6.0, speaker=2),
        ]
        starts = [0.0, 0.5, 2.0, 4.0, 4.5, 5.0]
        assert _runs(starts, 6.0, timeline) == []

    def test_beyond_horizon_words_form_their_own_unknown_run(self):
        timeline = [SpeakerSegment(start=0.0, end=1.5, speaker=2)]
        starts = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        assert _runs(starts, 3.0, timeline) == [(2, 0, 3), (-1, 3, 6)]

    def test_threshold_is_configurable(self):
        """Raising the word bound suppresses a split that the default allows."""
        timeline = [
            SpeakerSegment(start=0.0, end=1.5, speaker=2),
            SpeakerSegment(start=1.5, end=3.0, speaker=3),
        ]
        starts = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        strict = MinimumTurnThreshold(words=4, seconds=0.4)
        assert _runs(starts, 3.0, timeline, strict) == []

    def test_a_single_word_segment_is_never_split(self):
        timeline = [SpeakerSegment(start=0.0, end=3.0, speaker=2)]
        assert _runs([0.0], 1.0, timeline) == []
