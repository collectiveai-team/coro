"""Cycle 7: core response builder produces non-overlapping transcription response.

All inputs are Project-Owned Transcript Model types — no backend-native types
objects leak into this test.
"""

from __future__ import annotations

import pytest

from coro.core.response import build_transcription_response
from coro.core.models import (
    DiarizationItem,
    SpeakerSegment,
    TranscriptItem,
    TranscriptToken,
    TranscriptWord,
)


# ---------------------------------------------------------------------------
# build_transcription_response: basic shape
# ---------------------------------------------------------------------------


def test_empty_tokens_returns_response_shape():
    """Empty token list returns all five response fields with empty lists."""
    result = build_transcription_response(tokens=[], speaker_timeline=[], duration=5.0)
    assert result.segments == []
    assert result.word_segments == []
    assert result.transcript == []
    assert result.diarization == []
    assert result.raw_words == []


def test_single_token_produces_segment():
    """A single token produces one segment, one word_segment, one transcript entry, one raw_word."""
    tokens = [TranscriptToken(start=0.0, end=1.0, text=" hello", probability=0.9)]
    result = build_transcription_response(tokens=tokens, speaker_timeline=[], duration=1.0)
    assert len(result.segments) == 1
    assert len(result.word_segments) >= 1
    assert len(result.transcript) == 1
    assert len(result.raw_words) == 1


def test_adjacent_overlap_is_clamped():
    """Adjacent segments with overlapping times are clamped to non-overlapping ranges.

    Two punctuation-terminated tokens produce distinct segments; the first
    has end > second's start (overlap) which should be clamped.
    """
    # Punctuation terminates each token → two distinct segments.
    tokens = [
        TranscriptToken(start=0.0, end=10.0, text=" first.", probability=1.0),
        TranscriptToken(start=9.0, end=12.0, text=" second.", probability=1.0),
    ]
    result = build_transcription_response(tokens=tokens, speaker_timeline=[], duration=12.0)
    segs = result.segments
    assert len(segs) >= 2, f"Expected >=2 segments, got: {segs}"
    for i in range(len(segs) - 1):
        assert segs[i].end <= segs[i + 1].start, "Segments must not overlap"


def test_speaker_attribution_from_timeline():
    """Tokens overlapping a speaker timeline entry get that speaker label."""
    tokens = [TranscriptToken(start=0.0, end=2.0, text=" hello", probability=1.0)]
    timeline = [SpeakerSegment(start=0.0, end=3.0, speaker=1)]
    result = build_transcription_response(tokens=tokens, speaker_timeline=timeline, duration=3.0)
    seg = result.segments[0]
    assert seg.speaker == "1"


def test_diarization_convenience_field():
    """diarization field contains start/end/speaker for each segment."""
    tokens = [TranscriptToken(start=0.0, end=1.0, text=" hi", probability=1.0)]
    result = build_transcription_response(tokens=tokens, speaker_timeline=[], duration=1.0)
    for entry in result.diarization:
        assert isinstance(entry, DiarizationItem)
        assert isinstance(entry.start, float) and isinstance(entry.end, float)
        assert isinstance(entry.speaker, str)


def test_transcript_convenience_field():
    """transcript field contains start/end/text for each segment."""
    tokens = [TranscriptToken(start=0.0, end=1.0, text=" hi", probability=1.0)]
    result = build_transcription_response(tokens=tokens, speaker_timeline=[], duration=1.0)
    for entry in result.transcript:
        assert isinstance(entry, TranscriptItem)
        assert isinstance(entry.start, float) and isinstance(entry.end, float)
        assert isinstance(entry.text, str)


def test_unknown_speaker_emitted_as_minus_one():
    """Tokens with no speaker attribution are emitted with speaker='-1'."""
    tokens = [TranscriptToken(start=5.0, end=6.0, text=" unattributed", probability=1.0)]
    # Speaker timeline ends before the token.
    timeline = [SpeakerSegment(start=0.0, end=2.0, speaker=1)]
    result = build_transcription_response(tokens=tokens, speaker_timeline=timeline, duration=6.0)
    seg = result.segments[0]
    assert seg.speaker == "-1"


def test_raw_words_contains_probability():
    """raw_words entries carry the token's probability score."""
    tokens = [TranscriptToken(start=0.0, end=1.0, text=" word", probability=0.75)]
    result = build_transcription_response(tokens=tokens, speaker_timeline=[], duration=1.0)
    assert result.raw_words[0].score == pytest.approx(0.75)


def test_segment_words_are_typed_and_speaker_attributed():
    """Segment words are TranscriptWord values carrying the segment's speaker."""
    tokens = [
        TranscriptToken(start=0.0, end=1.0, text=" hello", probability=1.0),
        TranscriptToken(start=1.0, end=2.0, text=" world", probability=1.0),
    ]
    result = build_transcription_response(tokens=tokens, speaker_timeline=[], duration=2.0)
    words = result.segments[0].words
    assert len(words) == 2
    assert all(isinstance(w, TranscriptWord) for w in words)
    assert words[0].word == "hello"
    assert words[0].speaker == "1"
    assert isinstance(words[0].start, float)


# ---------------------------------------------------------------------------
# Real word timings and confidences
# ---------------------------------------------------------------------------


def test_words_carry_the_backends_real_timings_not_interpolation():
    """Uneven word timings survive; they are not spread evenly over the segment."""
    tokens = [
        TranscriptToken(start=0.0, end=0.2, text=" muy", probability=0.9),
        TranscriptToken(start=3.0, end=4.0, text=" tarde.", probability=0.4),
    ]
    result = build_transcription_response(tokens=tokens, speaker_timeline=[], duration=4.0)
    words = result.segments[0].words
    assert [(w.start, w.end) for w in words] == [(0.0, 0.2), (3.0, 4.0)]


def test_words_carry_the_backends_real_confidence():
    """Word score is the token probability, not a constant 1.0."""
    tokens = [
        TranscriptToken(start=0.0, end=0.5, text=" hola", probability=0.62),
        TranscriptToken(start=0.5, end=1.0, text=" mundo.", probability=0.31),
    ]
    result = build_transcription_response(tokens=tokens, speaker_timeline=[], duration=1.0)
    assert [w.score for w in result.segments[0].words] == pytest.approx([0.62, 0.31])


def test_missing_probability_defaults_to_one():
    tokens = [TranscriptToken(start=0.0, end=0.5, text=" hola.", probability=None)]
    result = build_transcription_response(tokens=tokens, speaker_timeline=[], duration=0.5)
    assert result.segments[0].words[0].score == 1.0


# ---------------------------------------------------------------------------
# Word-level speaker attribution
# ---------------------------------------------------------------------------


def test_segment_splits_where_the_word_level_speaker_changes():
    """A sentence spanning a speaker turn no longer inherits one label.

    The turn coincides with sentence-final punctuation, so it is a real turn
    and punctuation-aware realignment (see ``test_core_speakers.py``) leaves
    it untouched; an unpunctuated speaker change within a single sentence is
    instead flicker-corrected.
    """
    tokens = [
        TranscriptToken(start=0.0, end=0.4, text=" hola", probability=1.0),
        TranscriptToken(start=0.4, end=0.8, text=" mundo.", probability=1.0),
        TranscriptToken(start=2.0, end=2.4, text=" adios", probability=1.0),
        TranscriptToken(start=2.4, end=2.8, text=" amigo.", probability=1.0),
    ]
    timeline = [
        SpeakerSegment(start=0.0, end=1.0, speaker=2),
        SpeakerSegment(start=1.5, end=3.0, speaker=3),
    ]
    result = build_transcription_response(tokens=tokens, speaker_timeline=timeline, duration=3.0)

    assert [s.speaker for s in result.segments] == ["2", "3"]
    assert [s.text for s in result.segments] == ["hola mundo.", "adios amigo."]
    assert [w.speaker for w in result.word_segments] == ["2", "2", "3", "3"]


def test_single_word_flicker_within_a_sentence_does_not_split_the_segment():
    """Punctuation-aware realignment absorbs a one-word diarization blip."""
    tokens = [
        TranscriptToken(start=0.0, end=1.0, text=" esto", probability=1.0),
        TranscriptToken(start=1.0, end=2.0, text=" es", probability=1.0),
        TranscriptToken(start=2.0, end=2.2, text=" una", probability=1.0),
        TranscriptToken(start=2.2, end=3.0, text=" prueba.", probability=1.0),
    ]
    timeline = [
        SpeakerSegment(start=0.0, end=2.0, speaker=1),
        SpeakerSegment(start=2.0, end=2.2, speaker=2),
        SpeakerSegment(start=2.2, end=3.0, speaker=1),
    ]
    result = build_transcription_response(tokens=tokens, speaker_timeline=timeline, duration=3.0)
    assert [s.speaker for s in result.segments] == ["1"]
    assert [w.speaker for w in result.word_segments] == ["1", "1", "1", "1"]


def test_word_in_a_diarization_gap_is_marked_unknown():
    """Words with no timeline support are unknown, not the first speaker."""
    tokens = [
        TranscriptToken(start=0.0, end=0.4, text=" dentro", probability=1.0),
        TranscriptToken(start=5.0, end=5.4, text=" fuera.", probability=1.0),
    ]
    timeline = [
        SpeakerSegment(start=0.0, end=1.0, speaker=2),
        SpeakerSegment(start=8.0, end=9.0, speaker=3),
    ]
    result = build_transcription_response(tokens=tokens, speaker_timeline=timeline, duration=9.0)
    assert [s.speaker for s in result.segments] == ["2", "-1"]


def test_same_speaker_across_a_long_silence_does_not_swallow_the_gap():
    """Gap-bounded merging keeps a mid-gap word attributable to its own speaker."""
    tokens = [TranscriptToken(start=30.0, end=30.5, text=" medio.", probability=1.0)]
    timeline = [
        SpeakerSegment(start=0.0, end=1.0, speaker=1),
        SpeakerSegment(start=29.0, end=31.0, speaker=2),
        SpeakerSegment(start=59.0, end=60.0, speaker=1),
    ]
    result = build_transcription_response(tokens=tokens, speaker_timeline=timeline, duration=60.0)
    assert result.segments[0].speaker == "2"


# ---------------------------------------------------------------------------
# Overlapped speech
# ---------------------------------------------------------------------------


def test_overlapped_speech_is_flagged_on_words_and_segments():
    tokens = [TranscriptToken(start=1.2, end=1.8, text=" cruzado.", probability=1.0)]
    timeline = [
        SpeakerSegment(start=0.0, end=2.0, speaker=1),
        SpeakerSegment(start=1.0, end=3.0, speaker=2),
    ]
    result = build_transcription_response(tokens=tokens, speaker_timeline=timeline, duration=3.0)
    assert result.segments[0].overlap is True
    assert result.segments[0].words[0].overlap is True


def test_non_overlapped_speech_is_not_flagged():
    tokens = [TranscriptToken(start=0.0, end=0.5, text=" limpio.", probability=1.0)]
    timeline = [SpeakerSegment(start=0.0, end=2.0, speaker=1)]
    result = build_transcription_response(tokens=tokens, speaker_timeline=timeline, duration=2.0)
    assert result.segments[0].overlap is False
    assert result.segments[0].words[0].overlap is False


# ---------------------------------------------------------------------------
# Segmentation policy at the response level
# ---------------------------------------------------------------------------


def test_comma_does_not_split_a_response_segment():
    tokens = [
        TranscriptToken(start=0.0, end=0.4, text=" cuando,", probability=1.0),
        TranscriptToken(start=0.4, end=0.8, text=" llegue.", probability=1.0),
    ]
    result = build_transcription_response(tokens=tokens, speaker_timeline=[], duration=0.8)
    assert [s.text for s in result.segments] == ["cuando, llegue."]
