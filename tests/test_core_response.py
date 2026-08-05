"""Cycle 7: core response builder produces non-overlapping transcription response.

All inputs are Project-Owned Transcript Model types — no backend-native types
objects leak into this test.
"""

from __future__ import annotations

import pytest

from coro.core.response import _build_words_for_segment, build_transcription_response
from coro.core.models import (
    DiarizationItem,
    SpeakerSegment,
    TranscriptItem,
    TranscriptSegment,
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


def test_build_words_for_segment_returns_typed_words():
    """_build_words_for_segment returns list[TranscriptWord], not list[dict]."""
    seg = TranscriptSegment(
        start=0.0,
        end=2.0,
        text="hello world",
        speaker=1,
        tokens=[
            TranscriptToken(start=0.0, end=1.0, text=" hello", probability=0.9),
            TranscriptToken(start=1.0, end=2.0, text=" world", probability=0.8),
        ],
    )
    words = _build_words_for_segment(seg)
    assert len(words) == 2
    assert all(isinstance(w, TranscriptWord) for w in words)
    assert words[0].word == "hello"
    assert words[0].speaker == "1"
    assert isinstance(words[0].start, float)


# ---------------------------------------------------------------------------
# Measured Word Start (issue 06)
# ---------------------------------------------------------------------------


def test_response_word_starts_are_the_token_emission_times():
    """Word starts equal the ASR's token times, not an even division of the span.

    Interpolation would place three words in a 9 s segment at 0.0/3.0/6.0.
    The tokens were actually emitted at 0.0/0.4/8.0, and that is what a
    Speaker Boundary Split has to cut on.
    """
    tokens = [
        TranscriptToken(start=0.0, end=0.4, text=" alpha", probability=0.9),
        TranscriptToken(start=0.4, end=8.0, text=" beta", probability=0.9),
        TranscriptToken(start=8.0, end=9.0, text=" gamma.", probability=0.9),
    ]
    result = build_transcription_response(tokens=tokens, speaker_timeline=[], duration=9.0)

    words = result.segments[0].words
    assert [w.word for w in words] == ["alpha", "beta", "gamma."]
    assert [w.start for w in words] == [0.0, 0.4, 8.0]
    assert [w.end for w in words] == [0.4, 8.0, 9.0]


def test_word_confidence_comes_from_the_token_probability():
    """Confidence is the backend's own per-word probability."""
    tokens = [TranscriptToken(start=0.0, end=1.0, text=" word.", probability=0.75)]
    result = build_transcription_response(tokens=tokens, speaker_timeline=[], duration=1.0)

    assert result.segments[0].words[0].score == pytest.approx(0.75)


def test_word_confidence_is_absent_when_the_backend_supplies_none():
    """A backend without per-word probabilities yields no confidence, not 1.0.

    onnx-genai and the onnx-asr text-only fallback emit tokens with
    ``probability=None``. Reporting 1.0 there asserts perfect confidence the
    model never expressed.
    """
    tokens = [TranscriptToken(start=0.0, end=1.0, text=" word.", probability=None)]
    result = build_transcription_response(tokens=tokens, speaker_timeline=[], duration=1.0)

    assert result.segments[0].words[0].score is None
    assert result.raw_words[0].score is None


def test_segment_without_tokens_has_no_words():
    """No tokens means no word timings — never fabricated ones."""
    seg = TranscriptSegment(start=0.0, end=2.0, text="hello world", speaker=1)

    assert _build_words_for_segment(seg) == []
