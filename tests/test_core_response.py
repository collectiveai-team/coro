"""Cycle 7: core response builder produces non-overlapping WhisperX response.

All inputs are Project-Owned Transcript Model types — no whisperlivekit
objects leak into this test.
"""

from __future__ import annotations

import pytest

from asr_diar_server.core.response import build_whisperx_response
from asr_diar_server.core.types import SpeakerSegment, TranscriptToken


# ---------------------------------------------------------------------------
# build_whisperx_response: basic shape
# ---------------------------------------------------------------------------


def test_empty_tokens_returns_whisperx_shape():
    """Empty token list returns all five WhisperX keys with empty lists."""
    result = build_whisperx_response(tokens=[], speaker_timeline=[], duration=5.0)
    assert set(result.keys()) == {
        "segments",
        "word_segments",
        "transcript",
        "diarization",
        "raw_words",
    }
    for key in ("segments", "word_segments", "transcript", "diarization", "raw_words"):
        assert result[key] == []


def test_single_token_produces_segment():
    """A single token produces one segment, one word_segment, one transcript entry, one raw_word."""
    tokens = [TranscriptToken(start=0.0, end=1.0, text=" hello", probability=0.9)]
    result = build_whisperx_response(tokens=tokens, speaker_timeline=[], duration=1.0)
    assert len(result["segments"]) == 1
    assert len(result["word_segments"]) >= 1
    assert len(result["transcript"]) == 1
    assert len(result["raw_words"]) == 1


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
    result = build_whisperx_response(tokens=tokens, speaker_timeline=[], duration=12.0)
    segs = result["segments"]
    assert len(segs) >= 2, f"Expected >=2 segments, got: {segs}"
    for i in range(len(segs) - 1):
        assert segs[i]["end"] <= segs[i + 1]["start"], "Segments must not overlap"


def test_speaker_attribution_from_timeline():
    """Tokens overlapping a speaker timeline entry get that speaker label."""
    tokens = [TranscriptToken(start=0.0, end=2.0, text=" hello", probability=1.0)]
    timeline = [SpeakerSegment(start=0.0, end=3.0, speaker=1)]
    result = build_whisperx_response(tokens=tokens, speaker_timeline=timeline, duration=3.0)
    seg = result["segments"][0]
    assert seg["speaker"] == "1"


def test_diarization_convenience_field():
    """diarization field contains start/end/speaker for each segment."""
    tokens = [TranscriptToken(start=0.0, end=1.0, text=" hi", probability=1.0)]
    result = build_whisperx_response(tokens=tokens, speaker_timeline=[], duration=1.0)
    for entry in result["diarization"]:
        assert {"start", "end", "speaker"}.issubset(entry.keys())


def test_transcript_convenience_field():
    """transcript field contains start/end/text for each segment."""
    tokens = [TranscriptToken(start=0.0, end=1.0, text=" hi", probability=1.0)]
    result = build_whisperx_response(tokens=tokens, speaker_timeline=[], duration=1.0)
    for entry in result["transcript"]:
        assert {"start", "end", "text"}.issubset(entry.keys())


def test_unknown_speaker_emitted_as_minus_one():
    """Tokens with no speaker attribution are emitted with speaker='-1'."""
    tokens = [TranscriptToken(start=5.0, end=6.0, text=" unattributed", probability=1.0)]
    # Speaker timeline ends before the token.
    timeline = [SpeakerSegment(start=0.0, end=2.0, speaker=1)]
    result = build_whisperx_response(tokens=tokens, speaker_timeline=timeline, duration=6.0)
    seg = result["segments"][0]
    assert seg["speaker"] == "-1"


def test_raw_words_contains_probability():
    """raw_words entries carry the token's probability score."""
    tokens = [TranscriptToken(start=0.0, end=1.0, text=" word", probability=0.75)]
    result = build_whisperx_response(tokens=tokens, speaker_timeline=[], duration=1.0)
    assert result["raw_words"][0]["score"] == pytest.approx(0.75)
