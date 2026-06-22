"""Cycle 11: backend adapter conversion using fake backend-native objects.

Tests verify that the backend adapters:
- Convert native word/segment objects into Project-Owned TranscriptToken types.
- Convert native diarization segment objects into SpeakerSegment types.
- Filter no-speech segments (no_speech_prob > 0.9).
- Apply offset_seconds correctly to timestamps.
- Convert speaker labels to 1-indexed integers.

No real model inference is performed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from coro.backends.asr.faster_whisper import convert_asr_segments
from coro.backends.diarization.segments import convert_diarization_segments
from coro.core.models import SpeakerSegment, TranscriptToken


def _fake_word(word: str, start: float, end: float, probability: float = 0.9):
    return SimpleNamespace(word=word, start=start, end=end, probability=probability)


def _fake_segment(words, no_speech_prob: float = 0.0):
    return SimpleNamespace(words=words, no_speech_prob=no_speech_prob)


def _fake_diar_seg(start: float, end: float, speaker):
    return SimpleNamespace(start=start, end=end, speaker=speaker)


# ---------------------------------------------------------------------------
# convert_asr_segments
# ---------------------------------------------------------------------------


def test_convert_asr_segments_basic():
    """convert_asr_segments returns TranscriptToken list from fake native segments."""
    words = [_fake_word(" hello", 0.0, 0.5), _fake_word(" world", 0.5, 1.0)]
    segs = [_fake_segment(words)]
    tokens = convert_asr_segments(segs, offset_seconds=0.0)
    assert len(tokens) == 2
    assert all(isinstance(t, TranscriptToken) for t in tokens)


def test_convert_asr_segments_applies_offset():
    """Timestamps are shifted by offset_seconds."""
    words = [_fake_word(" hi", 1.0, 2.0)]
    segs = [_fake_segment(words)]
    tokens = convert_asr_segments(segs, offset_seconds=10.0)
    assert tokens[0].start == pytest.approx(11.0)
    assert tokens[0].end == pytest.approx(12.0)


def test_convert_asr_segments_filters_no_speech():
    """Segments with no_speech_prob > 0.9 are excluded."""
    words = [_fake_word(" noise", 0.0, 1.0)]
    segs = [_fake_segment(words, no_speech_prob=0.95)]
    tokens = convert_asr_segments(segs, offset_seconds=0.0)
    assert tokens == []


def test_convert_asr_segments_keeps_probability():
    """Token probability matches the native word probability."""
    words = [_fake_word(" test", 0.0, 1.0, probability=0.75)]
    tokens = convert_asr_segments([_fake_segment(words)], offset_seconds=0.0)
    assert tokens[0].probability == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# convert_diarization_segments
# ---------------------------------------------------------------------------


def test_convert_diarization_segments_basic():
    """convert_diarization_segments returns SpeakerSegment list."""
    segs = [_fake_diar_seg(0.0, 1.0, speaker=0), _fake_diar_seg(1.0, 2.0, speaker=1)]
    timeline = convert_diarization_segments(segs, duration=2.0)
    assert len(timeline) == 2
    assert all(isinstance(s, SpeakerSegment) for s in timeline)


def test_convert_diarization_segments_one_indexed():
    """Speaker labels are converted to 1-indexed integers."""
    segs = [_fake_diar_seg(0.0, 1.0, speaker=0)]
    timeline = convert_diarization_segments(segs, duration=2.0)
    assert timeline[0].speaker == 1


def test_convert_diarization_segments_string_speaker():
    """String speaker labels like 'SPEAKER_00' are converted correctly."""
    segs = [_fake_diar_seg(0.0, 1.0, speaker="SPEAKER_02")]
    timeline = convert_diarization_segments(segs, duration=2.0)
    assert timeline[0].speaker == 3  # 0-indexed digit 2 → +1 = 3


def test_convert_diarization_segments_end_clamped_to_duration():
    """Segment end times are clamped to the audio duration."""
    segs = [_fake_diar_seg(0.5, 5.0, speaker=0)]
    timeline = convert_diarization_segments(segs, duration=3.0)
    assert timeline[0].end <= 3.0
