"""whisperlivekit ML Model Integration.

Provides pure conversion functions that translate whisperlivekit-native ASR
and diarization objects into Project-Owned Transcript Model types.

Conversion functions are pure (no I/O, no model calls) so they can be
tested in isolation with fake SimpleNamespace objects.

Public surface:
    convert_asr_segments   — native segment/word list → list[TranscriptToken]
    convert_diarization_segments — native diar list → list[SpeakerSegment]
"""

from __future__ import annotations

import re

from asr_diar_server.core.types import SpeakerSegment, TranscriptToken

_NO_SPEECH_THRESHOLD = 0.9


def convert_asr_segments(
    native_segments,
    *,
    offset_seconds: float = 0.0,
) -> list[TranscriptToken]:
    """Convert faster-whisper/whisperlivekit segment objects to TranscriptTokens.

    Args:
        native_segments: Iterable of native segment objects with ``.words`` and
            ``.no_speech_prob`` attributes.
        offset_seconds: Timestamp offset to add to each word's start/end.

    Returns:
        List of TranscriptToken sorted by start time.

    """
    tokens: list[TranscriptToken] = []

    for seg in native_segments:
        if getattr(seg, "no_speech_prob", 0.0) > _NO_SPEECH_THRESHOLD:
            continue
        for word in getattr(seg, "words", []):
            start = round(float(getattr(word, "start", 0.0)) + offset_seconds, 3)
            end = round(float(getattr(word, "end", 0.0)) + offset_seconds, 3)
            text = getattr(word, "word", getattr(word, "text", ""))
            probability = getattr(word, "probability", None)
            tokens.append(TranscriptToken(start=start, end=end, text=text, probability=probability))

    return tokens


def _speaker_to_one_indexed(speaker) -> int:
    """Convert a zero-indexed or string speaker label to 1-indexed int.

    Args:
        speaker: int (0-indexed) or string like "SPEAKER_00" or "0".

    Returns:
        1-indexed integer speaker label.

    """
    if isinstance(speaker, int):
        return speaker + 1
    # numpy integer types
    try:
        import numpy as np

        if isinstance(speaker, np.integer):
            return int(speaker) + 1
    except ImportError:
        pass
    # String: extract first integer and add 1
    match = re.search(r"\d+", str(speaker))
    if match:
        return int(match.group(0)) + 1
    return 1


def convert_diarization_segments(
    native_segments,
    *,
    duration: float,
) -> list[SpeakerSegment]:
    """Convert whisperlivekit diarization segment objects to SpeakerSegments.

    Args:
        native_segments: Iterable of native diarization segment objects with
            ``.start``, ``.end``, and ``.speaker`` attributes.
        duration: Total audio duration in seconds; end times are clamped.

    Returns:
        Deduplicated list of SpeakerSegment sorted by start time.

    """
    timeline: list[SpeakerSegment] = []
    seen: set = set()

    for seg in native_segments:
        start = max(0.0, float(getattr(seg, "start", 0.0) or 0.0))
        end = min(duration, float(getattr(seg, "end", 0.0) or 0.0))
        if end <= start:
            continue
        speaker = _speaker_to_one_indexed(getattr(seg, "speaker", 0))
        key = (round(start, 3), round(end, 3), speaker)
        if key in seen:
            continue
        seen.add(key)
        timeline.append(SpeakerSegment(start=round(start, 3), end=round(end, 3), speaker=speaker))

    timeline.sort(key=lambda s: s.start)
    return timeline
