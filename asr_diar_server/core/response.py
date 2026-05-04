"""Core response builder.

Accepts Project-Owned Transcript Model types and produces the enriched
transcription response dict.  No FastAPI or backend-native types are used.

Key behaviours:
- Groups tokens into punctuation-boundary segments.
- Assigns speakers from a diarization timeline using maximum-overlap rule.
- Clamps adjacent segment overlaps to guarantee non-overlapping output.
- Emits speaker='-1' for tokens beyond the last diarization entry.
- Builds transcript, diarization, and raw_words convenience fields.
"""

from __future__ import annotations

from asr_diar_server.core.types import (
    SpeakerSegment,
    TranscriptSegment,
    TranscriptToken,
    TranscriptWord,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SILENCE_SENTINEL = -2
_UNKNOWN_SPEAKER = -1


def _assign_speakers(
    segments: list[TranscriptSegment],
    speaker_timeline: list[SpeakerSegment],
) -> None:
    """Assign speaker labels to segments in-place using max-overlap rule.

    Segments beyond the last timeline entry receive speaker=-1.
    Segments with no matching timeline data receive speaker=1 (default).
    """
    if not speaker_timeline:
        for seg in segments:
            seg.speaker = 1
        return

    # Merge consecutive same-speaker entries.
    merged: list[SpeakerSegment] = []
    for item in sorted(speaker_timeline, key=lambda x: x.start):
        if merged and item.speaker == merged[-1].speaker:
            merged[-1] = SpeakerSegment(
                start=merged[-1].start,
                end=max(merged[-1].end, item.end),
                speaker=item.speaker,
            )
        else:
            merged.append(item)

    last_end = merged[-1].end

    for seg in segments:
        if seg.start >= last_end:
            seg.speaker = _UNKNOWN_SPEAKER
            continue
        max_overlap = 0.0
        best = 1
        for entry in merged:
            overlap = max(0.0, min(seg.end, entry.end) - max(seg.start, entry.start))
            if overlap > max_overlap:
                max_overlap = overlap
                best = entry.speaker
        seg.speaker = best


def _group_tokens_into_segments(tokens: list[TranscriptToken]) -> list[TranscriptSegment]:
    """Group tokens into segments at whitespace/punctuation boundaries.

    Each segment spans contiguous non-silence tokens up to a punctuation mark.
    """
    if not tokens:
        return []

    segments: list[TranscriptSegment] = []
    current_tokens: list[TranscriptToken] = []

    def _flush():
        if not current_tokens:
            return
        text = "".join(t.text for t in current_tokens)
        if not text.strip():
            current_tokens.clear()
            return
        start = min(t.start for t in current_tokens)
        end = max(t.end for t in current_tokens)
        if end < start:
            start, end = end, start
        segments.append(TranscriptSegment(start=start, end=end, text=text))
        current_tokens.clear()

    _PUNCTUATION = ".!?,"

    for token in tokens:
        if not token.text:
            continue
        current_tokens.append(token)
        stripped = token.text.rstrip()
        if stripped and stripped[-1] in _PUNCTUATION:
            _flush()

    _flush()
    return segments


def _clamp_overlaps(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """Clamp adjacent segment end times to eliminate overlapping ranges."""
    ordered = sorted(segments, key=lambda s: s.start)
    for i in range(len(ordered) - 1):
        current = ordered[i]
        nxt = ordered[i + 1]
        if current.end > nxt.start:
            current.end = max(current.start, nxt.start)
    return ordered


def _build_words_for_segment(seg: TranscriptSegment) -> list[TranscriptWord]:
    """Build linearly interpolated word-level timestamps for a segment."""
    raw_words = seg.text.split()
    if not raw_words:
        return []
    word_duration = (seg.end - seg.start) / len(raw_words)
    speaker_str = str(seg.speaker)
    words = []
    for j, word in enumerate(raw_words):
        words.append(
            TranscriptWord(
                word=word,
                start=round(seg.start + j * word_duration, 2),
                end=round(seg.start + (j + 1) * word_duration, 2),
                score=1.0,
                speaker=speaker_str,
            )
        )
    return words


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_transcription_response(
    tokens: list[TranscriptToken],
    speaker_timeline: list[SpeakerSegment],
    duration: float,
) -> dict:
    """Build a transcription response dict from project-owned types.

    Args:
        tokens: Ordered transcript tokens (Project-Owned Transcript Model).
        speaker_timeline: Speaker timeline segments from the Diarization Adapter.
        duration: Total audio duration in seconds.

    Returns:
        Dict with keys: segments, word_segments, transcript, diarization, raw_words.

    """
    if not tokens:
        diar = [
            {"start": round(s.start, 3), "end": round(s.end, 3), "speaker": str(s.speaker)}
            for s in sorted(speaker_timeline, key=lambda x: x.start)
        ]
        return {
            "segments": [],
            "word_segments": [],
            "transcript": [],
            "diarization": diar,
            "raw_words": [],
        }

    raw_words = [
        {
            "word": t.text,
            "start": round(t.start, 3),
            "end": round(t.end, 3),
            "score": float(t.probability) if t.probability is not None else 1.0,
        }
        for t in tokens
        if t.text and t.text.strip()
    ]

    # Group and clamp
    seg_objects = _group_tokens_into_segments(tokens)
    _assign_speakers(seg_objects, speaker_timeline)
    seg_objects = _clamp_overlaps(seg_objects)

    segments = []
    word_segments = []

    for seg in seg_objects:
        speaker_str = str(seg.speaker)
        words = _build_words_for_segment(seg)
        word_dicts = [
            {
                "word": w.word,
                "start": w.start,
                "end": w.end,
                "score": w.score,
                "speaker": w.speaker,
            }
            for w in words
        ]
        word_segments.extend(word_dicts)
        segments.append(
            {
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
                "speaker": speaker_str,
                "words": word_dicts,
            }
        )

    transcript = [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segments]
    diarization = [
        {"start": s["start"], "end": s["end"], "speaker": s["speaker"]} for s in segments
    ]

    return {
        "segments": segments,
        "word_segments": word_segments,
        "transcript": transcript,
        "diarization": diarization,
        "raw_words": raw_words,
    }
