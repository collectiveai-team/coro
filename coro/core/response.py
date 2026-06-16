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

from coro.core.types import (
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
_PUNCTUATION = ".!?,"


def closes_segment(text: str) -> bool:
    """Return True if a token's text ends on a punctuation boundary."""
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in _PUNCTUATION


def segment_span_from_tokens(
    tokens: list[TranscriptToken],
) -> tuple[float, float, str] | None:
    """Collapse a run of tokens into a ``(start, end, text)`` span.

    Returns ``None`` when the run is empty or whitespace-only.  Mirrors the
    flush logic in :func:`_group_tokens_into_segments` so streaming and batch
    grouping stay identical.
    """
    if not tokens:
        return None
    text = "".join(t.text for t in tokens)
    if not text.strip():
        return None
    start = min(t.start for t in tokens)
    end = max(t.end for t in tokens)
    if end < start:
        start, end = end, start
    return start, end, text


def merge_speaker_timeline(
    speaker_timeline: list[SpeakerSegment],
) -> list[SpeakerSegment]:
    """Merge consecutive same-speaker entries into a sorted, coalesced timeline."""
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
    return merged


def speaker_for_span(
    start: float,
    end: float,
    merged: list[SpeakerSegment],
    last_end: float,
) -> int:
    """Return the max-overlap speaker for a ``[start, end)`` span.

    Spans starting at or beyond ``last_end`` (the diarization horizon) receive
    speaker=-1; spans with no overlap default to speaker=1.  ``merged`` must be
    the output of :func:`merge_speaker_timeline`.
    """
    if not merged:
        return 1
    if start >= last_end:
        return _UNKNOWN_SPEAKER
    max_overlap = 0.0
    best = 1
    for entry in merged:
        overlap = max(0.0, min(end, entry.end) - max(start, entry.start))
        if overlap > max_overlap:
            max_overlap = overlap
            best = entry.speaker
    return best


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

    merged = merge_speaker_timeline(speaker_timeline)
    last_end = merged[-1].end
    for seg in segments:
        seg.speaker = speaker_for_span(seg.start, seg.end, merged, last_end)


def _group_tokens_into_segments(tokens: list[TranscriptToken]) -> list[TranscriptSegment]:
    """Group tokens into segments at whitespace/punctuation boundaries.

    Each segment spans contiguous non-silence tokens up to a punctuation mark.
    """
    if not tokens:
        return []

    segments: list[TranscriptSegment] = []
    current_tokens: list[TranscriptToken] = []

    def _flush():
        span = segment_span_from_tokens(current_tokens)
        if span is not None:
            segments.append(TranscriptSegment(start=span[0], end=span[1], text=span[2]))
        current_tokens.clear()

    for token in tokens:
        if not token.text:
            continue
        current_tokens.append(token)
        if closes_segment(token.text):
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


def build_segment_dict(seg: TranscriptSegment) -> dict:
    """Serialise one speaker-attributed segment into the response shape.

    Produces ``{start, end, text, speaker, words}`` with interpolated word
    timings.  Shared by the batch builder and the streaming finalizer so both
    paths emit byte-identical segment dicts.
    """
    words = _build_words_for_segment(seg)
    word_dicts = [
        {"word": w.word, "start": w.start, "end": w.end, "score": w.score, "speaker": w.speaker}
        for w in words
    ]
    return {
        "start": round(seg.start, 2),
        "end": round(seg.end, 2),
        "text": seg.text.strip(),
        "speaker": str(seg.speaker),
        "words": word_dicts,
    }


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
        seg_dict = build_segment_dict(seg)
        word_segments.extend(seg_dict["words"])
        segments.append(seg_dict)

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
