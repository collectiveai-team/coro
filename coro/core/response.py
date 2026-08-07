"""Core response builder.

Accepts Project-Owned Transcript Model types and produces the enriched
:class:`TranscriptionResult`.  No FastAPI or backend-native types are used.

Key behaviours:
- Groups tokens into punctuation-boundary segments.
- Assigns speakers from a diarization timeline using maximum-overlap rule.
- Clamps adjacent segment overlaps to guarantee non-overlapping output.
- Emits speaker='-1' for tokens beyond the last diarization entry.
- Builds transcript, diarization, and raw_words convenience fields.
"""

from __future__ import annotations

from coro.core.models import (
    DiarizationItem,
    RawWord,
    ResponseSegment,
    SpeakerSegment,
    TranscriptionResult,
    TranscriptItem,
    TranscriptSegment,
    TranscriptToken,
    TranscriptWord,
)
from coro.core.segmentation import MinimumTurnThreshold, speaker_runs

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
            segments.append(
                TranscriptSegment(
                    start=span[0],
                    end=span[1],
                    text=span[2],
                    tokens=list(current_tokens),
                )
            )
        current_tokens.clear()

    for token in tokens:
        if not token.text:
            continue
        current_tokens.append(token)
        if closes_segment(token.text):
            _flush()

    _flush()
    return segments


def _split_segment_on_speakers(
    seg: TranscriptSegment,
    merged: list[SpeakerSegment],
    last_end: float,
    threshold: MinimumTurnThreshold,
) -> list[TranscriptSegment]:
    """Cut one segment at its speaker changes, using Measured Word Start values."""
    starts = [t.start for t in seg.tokens]
    runs = speaker_runs(starts, seg.end, merged, last_end, threshold)
    if not runs:
        return [seg]

    pieces: list[TranscriptSegment] = []
    for run in runs:
        tokens = seg.tokens[run.start : run.end]
        words = [t.text.strip() for t in tokens if t.text.strip()]
        if not words:
            continue
        end = starts[run.end] if run.end < len(starts) else seg.end
        pieces.append(
            TranscriptSegment(
                start=starts[run.start],
                end=end,
                text=" ".join(words),
                tokens=tokens,
            )
        )
    return pieces or [seg]


def _split_on_speaker_boundaries(
    segments: list[TranscriptSegment],
    speaker_timeline: list[SpeakerSegment],
    threshold: MinimumTurnThreshold,
) -> list[TranscriptSegment]:
    """Apply the Speaker Boundary Split across every segment.

    With no timeline — the default, diarization off — nothing is split and the
    output is byte-identical to punctuation-only segmentation.
    """
    if not speaker_timeline:
        return segments
    merged = merge_speaker_timeline(speaker_timeline)
    last_end = merged[-1].end
    split: list[TranscriptSegment] = []
    for seg in segments:
        split.extend(_split_segment_on_speakers(seg, merged, last_end, threshold))
    return split


def _clamp_overlaps(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """Clamp adjacent segment end times to eliminate overlapping ranges."""
    ordered = sorted(segments, key=lambda s: s.start)
    for i in range(len(ordered) - 1):
        current = ordered[i]
        nxt = ordered[i + 1]
        if current.end > nxt.start:
            current.end = max(current.start, nxt.start)
    return ordered


def build_segment(seg: TranscriptSegment) -> ResponseSegment:
    """Build one speaker-attributed :class:`ResponseSegment` from a segment.

    Produces ``start, end, text, speaker, words`` with **Measured Word Start**
    values taken from ``seg.tokens``.  Shared by the batch builder and the
    streaming finalizer so both paths emit byte-identical segments.
    """
    return ResponseSegment(
        start=round(seg.start, 2),
        end=round(seg.end, 2),
        text=seg.text.strip(),
        speaker=str(seg.speaker),
        words=_build_words_for_segment(seg),
    )


def _build_words_for_segment(seg: TranscriptSegment) -> list[TranscriptWord]:
    """Build word-level timings for a segment from the tokens that formed it.

    Starts are the ASR's own token emission times. Ends stay as the adapters
    already derive them — the following token's start — because no consumer
    reads word ends and the TDT duration that would give a true one is
    discarded inside ``onnx-asr`` (ADR 0008).

    A segment carrying no tokens yields no words. Dividing the span evenly, as
    this once did, invents positions the model never reported, and a Speaker
    Boundary Split cutting on them would cut in the wrong place.
    """
    speaker_str = str(seg.speaker)
    return [
        TranscriptWord(
            word=token.text.strip(),
            start=token.start,
            end=token.end,
            score=token.probability,
            speaker=speaker_str,
        )
        for token in seg.tokens
        if token.text.strip()
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_transcription_response(
    tokens: list[TranscriptToken],
    speaker_timeline: list[SpeakerSegment],
    duration: float,
    threshold: MinimumTurnThreshold | None = None,
) -> TranscriptionResult:
    """Build a :class:`TranscriptionResult` from project-owned types.

    Args:
        tokens: Ordered transcript tokens (Project-Owned Transcript Model).
        speaker_timeline: Speaker timeline segments from the Diarization Adapter.
        duration: Total audio duration in seconds.
        threshold: Minimum Turn Threshold governing the Speaker Boundary Split.

    Returns:
        TranscriptionResult with segments, word_segments, transcript,
        diarization, and raw_words.

    """
    if not tokens:
        diar = [
            DiarizationItem(start=round(s.start, 3), end=round(s.end, 3), speaker=str(s.speaker))
            for s in sorted(speaker_timeline, key=lambda x: x.start)
        ]
        return TranscriptionResult(diarization=diar)

    raw_words = [
        RawWord(
            word=t.text,
            start=round(t.start, 3),
            end=round(t.end, 3),
            score=t.probability,
        )
        for t in tokens
        if t.text and t.text.strip()
    ]

    # Group, split at speaker changes, attribute, then clamp. Splitting before
    # assignment means each piece is single-speaker by construction, so the
    # existing max-overlap rule labels it correctly and there is still only one
    # attribution path.
    seg_objects = _group_tokens_into_segments(tokens)
    seg_objects = _split_on_speaker_boundaries(
        seg_objects,
        speaker_timeline,
        threshold or MinimumTurnThreshold(),
    )
    _assign_speakers(seg_objects, speaker_timeline)
    seg_objects = _clamp_overlaps(seg_objects)

    segments: list[ResponseSegment] = []
    word_segments: list[TranscriptWord] = []

    for seg in seg_objects:
        rseg = build_segment(seg)
        word_segments.extend(rseg.words)
        segments.append(rseg)

    transcript = [TranscriptItem(start=s.start, end=s.end, text=s.text) for s in segments]
    diarization = [DiarizationItem(start=s.start, end=s.end, speaker=s.speaker) for s in segments]

    return TranscriptionResult(
        segments=segments,
        word_segments=word_segments,
        transcript=transcript,
        diarization=diarization,
        raw_words=raw_words,
    )
