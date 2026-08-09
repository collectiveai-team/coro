"""Speaker attribution against a Diarization Adapter timeline.

Pure Core Boundary transformations: a speaker timeline in, a per-span speaker
decision out. Attribution is computed at *word* granularity by the response
builder; this module owns the rules that decision depends on.

Three behaviours differ deliberately from a naive maximum-overlap rule:

- **Gap-bounded merging.** Same-speaker timeline entries are coalesced only
  when the silence between them is short. Coalescing across an arbitrary gap
  produces one long entry that outweighs every other speaker inside the gap.
- **Unknown rather than arbitrary.** A span that overlaps no timeline entry is
  attributed to :data:`UNKNOWN_SPEAKER`, not to whichever speaker happens to be
  first. This covers spans past the diarization horizon and spans in
  diarization gaps alike.
- **Overlapped speech is flagged.** The diarizer emits an independent timeline
  per speaker, so genuinely overlapped speech shows up as concurrently active
  entries. The winner is still a single label, but the span is marked so the
  collapse is visible instead of silent.
- **Punctuation-aware realignment.** Per-word attribution splits a run at
  every speaker change, including single-word flickers from a noisy
  diarization timeline. :func:`realign_speakers_with_punctuation` corrects a
  change that does not coincide with sentence-final punctuation (or a Spanish
  opening mark) by expanding to the enclosing sentence and relabelling it to
  the speaker with the most total word duration.
"""

from __future__ import annotations

from dataclasses import dataclass

from coro.core.models import SpeakerSegment, TranscriptToken
from coro.core.segmentation import closes_segment, opens_segment

MAX_SPEAKER_MERGE_GAP_SECONDS = 0.5
"""Longest silence across which two same-speaker entries are still one entry."""

UNKNOWN_SPEAKER = -1
"""Sentinel for spans with no diarization support."""

NO_DIARIZATION_SPEAKER = 1
"""Label used when the server runs without a Diarization Adapter at all."""


@dataclass(frozen=True)
class SpeakerAttribution:
    """The speaker decision for one span, plus whether the span was overlapped."""

    speaker: int
    overlap: bool


def _overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Return the length of the intersection of two intervals (0.0 if disjoint)."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def merge_speaker_timeline(
    speaker_timeline: list[SpeakerSegment],
    *,
    max_gap_seconds: float = MAX_SPEAKER_MERGE_GAP_SECONDS,
) -> list[SpeakerSegment]:
    """Coalesce same-speaker entries separated by at most ``max_gap_seconds``.

    Entries are grouped per speaker first, so interleaved speakers no longer
    prevent a merge, and a merge no longer swallows an arbitrarily long
    silence. The result is sorted by ``(start, speaker)`` and may contain
    entries that overlap *across* speakers — that is the overlapped-speech
    signal :func:`attribute_span` reads.
    """
    by_speaker: dict[int, list[SpeakerSegment]] = {}
    for item in speaker_timeline:
        if item.end <= item.start:
            continue
        by_speaker.setdefault(item.speaker, []).append(item)

    merged: list[SpeakerSegment] = []
    for speaker, items in by_speaker.items():
        items.sort(key=lambda x: x.start)
        current: SpeakerSegment | None = None
        for item in items:
            if current is not None and item.start - current.end <= max_gap_seconds:
                current = SpeakerSegment(
                    start=current.start,
                    end=max(current.end, item.end),
                    speaker=speaker,
                )
                continue
            if current is not None:
                merged.append(current)
            current = SpeakerSegment(start=item.start, end=item.end, speaker=speaker)
        if current is not None:
            merged.append(current)

    merged.sort(key=lambda s: (s.start, s.speaker))
    return merged


def attribute_span(
    start: float,
    end: float,
    merged: list[SpeakerSegment],
) -> SpeakerAttribution:
    """Attribute a ``[start, end)`` span against a merged speaker timeline.

    Args:
        start: Span start in seconds.
        end: Span end in seconds.
        merged: Output of :func:`merge_speaker_timeline`.

    Returns:
        The maximum-overlap speaker, or :data:`UNKNOWN_SPEAKER` when the span
        overlaps nothing, together with an ``overlap`` flag set when two
        different speakers are concurrently active inside the span.

    """
    if not merged:
        return SpeakerAttribution(speaker=NO_DIARIZATION_SPEAKER, overlap=False)

    hits = [entry for entry in merged if _overlap_seconds(start, end, entry.start, entry.end) > 0.0]
    if not hits:
        return SpeakerAttribution(speaker=UNKNOWN_SPEAKER, overlap=False)

    totals: dict[int, float] = {}
    for entry in hits:
        totals[entry.speaker] = totals.get(entry.speaker, 0.0) + _overlap_seconds(
            start, end, entry.start, entry.end
        )
    # sorted() makes the tie-break the lowest speaker id, so attribution is
    # deterministic regardless of timeline ordering.
    speaker = max(sorted(totals), key=lambda label: totals[label])
    return SpeakerAttribution(
        speaker=speaker,
        overlap=_has_concurrent_speakers(start, end, hits),
    )


def _has_concurrent_speakers(start: float, end: float, hits: list[SpeakerSegment]) -> bool:
    """Return True when two different speakers are active at the same instant.

    Only the portion of each entry inside ``[start, end)`` counts, so a span
    that merely straddles a clean speaker turn is not reported as overlapped.
    """
    clipped = [(entry.speaker, max(start, entry.start), min(end, entry.end)) for entry in hits]
    for index, (speaker_a, a_start, a_end) in enumerate(clipped):
        for speaker_b, b_start, b_end in clipped[index + 1 :]:
            if speaker_a == speaker_b:
                continue
            if _overlap_seconds(a_start, a_end, b_start, b_end) > 0.0:
                return True
    return False


_AttributedToken = tuple[TranscriptToken, SpeakerAttribution]


def realign_speakers_with_punctuation(
    attributed: list[_AttributedToken],
) -> list[_AttributedToken]:
    """Correct word-level speaker flicker using sentence-final punctuation.

    A word-level speaker change is real evidence of a turn only where it
    coincides with sentence-final punctuation or a Spanish opening mark (see
    :mod:`coro.core.segmentation`); everywhere else it is noise from a
    diarization timeline that changes hands mid-sentence. Every word inside a
    sentence that disagrees with the sentence is relabelled to the speaker
    with the most total word duration in that sentence — duration, not word
    count, because one long word is stronger evidence than several short
    ones.

    :data:`UNKNOWN_SPEAKER` words are excluded from the vote and are never
    relabelled: abstention must not be overwritten by a majority vote, and
    must not itself be allowed to win one.

    Args:
        attributed: Tokens paired with their per-word attribution, in order,
            from a single segmentation run (see
            :mod:`coro.core.segmentation`).

    Returns:
        A new list, same length and order, with flicker speakers corrected.

    """
    if not attributed:
        return []

    realigned = list(attributed)
    for lo, hi in _sentence_bounds(attributed):
        sentence = attributed[lo:hi]
        real_speakers = {a.speaker for _, a in sentence if a.speaker != UNKNOWN_SPEAKER}
        if len(real_speakers) <= 1:
            continue
        majority = _majority_speaker(sentence)
        for index in range(lo, hi):
            token, attribution = attributed[index]
            if attribution.speaker == UNKNOWN_SPEAKER or attribution.speaker == majority:
                continue
            realigned[index] = (
                token,
                SpeakerAttribution(speaker=majority, overlap=attribution.overlap),
            )
    return realigned


def _sentence_bounds(attributed: list[_AttributedToken]) -> list[tuple[int, int]]:
    """Split ``attributed`` into ``[start, stop)`` index ranges, one per sentence."""
    bounds: list[tuple[int, int]] = []
    start = 0
    total = len(attributed)
    for index in range(1, total + 1):
        at_end = index == total
        prev_closes = closes_segment(attributed[index - 1][0].text)
        next_opens = not at_end and opens_segment(attributed[index][0].text)
        if at_end or prev_closes or next_opens:
            bounds.append((start, index))
            start = index
    return bounds


def _majority_speaker(sentence: list[_AttributedToken]) -> int:
    """Return the non-unknown speaker with the most total word duration."""
    totals: dict[int, float] = {}
    for token, attribution in sentence:
        if attribution.speaker == UNKNOWN_SPEAKER:
            continue
        totals[attribution.speaker] = totals.get(attribution.speaker, 0.0) + token.duration()
    # sorted() ties the tie-break to the lowest speaker id, matching attribute_span.
    return max(sorted(totals), key=lambda label: totals[label])
