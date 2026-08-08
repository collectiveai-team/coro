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
"""

from __future__ import annotations

from dataclasses import dataclass

from coro.core.models import SpeakerSegment

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

    hits = [
        entry for entry in merged if _overlap_seconds(start, end, entry.start, entry.end) > 0.0
    ]
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
