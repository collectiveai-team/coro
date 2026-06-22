"""Shared diarization-segment normalization helpers.

Backend-neutral utilities that convert heterogeneous native diarization
outputs (NeMo strings/tuples, pyannote ``Segment``/label pairs, objects with
``start``/``end``/``speaker`` attributes) into Project-Owned ``SpeakerSegment``
values. Kept free of any backend import so every Diarization Adapter reuses
the same normalization at its edge.
"""

from __future__ import annotations

import re

import numpy as np

from coro.core.models import SpeakerSegment


def speaker_to_one_indexed(speaker) -> int:
    """Convert a zero-indexed or string speaker label to a 1-indexed int."""
    if isinstance(speaker, int):
        return speaker + 1
    if isinstance(speaker, np.integer):
        return int(speaker) + 1
    match = re.search(r"\d+", str(speaker))
    if match:
        return int(match.group(0)) + 1
    return 1


def coerce_diarization_segment(seg):
    """Coerce a native diarization entry into a ``(start, end, speaker)`` tuple."""
    if isinstance(seg, str):
        parts = seg.replace(",", " ").split()
        if len(parts) >= 3:
            return float(parts[0]), float(parts[1]), parts[2]
    if isinstance(seg, (tuple, list)) and len(seg) >= 3:
        return float(seg[0]), float(seg[1]), seg[2]
    return (
        float(getattr(seg, "start", 0.0) or 0.0),
        float(getattr(seg, "end", 0.0) or 0.0),
        getattr(seg, "speaker", 0),
    )


def convert_diarization_segments(
    native_segments,
    *,
    duration: float,
) -> list[SpeakerSegment]:
    """Convert native diarization segment objects to SpeakerSegments.

    Args:
        native_segments: Iterable of backend diarization outputs.
        duration: Total audio duration in seconds; end times are clamped.

    Returns:
        Deduplicated list of SpeakerSegment sorted by start time.

    """
    timeline: list[SpeakerSegment] = []
    seen: set[tuple[float, float, int]] = set()

    for seg in native_segments:
        start, end, speaker_label = coerce_diarization_segment(seg)
        start = max(0.0, start)
        end = min(duration, end)
        if end <= start:
            continue
        speaker = speaker_to_one_indexed(speaker_label)
        key = (round(start, 3), round(end, 3), speaker)
        if key in seen:
            continue
        seen.add(key)
        timeline.append(SpeakerSegment(start=key[0], end=key[1], speaker=speaker))

    timeline.sort(key=lambda s: s.start)
    return timeline
