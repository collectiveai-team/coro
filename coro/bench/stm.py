"""STM conversion library for Quality Benchmark scoring.

Pure functions that convert between server response segments / AMI
annotations and STM text. No subprocess calls; no IO beyond reading
AMI XML files from the local annotation tree.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from coro.bench.stm_ami import ami_meeting_to_stm
from coro.bench.stm_deepgram import deepgram_word_segments

if TYPE_CHECKING:
    from coro.bench.models.response import HypSegment, HypWord, TranscriptResponse
    from coro.core.models import SpeakerSegment

__all__ = [
    "DIARIZATION_ONLY_TEXT",
    "ami_meeting_to_stm",
    "hyp_response_to_stm",
    "hyp_segments_to_stm",
    "rttm_to_stm",
    "slice_stm_window",
    "speaker_timeline_to_stm",
    "word_segments_to_stm",
]

DIARIZATION_ONLY_TEXT = "<sd>"
"""Placeholder text for references that carry speaker turns but no transcript.

Diarization-only corpora (e.g. VoxConverse RTTM) have no words, but STM lines
require a text field. Lines whose text is exactly this sentinel mark the item as
diarization-only so scoring reports DER and omits the (meaningless) WER.
"""


def _clean_text(text: str) -> str:
    text = text.replace("\n", " ").strip()
    return re.sub(r"\s+", " ", text)


def _stm_text(lines: list[str]) -> str:
    """Join STM lines sorted by (start_time, speaker), with a trailing newline."""
    lines.sort(key=lambda line: (float(line.split()[3]), line.split()[2]))
    return "\n".join(lines) + "\n" if lines else ""


@dataclass(frozen=True)
class _TimedWord:
    start: float
    end: float
    word: str
    speaker: str


def _timed_words(word_segments: list[HypWord]) -> list[_TimedWord]:
    """Keep the words with times and text, ordered by start time."""
    usable: list[_TimedWord] = []
    for word in word_segments:
        start = word.get("start")
        end = word.get("end")
        text = str(word.get("word", ""))
        if start is None or end is None or not _clean_text(text):
            continue
        speaker = str(word.get("speaker", "UNKNOWN"))
        usable.append(_TimedWord(start=float(start), end=float(end), word=text, speaker=speaker))
    usable.sort(key=lambda word: word.start)
    return usable


def hyp_segments_to_stm(
    segments: list[HypSegment],
    recording_id: str,
    *,
    channel: str = "1",
) -> str:
    """Convert a diarized_json ``segments`` list to STM text.

    Speaker labels are passed through unchanged from the server response.
    Segments with missing times, empty text, or zero/negative duration
    are dropped.  Output lines are sorted by (start_time, speaker).
    """
    lines: list[str] = []
    for seg in segments:
        start = seg.get("start")
        end = seg.get("end")
        text = _clean_text(str(seg.get("text", "")))
        speaker = str(seg.get("speaker", "UNKNOWN"))

        if start is None or end is None or not text:
            continue

        start_f = float(start)
        end_f = float(end)

        if end_f <= start_f:
            continue

        lines.append(f"{recording_id} {channel} {speaker} {start_f:.3f} {end_f:.3f} {text}")

    return _stm_text(lines)


def word_segments_to_stm(
    word_segments: list[HypWord],
    recording_id: str,
    *,
    channel: str = "1",
) -> str:
    """Convert a response ``word_segments`` list to STM text.

    Consecutive words sharing a speaker are grouped into one maximal
    same-speaker run per STM line, so the file carries the per-word speaker
    truth rather than a segment-level summary of it. Words are ordered by start
    time first; a run ends whenever the speaker changes.
    """
    lines: list[str] = []
    for speaker, group in itertools.groupby(_timed_words(word_segments), key=lambda w: w.speaker):
        run = list(group)
        start = min(word.start for word in run)
        end = max(word.end for word in run)
        text = _clean_text(" ".join(word.word for word in run))
        if end > start and text:
            lines.append(f"{recording_id} {channel} {speaker} {start:.3f} {end:.3f} {text}")
    return _stm_text(lines)


def hyp_response_to_stm(
    response: TranscriptResponse,
    recording_id: str,
    *,
    channel: str = "1",
) -> str:
    """Build the Hypothesis STM from the richest speaker source the response has.

    Prefers per-word speaker labels and falls back to ``segments``. The
    preference is what keeps WDER measuring per-word attribution: once a
    response segment's speaker becomes a duration-weighted majority *summary*
    of its words (issue ``12``), scoring from ``segments`` would measure the
    summary instead, and abstention would vanish from the hypothesis before it
    could be counted.

    Words are looked for in two places, because they arrive in two shapes: the
    Deepgram-native nesting served by ``POST /v1/listen``, and a top-level
    ``word_segments``/``words`` list. The vendor shape is checked first — it
    carries no ``segments`` at all, so missing it does not merely degrade the
    hypothesis, it empties it, and scoring reports that as a clean run.

    The fallback is not dead code: the ``diarized_json`` wire format carries no
    word field, so every response from the OpenAI endpoint still takes it. Both
    sources agree while a segment holds exactly one speaker.
    """
    vendor_words = deepgram_word_segments(response, recording_id=recording_id)
    if vendor_words:
        return word_segments_to_stm(vendor_words, recording_id, channel=channel)
    word_segments = response.get("word_segments") or response.get("words") or []
    if word_segments and any(word.get("speaker") is not None for word in word_segments):
        return word_segments_to_stm(word_segments, recording_id, channel=channel)
    return hyp_segments_to_stm(response.get("segments", []), recording_id, channel=channel)


def slice_stm_window(
    stm_text: str,
    start: float,
    end: float,
    *,
    rebase: bool = True,
    recording_id: str | None = None,
) -> str:
    """Slice an STM to the ``[start, end)`` time window for short-clip benchmarks.

    Lines overlapping the window are kept and their times clamped to it; lines
    fully outside are dropped. When ``rebase`` is True (the default for cut audio
    that starts at 0), kept times are shifted so the window start becomes 0.0.
    When ``recording_id`` is given, the STM session id (column 1) is rewritten to
    it so the clip's reference matches a hypothesis keyed by the clip stem.
    Output is sorted by (start_time, speaker), matching the other STM builders.
    """
    if end <= start:
        return ""
    shift = start if rebase else 0.0
    lines: list[str] = []
    for raw in stm_text.splitlines():
        parts = raw.strip().split(maxsplit=5)
        if len(parts) < 6:
            continue
        try:
            seg_start = float(parts[3])
            seg_end = float(parts[4])
        except ValueError:
            continue
        if seg_end <= start or seg_start >= end:
            continue
        clamped_start = max(seg_start, start) - shift
        clamped_end = min(seg_end, end) - shift
        if clamped_end <= clamped_start:
            continue
        session = recording_id if recording_id is not None else parts[0]
        lines.append(
            f"{session} {parts[1]} {parts[2]} {clamped_start:.3f} {clamped_end:.3f} {parts[5]}"
        )
    return _stm_text(lines)


def rttm_to_stm(
    rttm_text: str,
    recording_id: str,
    *,
    channel: str = "1",
    text: str = DIARIZATION_ONLY_TEXT,
) -> str:
    """Convert RTTM ``SPEAKER`` turns to a diarization-only reference STM.

    RTTM has no transcript, so every emitted STM line carries ``text`` (the
    diarization-only sentinel by default) — enough for DER scoring, which uses
    only speaker labels and timings. ``SPEAKER`` lines provide onset/duration in
    columns 4/5 and the speaker label in column 8; turns with non-positive
    duration are dropped. Output is sorted by (start_time, speaker).
    """
    lines: list[str] = []
    for raw in rttm_text.splitlines():
        parts = raw.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        try:
            start_f = float(parts[3])
            dur_f = float(parts[4])
        except ValueError:
            continue
        end_f = start_f + dur_f
        if dur_f <= 0:
            continue
        speaker = parts[7]
        lines.append(f"{recording_id} {channel} {speaker} {start_f:.3f} {end_f:.3f} {text}")
    return _stm_text(lines)


def speaker_timeline_to_stm(
    timeline: list[SpeakerSegment],
    recording_id: str,
    *,
    channel: str = "1",
    text: str = DIARIZATION_ONLY_TEXT,
) -> str:
    """Convert a Project-Owned speaker timeline to a diarization-only STM.

    Diarization Adapters return SpeakerSegment timelines with no transcript, so
    every emitted STM line carries the diarization-only sentinel text — enough
    for DER scoring, which uses only speaker labels and timings. Output is
    sorted by (start_time, speaker), matching the other STM builders.
    """
    lines = [
        f"{recording_id} {channel} spk{seg.speaker} {seg.start:.3f} {seg.end:.3f} {text}"
        for seg in timeline
        if seg.end > seg.start
    ]
    return _stm_text(lines)
