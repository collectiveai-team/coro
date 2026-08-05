"""STM conversion library for Quality Benchmark scoring.

Pure functions that convert between server response segments / AMI
annotations and STM text. No subprocess calls; no IO beyond reading
AMI XML files from the local annotation tree.
"""

from __future__ import annotations

from dataclasses import dataclass
import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from coro.core.models import SpeakerSegment


_ID_RE = re.compile(r"id\(([^)]+)\)")

DIARIZATION_ONLY_TEXT = "<sd>"
"""Placeholder text for references that carry speaker turns but no transcript.

Diarization-only corpora (e.g. VoxConverse RTTM) have no words, but STM lines
require a text field. Lines whose text is exactly this sentinel mark the item as
diarization-only so scoring reports DER and omits the (meaningless) WER.
"""


def _clean_text(text: str) -> str:
    text = text.replace("\n", " ").strip()
    return re.sub(r"\s+", " ", text)


def hyp_segments_to_stm(
    segments: list[dict[str, Any]],
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

    lines.sort(key=lambda line: (float(line.split()[3]), line.split()[2]))
    return "\n".join(lines) + "\n" if lines else ""


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

    STM text carries no word timings, so a straddling segment can only have its
    TIMES clamped — its text column crosses the edge intact. Callers that can
    still see word times should window there instead; the AMI path does, via
    ``ami_meeting_to_stm(window=...)``. This remains correct for references that
    are timing-only (diarization RTTM), where there is no text to split.
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
    lines.sort(key=lambda line: (float(line.split()[3]), line.split()[2]))
    return "\n".join(lines) + "\n" if lines else ""


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
    lines.sort(key=lambda line: (float(line.split()[3]), line.split()[2]))
    return "\n".join(lines) + "\n" if lines else ""


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
    lines.sort(key=lambda line: (float(line.split()[3]), line.split()[2]))
    return "\n".join(lines) + "\n" if lines else ""


# ---------------------------------------------------------------------------
# AMI annotation helpers
# ---------------------------------------------------------------------------


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _get_id(elem: ET.Element) -> str | None:
    for key, value in elem.attrib.items():
        if key == "id" or key == "nite:id" or key.endswith("}id"):
            return value
    return None


def _get_time(elem: ET.Element, name: str) -> float | None:
    value = elem.attrib.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _first_time(elem: ET.Element, *names: str) -> float | None:
    """Return the first parsable time attribute among ``names``.

    Written as an explicit None check rather than ``or`` because a legitimate
    time of 0.0 is falsy.
    """
    for name in names:
        value = _get_time(elem, name)
        if value is not None:
            return value
    return None


def _normalize_token(text: str) -> str:
    text = html.unescape(text or "")
    text = text.strip()
    return re.sub(r"\s+", " ", text)


def _read_words(path: Path) -> tuple[list[dict], dict[str, int]]:
    """Read an AMI word file, returning its words and an id index.

    The index spans EVERY identified element, not only ``<w>``. AMI word files
    interleave words with ``<disfmarker>``, ``<vocalsound>`` and ``<gap>``, and a
    segment's href range may terminate on any of them; indexing words alone
    leaves those endpoints unresolvable and silently discards the segment.
    Non-word elements are indexed as positions but carry no word payload.
    """
    tree = ET.parse(path)
    words: list[dict] = []
    id_to_index: dict[str, int] = {}
    for elem in tree.getroot().iter():
        element_id = _get_id(elem)
        if not element_id:
            continue
        if _local_name(elem.tag) == "w":
            start = _get_time(elem, "starttime")
            end = _get_time(elem, "endtime")
            token = _normalize_token("".join(elem.itertext()))
            if start is not None and end is not None and token:
                id_to_index[element_id] = len(words)
                words.append({"id": element_id, "start": start, "end": end, "word": token})
                continue
        # A non-word (or unusable word) element still anchors a range endpoint:
        # point it at the next word position so a range through it resolves.
        id_to_index[element_id] = len(words)
    return words, id_to_index


def _is_word_id(words: list[dict], id_to_index: dict[str, int], element_id: str) -> bool:
    """Report whether an indexed element is itself a word."""
    index = id_to_index[element_id]
    return index < len(words) and words[index]["id"] == element_id


def _words_from_child_href(
    href: str,
    words: list[dict],
    id_to_index: dict[str, int],
) -> list[dict]:
    """Resolve a ``<child>`` href to the words it covers.

    An href is a single id or an ``id(a)..id(b)`` range. A non-word endpoint
    indexes the word that follows it, so it bounds the range without
    contributing a token of its own.
    """
    ids = _ID_RE.findall(href)
    if not ids:
        return []
    first, last = ids[0], ids[-1]
    if first not in id_to_index or last not in id_to_index:
        return []
    if id_to_index[first] > id_to_index[last]:
        first, last = last, first
    start = id_to_index[first]
    end = id_to_index[last]
    if _is_word_id(words, id_to_index, last):
        end += 1
    return words[start:end]


def _read_segments(
    path: Path,
    words: list[dict],
    id_to_index: dict[str, int],
) -> list[_Segment]:
    tree = ET.parse(path)
    segments: list[_Segment] = []
    for seg in tree.getroot().iter():
        if _local_name(seg.tag) != "segment":
            continue
        seg_words: list[dict] = []
        for child in seg.iter():
            if _local_name(child.tag) != "child":
                continue
            href = child.attrib.get("href", "")
            seg_words.extend(_words_from_child_href(href, words, id_to_index))
        if not seg_words:
            # AMI segments time themselves with transcriber_start/transcriber_end;
            # starttime/endtime is the word-level spelling and never appears here.
            start = _first_time(seg, "transcriber_start", "starttime")
            end = _first_time(seg, "transcriber_end", "endtime")
            if start is not None and end is not None:
                seg_words = [w for w in words if w["start"] >= start and w["end"] <= end]
        if not seg_words:
            continue
        if any(w["word"].strip() for w in seg_words):
            segments.append(_segment_from_words(seg_words))
    return segments


@dataclass
class _Segment:
    """One reference turn, retaining the words it was built from.

    The words are kept so a window can be applied per word; once rendered to STM
    text the timings are gone and only segment times can be clamped.
    """

    start: float
    end: float
    text: str
    words: list[dict]


def _segment_from_words(seg_words: list[dict]) -> _Segment:
    """Build a segment from its words, keeping the words for later windowing."""
    return _Segment(
        start=min(w["start"] for w in seg_words),
        end=max(w["end"] for w in seg_words),
        text=" ".join(w["word"] for w in seg_words).strip(),
        words=seg_words,
    )


def _window_segments(segments: list[_Segment], window: tuple[float, float]) -> list[_Segment]:
    """Restrict segments to the words whose start falls in ``[lo, hi)``.

    Windowing rendered STM text can only clamp a segment's times — its text
    column is copied whole, so a segment crossing the edge donates every word to
    a clip whose audio contains only some of them, and the surplus scores as
    deletions. Word times are still available here, so membership is decided per
    word. The bound is half-open on the word START, the same rule Overlap Token
    Acceptance uses, so adjacent windows partition words instead of sharing or
    dropping any at the seam.
    """
    lo, hi = window
    windowed: list[_Segment] = []
    for seg in segments:
        kept = [w for w in seg.words if lo <= w["start"] < hi]
        if not kept or not any(w["word"].strip() for w in kept):
            continue
        windowed.append(_segment_from_words(kept))
    return windowed


def _fallback_word_segments(words: list[dict], max_gap: float = 1.0) -> list[_Segment]:
    if not words:
        return []
    chunks = []
    current = [words[0]]
    for word in words[1:]:
        gap = word["start"] - current[-1]["end"]
        if gap > max_gap:
            chunks.append(current)
            current = [word]
        else:
            current.append(word)
    chunks.append(current)
    return [_segment_from_words(chunk) for chunk in chunks]


def _find_annotation_file(root: Path, kind: str, meeting: str, speaker: str) -> Path | None:
    pattern = f"**/{kind}/{meeting}.{speaker}.{kind}.xml"
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


def _speakers_for_meeting(root: Path, meeting: str) -> list[str]:
    speakers = set()
    for path in root.glob(f"**/words/{meeting}.*.words.xml"):
        parts = path.name.split(".")
        if len(parts) >= 4:
            speakers.add(parts[1])
    return sorted(speakers)


def ami_meeting_to_stm(
    ami_root: Path,
    meeting_id: str,
    *,
    window: tuple[float, float] | None = None,
) -> str:
    """Produce a Reference STM string for an AMI meeting from its annotation tree.

    Walks the AMI annotation XML files under *ami_root*, extracts per-speaker
    word timing, groups words into segments, and returns STM text sorted by
    (start_time, speaker).

    ``window`` restricts the result to a ``[lo, hi)`` span in meeting-absolute
    seconds, applied per word rather than per segment so a segment straddling
    the edge contributes only the words the span actually contains. Times are
    not rebased — see ``slice_stm_window`` for that.
    """
    lines: list[str] = []

    for speaker in _speakers_for_meeting(ami_root, meeting_id):
        words_path = _find_annotation_file(ami_root, "words", meeting_id, speaker)
        segments_path = _find_annotation_file(ami_root, "segments", meeting_id, speaker)

        if words_path is None:
            continue

        words, id_to_index = _read_words(words_path)

        if segments_path is not None:
            segments = _read_segments(segments_path, words, id_to_index)
        else:
            segments = _fallback_word_segments(words)

        if window is not None:
            segments = _window_segments(segments, window)

        for seg in segments:
            text = seg.text.replace("\n", " ").strip()
            if not text:
                continue
            lines.append(f"{meeting_id} 1 {speaker} {seg.start:.3f} {seg.end:.3f} {text}")

    lines.sort(key=lambda line: (float(line.split()[3]), line.split()[2]))
    return "\n".join(lines) + "\n" if lines else ""
