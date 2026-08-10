"""STM conversion library for Quality Benchmark scoring.

Pure functions that convert between server response segments / AMI
annotations and STM text. No subprocess calls; no IO beyond reading
AMI XML files from the local annotation tree.
"""

from __future__ import annotations

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


def word_segments_to_stm(
    word_segments: list[dict[str, Any]],
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
    usable = [
        word
        for word in word_segments
        if word.get("start") is not None
        and word.get("end") is not None
        and _clean_text(str(word.get("word", "")))
    ]
    usable.sort(key=lambda word: float(word["start"]))

    lines: list[str] = []
    run: list[dict[str, Any]] = []

    def flush() -> None:
        if not run:
            return
        start = min(float(word["start"]) for word in run)
        end = max(float(word["end"]) for word in run)
        text = _clean_text(" ".join(str(word.get("word", "")) for word in run))
        if end > start and text:
            speaker = str(run[0].get("speaker", "UNKNOWN"))
            lines.append(f"{recording_id} {channel} {speaker} {start:.3f} {end:.3f} {text}")

    for word in usable:
        if run and str(word.get("speaker", "UNKNOWN")) != str(run[0].get("speaker", "UNKNOWN")):
            flush()
            run = []
        run.append(word)
    flush()

    lines.sort(key=lambda line: (float(line.split()[3]), line.split()[2]))
    return "\n".join(lines) + "\n" if lines else ""


def hyp_response_to_stm(
    response: dict[str, Any],
    recording_id: str,
    *,
    channel: str = "1",
) -> str:
    """Build the Hypothesis STM from the richest speaker source the response has.

    Prefers ``word_segments`` — the per-word speaker labels — and falls back to
    ``segments``. The preference is what keeps WDER measuring per-word
    attribution: once a response segment's speaker becomes a duration-weighted
    majority *summary* of its words (issue ``12``), scoring from ``segments``
    would measure the summary instead, and abstention would vanish from the
    hypothesis before it could be counted.

    The fallback is not dead code: the ``diarized_json`` wire format carries no
    word field today, so every response the benchmark currently receives takes
    it. Both sources agree while a segment holds exactly one speaker.
    """
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


def _normalize_token(text: str) -> str:
    text = html.unescape(text or "")
    text = text.strip()
    return re.sub(r"\s+", " ", text)


def _read_words(path: Path) -> tuple[list[dict], dict[str, int]]:
    tree = ET.parse(path)
    words: list[dict] = []
    for elem in tree.getroot().iter():
        if _local_name(elem.tag) != "w":
            continue
        word_id = _get_id(elem)
        start = _get_time(elem, "starttime")
        end = _get_time(elem, "endtime")
        token = _normalize_token("".join(elem.itertext()))
        if not word_id or start is None or end is None or not token:
            continue
        words.append({"id": word_id, "start": start, "end": end, "word": token})
    id_to_index = {w["id"]: i for i, w in enumerate(words)}
    return words, id_to_index


def _words_from_child_href(
    href: str,
    words: list[dict],
    id_to_index: dict[str, int],
) -> list[dict]:
    ids = _ID_RE.findall(href)
    if not ids:
        return []
    if len(ids) == 1:
        idx = id_to_index.get(ids[0])
        return [] if idx is None else [words[idx]]
    start_idx = id_to_index.get(ids[0])
    end_idx = id_to_index.get(ids[-1])
    if start_idx is None or end_idx is None:
        return []
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
    return words[start_idx : end_idx + 1]


def _read_segments(
    path: Path,
    words: list[dict],
    id_to_index: dict[str, int],
) -> list[dict]:
    tree = ET.parse(path)
    segments: list[dict] = []
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
            start = _get_time(seg, "starttime")
            end = _get_time(seg, "endtime")
            if start is not None and end is not None:
                seg_words = [w for w in words if w["start"] >= start and w["end"] <= end]
        if not seg_words:
            continue
        start = min(w["start"] for w in seg_words)
        end = max(w["end"] for w in seg_words)
        text = " ".join(w["word"] for w in seg_words)
        if text.strip():
            segments.append({"start": start, "end": end, "text": text.strip()})
    return segments


def _fallback_word_segments(words: list[dict], max_gap: float = 1.0) -> list[dict]:
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
    return [
        {
            "start": min(w["start"] for w in chunk),
            "end": max(w["end"] for w in chunk),
            "text": " ".join(w["word"] for w in chunk),
        }
        for chunk in chunks
    ]


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


def ami_meeting_to_stm(ami_root: Path, meeting_id: str) -> str:
    """Produce a Reference STM string for an AMI meeting from its annotation tree.

    Walks the AMI annotation XML files under *ami_root*, extracts per-speaker
    word timing, groups words into segments, and returns STM text sorted by
    (start_time, speaker).

    Known limitations of this reference, measured and tracked, not fixed here:

    - Each emitted line spans ``min(word.start)..max(word.end)`` of one AMI
      segment, so intra-turn pauses are marked as speech. The community AMI
      diarization setup (``BUTSpeechFIT/AMI-diarization-setup``) explicitly does
      not merge across pauses, and forced-alignment references are tighter still.
      Scored against forced-alignment RTTMs at a 0 s collar, this reference differs
      by ~43% DER — almost entirely boundaries, not speaker identity.
    - :func:`_read_segments` drops any AMI segment whose word-id range starts or
      ends on a non-``<w>`` element (``<disfmarker>``, ``<vocalsound>``,
      ``<gap>``), which loses 18-31% of the annotated words.

    Consequently both DER and every WER metric scored against this reference are
    internally comparable but not comparable to published AMI results.
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

        for seg in segments:
            text = seg["text"].replace("\n", " ").strip()
            if not text:
                continue
            lines.append(f"{meeting_id} 1 {speaker} {seg['start']:.3f} {seg['end']:.3f} {text}")

    lines.sort(key=lambda line: (float(line.split()[3]), line.split()[2]))
    return "\n".join(lines) + "\n" if lines else ""
