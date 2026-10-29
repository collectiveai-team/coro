"""Reference STM builder for AMI meetings, from the AMI annotation XML tree."""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

_ID_RE = re.compile(r"id\(([^)]+)\)")


@dataclass(frozen=True)
class _Word:
    id: str
    start: float
    end: float
    word: str


@dataclass(frozen=True)
class _Segment:
    start: float
    end: float
    text: str


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


def _read_words(path: Path) -> tuple[list[_Word], dict[str, int]]:
    tree = ET.parse(path)  # noqa: S314 - trusted local AMI corpus annotations
    words: list[_Word] = []
    for elem in tree.getroot().iter():
        if _local_name(elem.tag) != "w":
            continue
        word_id = _get_id(elem)
        start = _get_time(elem, "starttime")
        end = _get_time(elem, "endtime")
        token = _normalize_token("".join(elem.itertext()))
        if not word_id or start is None or end is None or not token:
            continue
        words.append(_Word(id=word_id, start=start, end=end, word=token))
    id_to_index = {w.id: i for i, w in enumerate(words)}
    return words, id_to_index


def _words_from_child_href(
    href: str,
    words: list[_Word],
    id_to_index: dict[str, int],
) -> list[_Word]:
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


def _words_in_window(seg: ET.Element, words: list[_Word]) -> list[_Word]:
    """Return the words inside the segment's own start/end times, if it has them."""
    start = _get_time(seg, "starttime")
    end = _get_time(seg, "endtime")
    if start is None or end is None:
        return []
    return [w for w in words if w.start >= start and w.end <= end]


def _segment_words(
    seg: ET.Element,
    words: list[_Word],
    id_to_index: dict[str, int],
) -> list[_Word]:
    """Resolve a segment's words from its ``child`` hrefs, else from its time window."""
    seg_words: list[_Word] = []
    for child in seg.iter():
        if _local_name(child.tag) == "child":
            href = child.attrib.get("href", "")
            seg_words.extend(_words_from_child_href(href, words, id_to_index))
    return seg_words or _words_in_window(seg, words)


def _read_segments(
    path: Path,
    words: list[_Word],
    id_to_index: dict[str, int],
) -> list[_Segment]:
    tree = ET.parse(path)  # noqa: S314 - trusted local AMI corpus annotations
    segments: list[_Segment] = []
    for seg in tree.getroot().iter():
        if _local_name(seg.tag) != "segment":
            continue
        seg_words = _segment_words(seg, words, id_to_index)
        if not seg_words:
            continue
        text = " ".join(w.word for w in seg_words).strip()
        if text:
            start = min(w.start for w in seg_words)
            end = max(w.end for w in seg_words)
            segments.append(_Segment(start=start, end=end, text=text))
    return segments


def _fallback_word_segments(words: list[_Word], max_gap: float = 1.0) -> list[_Segment]:
    if not words:
        return []
    chunks: list[list[_Word]] = []
    current = [words[0]]
    for word in words[1:]:
        gap = word.start - current[-1].end
        if gap > max_gap:
            chunks.append(current)
            current = [word]
        else:
            current.append(word)
    chunks.append(current)
    return [
        _Segment(
            start=min(w.start for w in chunk),
            end=max(w.end for w in chunk),
            text=" ".join(w.word for w in chunk),
        )
        for chunk in chunks
    ]


def _find_annotation_file(root: Path, kind: str, meeting: str, speaker: str) -> Path | None:
    pattern = f"**/{kind}/{meeting}.{speaker}.{kind}.xml"
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


def _speakers_for_meeting(root: Path, meeting: str) -> list[str]:
    speakers: set[str] = set()
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

    Two known limitations of this reference, measured, neither fixed here:

    - **Dropped segments (has a fix in flight).** :func:`_read_segments` resolves a
      segment's word-id range against an index built from ``<w>`` alone, but AMI
      interleaves ``<disfmarker>``, ``<vocalsound>`` and ``<gap>`` into the same id
      sequence, so a range terminating on one resolves to nothing and the whole
      segment is silently discarded — 18-31% of the annotated words, depending on
      the meeting. PR #32 repairs this by indexing every identified element; until
      it lands, every AMI WER and DER figure is scored against an incomplete
      reference.
    - **Loose boundaries (survives that fix).** Each emitted line spans
      ``min(word.start)..max(word.end)`` of one AMI segment, so intra-turn pauses
      are marked as speech. That matches the community setup
      (``BUTSpeechFIT/AMI-diarization-setup``) in speech volume but not the
      forced-alignment references model cards score AMI against, which are ~24%
      tighter. The residual disagreement is ~29% DER, almost entirely boundaries
      rather than speaker identity.

    The second limitation is not cosmetic: the two references disagree on whether
    the diarizer over- or under-detects speech, so an error decomposition taken
    against this reference can select the wrong post-processing parameters.
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
            text = seg.text.replace("\n", " ").strip()
            if not text:
                continue
            lines.append(f"{meeting_id} 1 {speaker} {seg.start:.3f} {seg.end:.3f} {text}")

    lines.sort(key=lambda line: (float(line.split()[3]), line.split()[2]))
    return "\n".join(lines) + "\n" if lines else ""
