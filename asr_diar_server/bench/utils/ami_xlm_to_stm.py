#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path


ID_RE = re.compile(r"id\(([^)]+)\)")


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def get_id(elem: ET.Element) -> str | None:
    for key, value in elem.attrib.items():
        if key == "id" or key == "nite:id" or key.endswith("}id"):
            return value
    return None


def get_time(elem: ET.Element, name: str) -> float | None:
    value = elem.attrib.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def normalize_token(text: str) -> str:
    text = html.unescape(text or "")
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def read_words(path: Path) -> tuple[list[dict], dict[str, int]]:
    tree = ET.parse(path)
    words: list[dict] = []

    for elem in tree.getroot().iter():
        if local_name(elem.tag) != "w":
            continue

        word_id = get_id(elem)
        start = get_time(elem, "starttime")
        end = get_time(elem, "endtime")
        token = normalize_token("".join(elem.itertext()))

        if not word_id or start is None or end is None or not token:
            continue

        words.append(
            {
                "id": word_id,
                "start": start,
                "end": end,
                "word": token,
            }
        )

    id_to_index = {w["id"]: i for i, w in enumerate(words)}
    return words, id_to_index


def words_from_child_href(
    href: str,
    words: list[dict],
    id_to_index: dict[str, int],
) -> list[dict]:
    ids = ID_RE.findall(href)

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


def read_segments(
    path: Path,
    words: list[dict],
    id_to_index: dict[str, int],
) -> list[dict]:
    tree = ET.parse(path)
    segments: list[dict] = []

    for seg in tree.getroot().iter():
        if local_name(seg.tag) != "segment":
            continue

        seg_words: list[dict] = []

        for child in seg.iter():
            if local_name(child.tag) != "child":
                continue
            href = child.attrib.get("href", "")
            seg_words.extend(words_from_child_href(href, words, id_to_index))

        # Fallback: use segment timestamps if child references are absent.
        if not seg_words:
            start = get_time(seg, "starttime")
            end = get_time(seg, "endtime")
            if start is not None and end is not None:
                seg_words = [
                    w for w in words
                    if w["start"] >= start and w["end"] <= end
                ]

        if not seg_words:
            continue

        start = min(w["start"] for w in seg_words)
        end = max(w["end"] for w in seg_words)
        text = " ".join(w["word"] for w in seg_words)

        if text.strip():
            segments.append(
                {
                    "start": start,
                    "end": end,
                    "text": text.strip(),
                }
            )

    return segments


def fallback_word_segments(words: list[dict], max_gap: float = 1.0) -> list[dict]:
    if not words:
        return []

    segments = []
    current = [words[0]]

    for word in words[1:]:
        gap = word["start"] - current[-1]["end"]
        if gap > max_gap:
            segments.append(current)
            current = [word]
        else:
            current.append(word)

    segments.append(current)

    return [
        {
            "start": min(w["start"] for w in chunk),
            "end": max(w["end"] for w in chunk),
            "text": " ".join(w["word"] for w in chunk),
        }
        for chunk in segments
    ]


def find_annotation_file(root: Path, kind: str, meeting: str, speaker: str) -> Path | None:
    pattern = f"**/{kind}/{meeting}.{speaker}.{kind}.xml"
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


def speakers_for_meeting(root: Path, meeting: str) -> list[str]:
    speakers = set()

    for path in root.glob(f"**/words/{meeting}.*.words.xml"):
        # EN2001a.A.words.xml -> A
        parts = path.name.split(".")
        if len(parts) >= 4:
            speakers.add(parts[1])

    return sorted(speakers)


def convert_meeting(root: Path, meeting: str, out_path: Path) -> None:
    lines: list[str] = []

    for speaker in speakers_for_meeting(root, meeting):
        words_path = find_annotation_file(root, "words", meeting, speaker)
        segments_path = find_annotation_file(root, "segments", meeting, speaker)

        if words_path is None:
            continue

        words, id_to_index = read_words(words_path)

        if segments_path is not None:
            segments = read_segments(segments_path, words, id_to_index)
        else:
            segments = fallback_word_segments(words)

        for seg in segments:
            text = seg["text"].replace("\n", " ").strip()
            if not text:
                continue

            lines.append(
                f"{meeting} 1 {speaker} "
                f"{seg['start']:.3f} {seg['end']:.3f} {text}"
            )

    lines.sort(key=lambda line: (float(line.split()[3]), line.split()[2]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ami-root", type=Path, default=Path("amicorpus"))
    parser.add_argument("--out-dir", type=Path, default=Path("stm"))
    parser.add_argument("meetings", nargs="+")
    args = parser.parse_args()

    for meeting in args.meetings:
        out_path = args.out_dir / f"{meeting}.ref.stm"
        convert_meeting(args.ami_root, meeting, out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()