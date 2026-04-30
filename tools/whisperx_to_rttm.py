#!/usr/bin/env python3
"""Convert WhisperX diarization JSON to RTTM."""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert WhisperX JSON output with speaker turns to RTTM."
    )
    parser.add_argument("input", type=Path, help="WhisperX JSON file")
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        help="Output RTTM file (defaults to stdout)",
    )
    parser.add_argument(
        "--uri",
        help="RTTM recording URI/file-id (default: JSON metadata URI or input stem)",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "diarization", "segments", "lines"],
        default="auto",
        help="JSON field to convert (default: auto)",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=0.0,
        help="Merge adjacent same-speaker turns separated by this many seconds (default: 0)",
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Preserve input order instead of sorting turns by start time",
    )
    return parser.parse_args()


def normalize_speaker(speaker: Any) -> str | None:
    if speaker in (None, "", -1, "-1"):
        return None
    speaker_id = str(speaker).strip()
    if not speaker_id:
        return None
    if speaker_id.upper().startswith("SPEAKER_"):
        return speaker_id
    if speaker_id.isdigit():
        return f"SPEAKER_{int(speaker_id):02d}"
    return re.sub(r"\s+", "_", speaker_id)


def load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except OSError as exc:
        raise SystemExit(f"Error reading {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Error parsing {path}: {exc}") from exc


def select_items(payload: Any, source: str) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise SystemExit("Error: expected a JSON object or list")

    if source != "auto":
        items = payload.get(source)
        if isinstance(items, list):
            return items
        raise SystemExit(f"Error: JSON field '{source}' is not a list")

    for key in ("diarization", "segments", "lines"):
        items = payload.get(key)
        if isinstance(items, list) and items:
            return items
    return []


def payload_uri(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        uri = metadata.get("uri") or metadata.get("file") or metadata.get("filename")
        if uri:
            return Path(str(uri)).stem
    for key in ("uri", "file", "filename"):
        if payload.get(key):
            return Path(str(payload[key])).stem
    return None


def extract_turns(items: list[Any]) -> list[tuple[float, float, str]]:
    turns: list[tuple[float, float, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        speaker = normalize_speaker(item.get("speaker"))
        if speaker is None:
            continue
        try:
            start = float(item.get("start", 0.0) or 0.0)
            end = float(item.get("end", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if end > start:
            turns.append((start, end, speaker))
    return turns


def merge_turns(
    turns: list[tuple[float, float, str]], merge_gap: float
) -> list[tuple[float, float, str]]:
    if merge_gap < 0:
        raise SystemExit("Error: --merge-gap must be >= 0")
    if not turns:
        return []

    merged = [turns[0]]
    for start, end, speaker in turns[1:]:
        prev_start, prev_end, prev_speaker = merged[-1]
        if speaker == prev_speaker and start - prev_end <= merge_gap:
            merged[-1] = (prev_start, max(prev_end, end), prev_speaker)
        else:
            merged.append((start, end, speaker))
    return merged


def rttm_lines(uri: str, turns: list[tuple[float, float, str]]) -> list[str]:
    return [
        f"SPEAKER {uri} 1 {start:.3f} {end - start:.3f} <NA> <NA> {speaker} <NA> <NA>"
        for start, end, speaker in turns
    ]


def main() -> int:
    args = parse_args()
    payload = load_json(args.input)
    uri = args.uri or payload_uri(payload) or args.input.stem
    items = select_items(payload, args.source)
    turns = extract_turns(items)
    if not args.no_sort:
        turns.sort(key=lambda turn: (turn[0], turn[1], turn[2]))
    turns = merge_turns(turns, args.merge_gap)
    output = "\n".join(rttm_lines(uri, turns))
    if output:
        output += "\n"

    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
