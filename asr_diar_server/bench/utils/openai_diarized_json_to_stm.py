#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def clean_text(text: str) -> str:
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_speaker(speaker: str, prefix: str = "SPEAKER") -> str:
    """
    OpenAI usually returns A, B, C... unless known speaker references are used.
    MeetEval does not require the same speaker names as the reference for cpWER,
    but stable labels are useful.
    """
    speaker = str(speaker).strip()

    if not speaker:
        return f"{prefix}_UNKNOWN"

    if speaker.startswith(prefix):
        return speaker

    return f"{prefix}_{speaker}"


def load_segments(data: Any) -> list[dict[str, Any]]:
    """
    Supports:
      1. OpenAI diarized_json: {"segments": [...]}
      2. A raw list of segments: [...]
    """
    if isinstance(data, dict) and isinstance(data.get("segments"), list):
        return data["segments"]

    if isinstance(data, list):
        return data

    raise ValueError("Expected OpenAI diarized JSON with a 'segments' list.")


def convert_to_stm(
    input_path: Path,
    output_path: Path,
    recording_id: str,
    channel: str = "1",
    speaker_prefix: str = "SPEAKER",
) -> None:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    segments = load_segments(data)

    lines: list[str] = []

    for segment in segments:
        start = segment.get("start")
        end = segment.get("end")
        text = clean_text(str(segment.get("text", "")))
        speaker = normalize_speaker(
            str(segment.get("speaker", "UNKNOWN")),
            prefix=speaker_prefix,
        )

        if start is None or end is None or not text:
            continue

        start_f = float(start)
        end_f = float(end)

        if end_f <= start_f:
            continue

        lines.append(
            f"{recording_id} {channel} {speaker} "
            f"{start_f:.3f} {end_f:.3f} {text}"
        )

    lines.sort(key=lambda line: (float(line.split()[3]), line.split()[2]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output_stm", type=Path)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--channel", default="1")
    parser.add_argument("--speaker-prefix", default="SPEAKER")
    args = parser.parse_args()

    convert_to_stm(
        input_path=args.input_json,
        output_path=args.output_stm,
        recording_id=args.recording_id,
        channel=args.channel,
        speaker_prefix=args.speaker_prefix,
    )


if __name__ == "__main__":
    main()