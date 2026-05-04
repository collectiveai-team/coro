#!/usr/bin/env python3

"""CLI wrapper around asr_diar_server.bench.stm.hyp_segments_to_stm.

Converts an OpenAI diarized_json file to STM format.
All conversion logic lives in the library module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from asr_diar_server.bench.stm import hyp_segments_to_stm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert OpenAI diarized_json to STM format.",
    )
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output_stm", type=Path)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--channel", default="1")
    args = parser.parse_args()

    data = json.loads(args.input_json.read_text(encoding="utf-8"))
    segments = data if isinstance(data, list) else data.get("segments", [])

    stm_text = hyp_segments_to_stm(segments, args.recording_id, channel=args.channel)

    args.output_stm.parent.mkdir(parents=True, exist_ok=True)
    args.output_stm.write_text(stm_text, encoding="utf-8")


if __name__ == "__main__":
    main()
