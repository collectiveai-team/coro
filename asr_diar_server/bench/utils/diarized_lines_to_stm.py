#!/usr/bin/env python3

"""Convert a diarized transcript JSON (``lines``/``segments``) to a reference STM.

Builds a speaker-attributed reference STM from a transcript JSON whose entries
carry ``{start, end, text, speaker}`` — e.g. the curated Spanish RNE14 reference
under benchmark/groundtruth/. Conversion logic is reused from
asr_diar_server.bench.stm.hyp_segments_to_stm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from asr_diar_server.bench.stm import hyp_segments_to_stm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a diarized transcript JSON to reference STM.",
    )
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output_stm", type=Path)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument(
        "--field",
        default="lines",
        help="JSON key holding the diarized segments (default: 'lines').",
    )
    parser.add_argument("--channel", default="1")
    args = parser.parse_args()

    data = json.loads(args.input_json.read_text(encoding="utf-8"))
    segments = data if isinstance(data, list) else data.get(args.field, [])

    stm_text = hyp_segments_to_stm(segments, args.recording_id, channel=args.channel)

    args.output_stm.parent.mkdir(parents=True, exist_ok=True)
    args.output_stm.write_text(stm_text, encoding="utf-8")
    print(f"wrote {args.output_stm} ({len(stm_text.splitlines())} lines)")


if __name__ == "__main__":
    main()
