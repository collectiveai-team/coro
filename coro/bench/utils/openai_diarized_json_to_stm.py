#!/usr/bin/env python3

"""CLI wrapper around coro.bench.stm.hyp_segments_to_stm.

Converts an OpenAI diarized_json file to STM format.
All conversion logic lives in the library module.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from coro.bench.stm import hyp_segments_to_stm
from coro.cli_support import invoke, new_app

app = new_app("openai_diarized_json_to_stm", help="Convert OpenAI diarized_json to STM format.")


@app.default
def _convert(
    input_json: Path,
    output_stm: Path,
    /,
    *,
    recording_id: str,
    channel: str = "1",
) -> None:
    """Convert OpenAI diarized_json to STM format.

    Args:
        input_json: diarized_json response to read.
        output_stm: STM file to write.
        recording_id: Session id written into every STM line.
        channel: STM channel column.

    """
    data = json.loads(input_json.read_text(encoding="utf-8"))
    segments = data if isinstance(data, list) else data.get("segments", [])

    stm_text = hyp_segments_to_stm(segments, recording_id, channel=channel)

    output_stm.parent.mkdir(parents=True, exist_ok=True)
    output_stm.write_text(stm_text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    """Run the diarized_json converter command line."""
    invoke(app, argv)


if __name__ == "__main__":
    main()
