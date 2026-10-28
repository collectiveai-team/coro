#!/usr/bin/env python3

"""CLI wrapper around coro.bench.stm.ami_meeting_to_stm.

Converts AMI meeting annotations to Reference STM format.
All conversion logic lives in the library module.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from coro.bench.stm import ami_meeting_to_stm
from coro.cli_support import fail, invoke, new_app, out_console

app = new_app("ami_xlm_to_stm", help="Convert AMI meeting annotations to STM format.")


@app.default
def _convert(
    *meetings: str,
    ami_root: Path = Path("amicorpus"),
    out_dir: Path = Path("stm"),
) -> None:
    """Convert AMI meeting annotations to STM format.

    Args:
        meetings: AMI meeting IDs to convert; at least one.
        ami_root: Root of the AMI corpus.
        out_dir: Directory the Reference STM files are written into.

    """
    if not meetings:
        fail("ami_xlm_to_stm", "the following arguments are required: meetings")
    for meeting_id in meetings:
        out_path = out_dir / f"{meeting_id}.ref.stm"
        stm_text = ami_meeting_to_stm(ami_root, meeting_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(stm_text, encoding="utf-8")
        out_console.out(f"wrote {out_path}")


def main(argv: Sequence[str] | None = None) -> None:
    """Run the AMI annotation converter command line."""
    invoke(app, argv)


if __name__ == "__main__":
    main()
