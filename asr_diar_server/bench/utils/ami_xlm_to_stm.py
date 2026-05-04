#!/usr/bin/env python3

"""CLI wrapper around asr_diar_server.bench.stm.ami_meeting_to_stm.

Converts AMI meeting annotations to Reference STM format.
All conversion logic lives in the library module.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from asr_diar_server.bench.stm import ami_meeting_to_stm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert AMI meeting annotations to STM format.",
    )
    parser.add_argument("--ami-root", type=Path, default=Path("amicorpus"))
    parser.add_argument("--out-dir", type=Path, default=Path("stm"))
    parser.add_argument("meetings", nargs="+")
    args = parser.parse_args()

    for meeting in args.meetings:
        out_path = args.out_dir / f"{meeting}.ref.stm"
        stm_text = ami_meeting_to_stm(args.ami_root, meeting)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(stm_text, encoding="utf-8")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
