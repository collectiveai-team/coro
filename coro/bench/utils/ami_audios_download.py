#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path


BASE_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
ANNOTATIONS_URL = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/"
    "ami_public_manual_1.6.2.zip"
)


def audio_filename(meeting_id: str, mic: str) -> str:
    match mic:
        case "mix-headset" | "ihm-mix":
            return f"{meeting_id}.Mix-Headset.wav"
        case "sdm":
            return f"{meeting_id}.Array1-01.wav"
        case _:
            raise ValueError(f"Unsupported mic type: {mic}")


def audio_url(meeting_id: str, mic: str) -> str:
    filename = audio_filename(meeting_id, mic)
    return f"{BASE_URL}/{meeting_id}/audio/{filename}"


def download_file(url: str, output_path: Path, *, force: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        print(f"skip existing: {output_path}")
        return

    tmp_path = output_path.with_suffix(output_path.suffix + ".part")

    print(f"download: {url}")
    print(f"     to: {output_path}")

    try:
        with urllib.request.urlopen(url) as response:
            with tmp_path.open("wb") as f:
                shutil.copyfileobj(response, f)
    except urllib.error.HTTPError as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"HTTP {e.code} while downloading {url}") from e
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    tmp_path.replace(output_path)


def download_meeting_audio(
    meeting_id: str,
    out_dir: Path,
    *,
    mic: str = "mix-headset",
    force: bool = False,
) -> Path:
    filename = audio_filename(meeting_id, mic)
    output_path = out_dir / meeting_id / "audio" / filename
    download_file(audio_url(meeting_id, mic), output_path, force=force)
    return output_path


def download_annotations(out_dir: Path, *, force: bool = False) -> Path:
    output_path = out_dir / "ami_public_manual_1.6.2.zip"
    download_file(ANNOTATIONS_URL, output_path, force=force)
    return output_path


def read_meetings_file(path: Path) -> list[str]:
    meetings = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            meetings.append(line)
    return meetings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "meetings",
        nargs="*",
        help="AMI meeting IDs, e.g. EN2001a IB4001 IN1001",
    )
    parser.add_argument(
        "--meetings-file",
        type=Path,
        help="Text file with one meeting ID per line.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("amicorpus"),
    )
    parser.add_argument(
        "--mic",
        default="mix-headset",
        choices=["mix-headset", "ihm-mix", "sdm"],
        help="mix-headset/ihm-mix = Mix-Headset WAV; sdm = Array1-01 WAV.",
    )
    parser.add_argument(
        "--annotations",
        action="store_true",
        help="Also download AMI manual annotations zip.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    args = parser.parse_args()

    meetings = list(args.meetings)

    if args.meetings_file:
        meetings.extend(read_meetings_file(args.meetings_file))

    if not meetings and not args.annotations:
        parser.error("Provide at least one meeting ID or use --annotations.")

    for meeting_id in meetings:
        download_meeting_audio(
            meeting_id=meeting_id,
            out_dir=args.out_dir,
            mic=args.mic,
            force=args.force,
        )

    if args.annotations:
        download_annotations(args.out_dir, force=args.force)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())