#!/usr/bin/env python3

from __future__ import annotations

import shutil
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from coro.cli_support import fail, invoke, new_app
from coro.core.logger import get_logger

log = get_logger(__name__)


BASE_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
ANNOTATIONS_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"


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
        log.info("skip existing", path=str(output_path))
        return

    tmp_path = output_path.with_suffix(output_path.suffix + ".part")

    log.info("download", url=url, path=str(output_path))

    try:
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as f:
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


class AmiMirrorDownloads:
    """Adapter of the AMI downloads port: the Edinburgh AMI corpus mirror."""

    def download_meeting_audio(self, meeting_id: str, out_dir: Path) -> Path:
        """Download the Mix-Headset audio of one meeting under ``out_dir``."""
        return download_meeting_audio(meeting_id, out_dir)

    def download_annotations(self, out_dir: Path) -> Path:
        """Download the AMI manual annotations archive into ``out_dir``."""
        return download_annotations(out_dir)


def read_meetings_file(path: Path) -> list[str]:
    meetings = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            meetings.append(line)
    return meetings


app = new_app("ami_audios_download", help="Download AMI meeting audio and annotations.")


@app.default
def _download(
    *meetings: str,
    meetings_file: Path | None = None,
    out_dir: Path = Path("amicorpus"),
    mic: Literal["mix-headset", "ihm-mix", "sdm"] = "mix-headset",
    annotations: bool = False,
    force: bool = False,
) -> None:
    """Download AMI meeting audio and annotations.

    Args:
        meetings: AMI meeting IDs, e.g. EN2001a IB4001 IN1001
        meetings_file: Text file with one meeting ID per line.
        out_dir: Directory the corpus is written under.
        mic: mix-headset/ihm-mix = Mix-Headset WAV; sdm = Array1-01 WAV.
        annotations: Also download AMI manual annotations zip.
        force: Re-download files even if they already exist.

    """
    selected = list(meetings)

    if meetings_file:
        selected.extend(read_meetings_file(meetings_file))

    if not selected and not annotations:
        fail("ami_audios_download", "Provide at least one meeting ID or use --annotations.")

    for meeting_id in selected:
        download_meeting_audio(meeting_id=meeting_id, out_dir=out_dir, mic=mic, force=force)

    if annotations:
        download_annotations(out_dir, force=force)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the AMI download command line."""
    invoke(app, argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
