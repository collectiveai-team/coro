"""An in-memory adapter of the AMI downloads port, with a small annotation archive."""

from __future__ import annotations

import zipfile
from collections.abc import Sequence
from pathlib import Path

from coro.bench.ami import get_audio_path
from support.corpus import write_silent_wav

ANNOTATIONS_ZIP = "ami_public_manual_1.6.2.zip"

_WORDS_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<nite:root xmlns:nite="http://nite.sourceforge.net/">\n'
    '  <w nite:id="{meeting}.A.words0" starttime="0.0" endtime="0.5">hello</w>\n'
    '  <w nite:id="{meeting}.A.words1" starttime="0.5" endtime="1.0">world</w>\n'
    "</nite:root>\n"
)


def annotation_file(meeting_id: str) -> tuple[str, str]:
    """The archive name and XML of the words of one meeting.

    Speaker A says "hello world" in the first second.
    """
    return f"words/{meeting_id}.A.words.xml", _WORDS_XML.format(meeting=meeting_id)


def reference_stm(meeting_id: str) -> str:
    """The Reference STM that :func:`annotation_file` of ``meeting_id`` converts to."""
    return f"{meeting_id} 1 A 0.000 1.000 hello world\n"


def write_annotations(ami_root: Path, meeting_ids: Sequence[str]) -> None:
    """Write the extracted annotation tree of ``meeting_ids`` under ``ami_root``."""
    for meeting_id in meeting_ids:
        name, text = annotation_file(meeting_id)
        path = ami_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


class InMemoryAmiDownloads:
    """The AMI downloads port without the network.

    Audio is a silent WAV and the annotations archive annotates ``annotated``.
    Like the mirror adapter, an audio file that is already present is kept.
    """

    def __init__(self, annotated: Sequence[str] = ("IB4001", "IN1001")) -> None:
        self.annotated = list(annotated)
        self.audio_downloads: list[str] = []
        self.annotation_downloads = 0

    def download_meeting_audio(self, meeting_id: str, out_dir: Path) -> Path:
        """Write a silent WAV at the meeting's audio path unless one is there."""
        self.audio_downloads.append(meeting_id)
        path = get_audio_path(out_dir, meeting_id)
        if not path.exists():
            write_silent_wav(path)
        return path

    def download_annotations(self, out_dir: Path) -> Path:
        """Write the annotations zip archive into ``out_dir``."""
        self.annotation_downloads += 1
        zip_path = out_dir / ANNOTATIONS_ZIP
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w") as archive:
            for meeting_id in self.annotated:
                archive.writestr(*annotation_file(meeting_id))
        return zip_path
