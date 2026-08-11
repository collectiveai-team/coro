"""Shared audio helpers for materialising benchmark clips.

Corpus materialisers turn arbitrary encoded audio (wav, flac, opus, mp3) into
the 16 kHz mono WAV clips a ``--clips-dir`` **Workload Set** expects. Requires
ffmpeg/ffprobe on PATH.
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

_FFMPEG_WAV_ARGS = ("-ac", "1", "-ar", "16000")


def transcode_bytes_to_wav(data: bytes, dst: Path) -> None:
    """Transcode in-memory encoded audio to a 16 kHz mono WAV at ``dst``.

    Audio is piped to ffmpeg on stdin so corpus rows never touch a temporary
    file on the way to the clip.

    Args:
        data: Encoded audio bytes exactly as stored by the corpus.
        dst: Destination WAV path; parent directories are created.

    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", "pipe:0", *_FFMPEG_WAV_ARGS, str(dst)],
        input=data,
        check=True,
    )


def transcode_to_wav(src: Path, dst: Path) -> None:
    """Transcode an audio file to a 16 kHz mono WAV at ``dst``."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), *_FFMPEG_WAV_ARGS, str(dst)],
        check=True,
    )


def wav_duration_seconds(path: Path) -> float:
    """Return the duration of a WAV file in seconds (0.0 when unreadable)."""
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
    return frames / rate if rate else 0.0
