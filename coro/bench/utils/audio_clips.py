"""Shared audio helpers for materialising benchmark clips.

Corpus materialisers turn arbitrary encoded audio (wav, flac, opus, mp3) into
the 16 kHz mono WAV clips a ``--clips-dir`` **Workload Set** expects. Requires
ffmpeg/ffprobe on PATH.
"""

from __future__ import annotations

from collections.abc import Sequence
import subprocess
import wave
from pathlib import Path

_FFMPEG_WAV_ARGS = ("-ac", "1", "-ar", "16000")
_PCM16_SAMPLE_RATE = 16000


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


def concat_wav_clips(
    clip_paths: Sequence[Path],
    dst: Path,
    *,
    gap_seconds: float = 1.0,
) -> list[tuple[float, float]]:
    """Concatenate 16 kHz mono PCM16 WAV clips into one file, silence-gapped.

    Every input clip must already be a 16 kHz mono PCM16 WAV (e.g. produced by
    :func:`transcode_bytes_to_wav`/:func:`transcode_to_wav`) -- this function
    does no resampling of its own, so mismatched inputs fail loudly rather
    than silently producing a corrupt or mistimed concatenation.

    Args:
        clip_paths: Clips in the order they should appear in the output.
        dst: Destination WAV path; parent directories are created.
        gap_seconds: Silence inserted between consecutive clips (not before
            the first or after the last).

    Returns:
        The ``(start_seconds, end_seconds)`` span of each input clip within
        the concatenated output, in the same order as ``clip_paths``.

    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    gap_frames = round(gap_seconds * _PCM16_SAMPLE_RATE)
    silence = b"\x00\x00" * gap_frames

    spans: list[tuple[float, float]] = []
    chunks: list[bytes] = []
    cursor_frames = 0
    for index, path in enumerate(clip_paths):
        with wave.open(str(path), "rb") as handle:
            if (
                handle.getframerate() != _PCM16_SAMPLE_RATE
                or handle.getnchannels() != 1
                or handle.getsampwidth() != 2
            ):
                raise ValueError(f"{path} is not a 16 kHz mono PCM16 WAV")
            frame_count = handle.getnframes()
            data = handle.readframes(frame_count)

        if index > 0:
            chunks.append(silence)
            cursor_frames += gap_frames

        start = cursor_frames / _PCM16_SAMPLE_RATE
        chunks.append(data)
        cursor_frames += frame_count
        spans.append((start, cursor_frames / _PCM16_SAMPLE_RATE))

    with wave.open(str(dst), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(_PCM16_SAMPLE_RATE)
        out.writeframes(b"".join(chunks))

    return spans
