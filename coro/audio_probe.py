"""Audio duration from the container, before any decoding: ``ffprobe``.

Used to charge an upload against the audio-minutes rate limit before the
pipeline spends CPU on it (ADR 0024). The container's declared duration is
read first; containers that omit it (a browser ``MediaRecorder`` WebM is the
common case) fall back to the end of the last audio packet, which demuxes the
file without decoding it.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


async def _ffprobe(path: str, *args: str) -> str | None:
    process = await asyncio.create_subprocess_exec(
        "ffprobe",
        "-v",
        "error",
        *args,
        "-of",
        "csv=p=0",
        path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await process.communicate()
    if process.returncode != 0:
        return None
    return stdout.decode(errors="replace")


def _seconds(value: str) -> float | None:
    """Parse an ffprobe time; None for ``N/A``, garbage or a negative value."""
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def _positive(value: str) -> float | None:
    seconds = _seconds(value)
    return seconds if seconds else None


async def probe_duration_seconds(path: str) -> float | None:
    """Return the audio duration of the file at ``path``, or None if unknown.

    None means ffprobe could not read the file, or it has no audio stream; the
    pipeline's own decoder then decides whether the upload is valid.
    """
    declared = await _ffprobe(path, "-show_entries", "format=duration")
    if declared is not None and (seconds := _positive(declared.strip())) is not None:
        return seconds
    packets = await _ffprobe(
        path, "-select_streams", "a:0", "-show_entries", "packet=pts_time,duration_time"
    )
    if packets is None:
        return None
    end = None
    for line in packets.splitlines():
        # "pts,duration", sometimes with a trailing comma.
        fields = line.split(",")
        start = _seconds(fields[0])
        if start is not None:
            end = start + ((_seconds(fields[1]) if len(fields) > 1 else None) or 0.0)
    if not end:
        logger.info("audio_probe no duration for %s", path)
        return None
    return end
