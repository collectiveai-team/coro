"""NeMo ML Model Integration."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np

from asr_diar_server.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
from asr_diar_server.core.types import SpeakerSegment

logger = logging.getLogger(__name__)


def _speaker_to_one_indexed(speaker) -> int:
    """Convert a zero-indexed or string speaker label to 1-indexed int."""
    if isinstance(speaker, int):
        return speaker + 1
    if isinstance(speaker, np.integer):
        return int(speaker) + 1
    match = re.search(r"\d+", str(speaker))
    if match:
        return int(match.group(0)) + 1
    return 1


def _coerce_diarization_segment(seg):
    if isinstance(seg, str):
        parts = seg.replace(",", " ").split()
        if len(parts) >= 3:
            return float(parts[0]), float(parts[1]), parts[2]
    if isinstance(seg, (tuple, list)) and len(seg) >= 3:
        return float(seg[0]), float(seg[1]), seg[2]
    return (
        float(getattr(seg, "start", 0.0) or 0.0),
        float(getattr(seg, "end", 0.0) or 0.0),
        getattr(seg, "speaker", 0),
    )


def convert_diarization_segments(
    native_segments,
    *,
    duration: float,
) -> list[SpeakerSegment]:
    """Convert NeMo diarization segment objects to SpeakerSegments.

    Args:
        native_segments: Iterable of NeMo diarization outputs.
        duration: Total audio duration in seconds; end times are clamped.

    Returns:
        Deduplicated list of SpeakerSegment sorted by start time.

    """
    timeline: list[SpeakerSegment] = []
    seen: set[tuple[float, float, int]] = set()

    for seg in native_segments:
        start, end, speaker_label = _coerce_diarization_segment(seg)
        start = max(0.0, start)
        end = min(duration, end)
        if end <= start:
            continue
        speaker = _speaker_to_one_indexed(speaker_label)
        key = (round(start, 3), round(end, 3), speaker)
        if key in seen:
            continue
        seen.add(key)
        timeline.append(SpeakerSegment(start=key[0], end=key[1], speaker=speaker))

    timeline.sort(key=lambda s: s.start)
    return timeline


class NemoDiarizationAdapter:
    """DiarizationAdapter that wraps a NeMo Sortformer model."""

    def __init__(self, model) -> None:
        self._model = model

    async def diarize_pcm(self, pcm: bytes) -> list[SpeakerSegment]:
        """Run batch diarization over full PCM audio."""
        return await asyncio.to_thread(self._diarize_sync, pcm)

    def _diarize_sync(self, pcm: bytes) -> list[SpeakerSegment]:
        duration = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        fd, path = tempfile.mkstemp(prefix="asr-diar-nemo-", suffix=".wav")
        os.close(fd)
        try:
            with wave.open(path, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(BYTES_PER_SAMPLE)
                wav.setframerate(SAMPLE_RATE)
                wav.writeframes(pcm)
            predicted = self._model.diarize(audio=path, batch_size=1)
        finally:
            with contextlib.suppress(OSError):
                Path(path).unlink()

        if len(predicted) == 1 and isinstance(predicted[0], list):
            predicted = predicted[0]
        return convert_diarization_segments(predicted, duration=duration)


def build_diarization_adapter(model_diarization: str) -> NemoDiarizationAdapter:
    """Construct and return a NemoDiarizationAdapter."""
    from nemo.collections.asr.models import SortformerEncLabelModel

    logger.info("Loading diarization model '%s' with NeMo.", model_diarization)
    model: Any = SortformerEncLabelModel.from_pretrained(model_diarization)
    model.eval()
    logger.info("Diarization model loaded.")
    return NemoDiarizationAdapter(model)
