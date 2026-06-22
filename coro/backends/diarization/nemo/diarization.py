"""NeMo batch Sortformer ML Model Integration."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import tempfile
import wave
from pathlib import Path
from typing import Any

import torch

from coro.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
from coro.backends.diarization.segments import convert_diarization_segments
from coro.core.models import SpeakerSegment

logger = logging.getLogger(__name__)


class NemoDiarizationAdapter:
    """DiarizationAdapter that wraps a NeMo Sortformer model."""

    def __init__(self, model) -> None:
        self._model = model

    async def diarize_pcm(self, pcm: bytes) -> list[SpeakerSegment]:
        """Run batch diarization over full PCM audio."""
        return await asyncio.to_thread(self._diarize_sync, pcm)

    def _diarize_sync(self, pcm: bytes) -> list[SpeakerSegment]:
        duration = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        fd, path = tempfile.mkstemp(prefix="coro-nemo-", suffix=".wav")
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


def build_diarization_adapter(
    model_diarization: str,
    *,
    device: str = "auto",
) -> NemoDiarizationAdapter:
    """Construct and return a NemoDiarizationAdapter."""
    from nemo.collections.asr.models import SortformerEncLabelModel

    logger.info(
        "Loading diarization model '%s' with NeMo on device '%s'.",
        model_diarization,
        device,
    )
    map_location = torch.device(device) if device != "auto" else None
    model: Any = SortformerEncLabelModel.from_pretrained(
        model_diarization,
        map_location=map_location,
    )
    model.eval()
    logger.info("Diarization model loaded on device '%s'.", getattr(model, "device", "unknown"))
    return NemoDiarizationAdapter(model)
