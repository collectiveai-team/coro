"""pyannote.audio ML Model Integration (batch-only diarization).

Wraps the ``pyannote/speaker-diarization-community-1`` pipeline as a batch
Diarization Adapter. This backend is offline/whole-file only: the pyannote
pipeline requires the complete audio to cluster speakers, so it is
incompatible with the Streaming Pipeline (rejected at startup in settings).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
import torch

from coro.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
from coro.backends.diarization.segments import convert_diarization_segments
from coro.core.models import SpeakerSegment

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> torch.device:
    """Resolve a settings device string to a concrete torch device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _pcm_to_waveform(pcm: bytes) -> torch.Tensor:
    """Convert 16-bit mono PCM bytes to a float32 ``(1, num_samples)`` tensor."""
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    return torch.from_numpy(samples).unsqueeze(0)


def _extract_annotation(output: Any) -> Any:
    """Return the speaker-diarization Annotation from a pyannote output.

    community-1 returns a result object exposing ``speaker_diarization``
    (and ``exclusive_speaker_diarization``); older pipelines return the
    Annotation directly. Handle both shapes.
    """
    annotation = getattr(output, "speaker_diarization", None)
    if annotation is not None:
        return annotation
    return output


class PyannoteDiarizationAdapter:
    """DiarizationAdapter that wraps a pyannote.audio pipeline."""

    def __init__(self, pipeline) -> None:
        self._pipeline = pipeline

    async def diarize_pcm(self, pcm: bytes) -> list[SpeakerSegment]:
        """Run batch diarization over full PCM audio."""
        return await asyncio.to_thread(self._diarize_sync, pcm)

    def _diarize_sync(self, pcm: bytes) -> list[SpeakerSegment]:
        duration = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        waveform = _pcm_to_waveform(pcm)
        output = self._pipeline(
            {"waveform": waveform, "sample_rate": SAMPLE_RATE},
        )
        annotation = _extract_annotation(output)
        native_segments = [
            (segment.start, segment.end, label)
            for segment, _track, label in annotation.itertracks(yield_label=True)
        ]
        return convert_diarization_segments(native_segments, duration=duration)


def build_pyannote_diarization_adapter(
    model_diarization: str,
    *,
    device: str = "auto",
    hf_token: str | None = None,
) -> PyannoteDiarizationAdapter:
    """Construct and return a PyannoteDiarizationAdapter.

    Args:
        model_diarization: pyannote pipeline id or local directory path.
        device: ``auto``/``cuda``/``cpu`` device selector.
        hf_token: HuggingFace access token for the gated model. May be
            ``None`` when using a local path or a cached/authenticated env.

    Returns:
        A ready-to-use PyannoteDiarizationAdapter.

    """
    from pyannote.audio import Pipeline

    target = _resolve_device(device)
    logger.info(
        "Loading diarization pipeline '%s' with pyannote.audio on device '%s'.",
        model_diarization,
        target,
    )
    pipeline = Pipeline.from_pretrained(model_diarization, token=hf_token)
    if pipeline is None:
        msg = (
            f"pyannote.audio could not load pipeline '{model_diarization}'. "
            "Verify the model id, that you accepted the user conditions, and "
            "that a valid HuggingFace token is configured (CORO_HF_TOKEN / "
            "HF_TOKEN)."
        )
        raise RuntimeError(msg)
    pipeline.to(target)
    logger.info("Diarization pipeline loaded on device '%s'.", target)
    return PyannoteDiarizationAdapter(pipeline)
