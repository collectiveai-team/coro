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
from coro.backends.diarization.nemo.postprocessing import (
    DEFAULT_MAX_SPEAKERS,
    baseline_postprocessing_params,
    postprocessing_gate_open,
    resolve_postprocessing_yaml,
    segments_from_predictions,
)
from coro.backends.diarization.segments import convert_diarization_segments
from coro.core.models import SpeakerSegment

logger = logging.getLogger(__name__)


class NemoDiarizationAdapter:
    """DiarizationAdapter that wraps a NeMo Sortformer model."""

    def __init__(
        self,
        model,
        *,
        postprocessing_yaml: str | None = None,
        max_speakers: int = DEFAULT_MAX_SPEAKERS,
    ) -> None:
        self._model = model
        self._postprocessing_yaml = postprocessing_yaml
        self._max_speakers = max_speakers

    @property
    def model(self):
        """The wrapped Sortformer model.

        Exposed so the diarization Backend Adapter Factory can build the
        streaming diarizer from the same shared model without reaching into a
        private attribute.
        """
        return self._model

    @property
    def postprocessing_yaml(self) -> str | None:
        """The resolved Diarization Post-Processing Configuration path.

        Exposed so the Backend Adapter Factory can build the streaming
        diarizer factory from the same resolved value — resolved once, shared
        by both Diarization Flows. See ADR 0009.
        """
        return self._postprocessing_yaml

    @property
    def max_speakers(self) -> int:
        """The Speaker-Count Post-Processing Gate ceiling in force.

        Exposed alongside ``postprocessing_yaml`` so the Backend Adapter
        Factory can build a streaming factory that gates identically.
        """
        return self._max_speakers

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
            # include_tensor_outputs returns the raw speaker-activity matrix
            # alongside NeMo's own post-processed segments, so the
            # Speaker-Count Post-Processing Gate can be evaluated without a
            # second inference pass. See ADR 0009.
            predicted, preds_list = self._model.diarize(
                audio=path,
                batch_size=1,
                include_tensor_outputs=True,
                postprocessing_yaml=self._postprocessing_yaml,
            )
        finally:
            with contextlib.suppress(OSError):
                Path(path).unlink()

        if len(predicted) == 1 and isinstance(predicted[0], list):
            predicted = predicted[0]

        gated = self._gated_segments(preds_list)
        if gated is not None:
            return convert_diarization_segments(gated, duration=duration)
        return convert_diarization_segments(predicted, duration=duration)

    def _gated_segments(self, preds_list) -> list[tuple[float, float, int]] | None:
        """Re-derive segments without the tuned thresholds when the gate closes.

        Returns ``None`` in the common case, meaning NeMo's own post-processed
        output stands. Segments are only recomputed when a tuned Diarization
        Post-Processing Configuration is active *and* the estimated speaker
        count exceeds the ceiling, because tuned short-segment deletion is
        reported to degrade DER in exactly that range.
        """
        if self._postprocessing_yaml is None or not preds_list:
            return None

        preds = preds_list[0]
        subsampling_factor = self._subsampling_factor()
        gate_open, estimated = postprocessing_gate_open(
            preds,
            subsampling_factor=subsampling_factor,
            max_speakers=self._max_speakers,
        )
        if gate_open:
            return None

        logger.info(
            "batch diarization postprocessing gate closed estimated_speakers=%d "
            "max_speakers=%d — reverting to the NeMo baseline for this recording",
            estimated,
            self._max_speakers,
        )
        n_spk = preds.shape[-1]
        return segments_from_predictions(
            preds,
            n_spk=n_spk,
            params=baseline_postprocessing_params(),
            subsampling_factor=subsampling_factor,
        )

    def _subsampling_factor(self) -> int:
        cfg = getattr(self._model, "_cfg", None)
        encoder = getattr(cfg, "encoder", None) if cfg is not None else None
        return int(getattr(encoder, "subsampling_factor", 8) or 8)


def build_nemo_diarization_adapter(
    model_diarization: str,
    *,
    device: str = "auto",
    postprocessing: str | None = None,
    max_speakers: int = DEFAULT_MAX_SPEAKERS,
) -> NemoDiarizationAdapter:
    """Construct and return a NemoDiarizationAdapter.

    Args:
        model_diarization: Diarization Model Selection.
        device: ``auto``/``cuda``/``cpu`` device selector.
        postprocessing: Diarization Post-Processing Configuration value — a
            preset name, a custom YAML path, or ``None`` to keep NeMo's own
            unconfigured baseline. Resolved once here; see ADR 0009.
        max_speakers: Speaker-Count Post-Processing Gate ceiling. Above this
            estimated speaker count the tuned thresholds are bypassed.

    """
    from nemo.collections.asr.models import SortformerEncLabelModel

    postprocessing_yaml = resolve_postprocessing_yaml(postprocessing)

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
    return NemoDiarizationAdapter(
        model,
        postprocessing_yaml=postprocessing_yaml,
        max_speakers=max_speakers,
    )
