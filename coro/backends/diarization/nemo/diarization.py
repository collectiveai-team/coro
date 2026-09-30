"""NeMo batch Sortformer ML Model Integration."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
import torch

from coro.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
from coro.backends.diarization.nemo.postprocessing import (
    DEFAULT_MAX_SPEAKERS,
    apply_gated_postprocessing,
    resolve_postprocessing_yaml,
)
from coro.backends.diarization.segments import convert_diarization_segments
from coro.core.models import SpeakerSegment

logger = logging.getLogger(__name__)


class NemoDiarizationAdapter:
    """DiarizationAdapter that wraps a NeMo Sortformer model.

    Concurrency policy: **concurrent**. The adapter calls the model's
    ``forward`` directly instead of NeMo's ``diarize()`` convenience wrapper.
    ``diarize()`` keeps per-call state on the shared model instance
    (``_diarize_audio_rttm_map``, preprocessor ``dither``/``pad_to``, the
    train/eval mode and NeMo's global log level) and restores it afterwards, so
    overlapping calls clobber each other — one failed with a ``KeyError`` naming
    the other's temp file (issue #81). ``forward`` in eval mode only reads the
    model, so overlapping requests are independent: per-call state lives in
    locals, and post-processing is the same shared helper the Streaming
    Diarization Flow uses.
    """

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
        by both Diarization Flows. See ADR 0010.
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
        """Diarize one recording with state that lives only in this call.

        The raw speaker-activity matrix feeds the shared gated post-processing,
        so the Speaker-Count Post-Processing Gate is evaluated without a second
        inference pass. See ADR 0010.
        """
        duration = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        preds = self._predict(pcm)
        segments = apply_gated_postprocessing(
            preds,
            n_spk=preds.shape[-1],
            postprocessing_yaml=self._postprocessing_yaml,
            subsampling_factor=self._subsampling_factor(),
            max_speakers=self._max_speakers,
        )
        return convert_diarization_segments(segments, duration=duration)

    def _predict(self, pcm: bytes) -> torch.Tensor:
        """Run the Sortformer forward pass; returns ``(1, frames, speakers)`` sigmoids."""
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        device = self._model.device
        with torch.inference_mode():
            signal = torch.from_numpy(audio).unsqueeze(0).to(device)
            length = torch.tensor([audio.shape[0]], dtype=torch.long, device=device)
            preds = self._model.forward(audio_signal=signal, audio_signal_length=length)
        return preds.detach().cpu()

    def _subsampling_factor(self) -> int:
        cfg = getattr(self._model, "_cfg", None)
        encoder = getattr(cfg, "encoder", None) if cfg is not None else None
        return int(getattr(encoder, "subsampling_factor", 8) or 8)


def prepare_for_inference(model) -> None:
    """Put a Sortformer model into the fixed state ``diarize()`` would set per call.

    NeMo's ``diarize()`` switches to eval mode and zeroes the preprocessor's
    ``dither`` and ``pad_to`` on entry, then restores them on exit — shared
    writes that race under concurrency. Setting them once at load keeps
    ``forward`` numerically identical to ``diarize()`` while leaving it
    read-only for the lifetime of the server.
    """
    model.eval()
    featurizer = getattr(getattr(model, "preprocessor", None), "featurizer", None)
    if featurizer is None:
        return
    if hasattr(featurizer, "dither"):
        featurizer.dither = 0.0
    if hasattr(featurizer, "pad_to"):
        featurizer.pad_to = 0


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
            unconfigured baseline. Resolved once here; see ADR 0010.
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
    prepare_for_inference(model)
    logger.info("Diarization model loaded on device '%s'.", getattr(model, "device", "unknown"))
    return NemoDiarizationAdapter(
        model,
        postprocessing_yaml=postprocessing_yaml,
        max_speakers=max_speakers,
    )
