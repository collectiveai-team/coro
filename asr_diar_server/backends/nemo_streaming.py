"""Streaming Sortformer diarizer."""

from __future__ import annotations

from collections.abc import Callable
import logging
import time

import numpy as np
import torch

from asr_diar_server.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
from asr_diar_server.backends.nemo import convert_diarization_segments
from asr_diar_server.core.types import SpeakerSegment

logger = logging.getLogger(__name__)

LATENCY_TIER_PARAMS: dict[str, dict[str, int]] = {
    "very-high": {
        "chunk_len": 340,
        "chunk_right_context": 40,
        "fifo_len": 40,
        "spkcache_update_period": 300,
        "spkcache_len": 188,
    },
    "high": {
        "chunk_len": 124,
        "chunk_right_context": 1,
        "fifo_len": 124,
        "spkcache_update_period": 124,
        "spkcache_len": 188,
    },
    "low": {
        "chunk_len": 6,
        "chunk_right_context": 7,
        "fifo_len": 188,
        "spkcache_update_period": 144,
        "spkcache_len": 188,
    },
    "ultra-low": {
        "chunk_len": 3,
        "chunk_right_context": 1,
        "fifo_len": 188,
        "spkcache_update_period": 144,
        "spkcache_len": 188,
    },
}


def get_latency_tier_params(tier: str) -> dict[str, int]:
    return dict(LATENCY_TIER_PARAMS[tier])


class StreamingDiarizerFactory:
    """Produces fresh per-request StreamingDiarizer instances bound to a shared model."""

    def __init__(self, model, *, tier: str = "very-high") -> None:
        self._model = model
        self._tier = tier
        self._tier_params = get_latency_tier_params(tier)
        subsampling_factor = getattr(model.sortformer_modules, "subsampling_factor", 8)
        n_spk = getattr(model.sortformer_modules, "n_spk", 4)
        model.sortformer_modules.chunk_len = self._tier_params["chunk_len"]
        model.sortformer_modules.chunk_right_context = self._tier_params["chunk_right_context"]
        model.sortformer_modules.fifo_len = self._tier_params["fifo_len"]
        model.sortformer_modules.spkcache_update_period = (
            self._tier_params["spkcache_update_period"]
        )
        model.sortformer_modules.spkcache_len = self._tier_params["spkcache_len"]
        model.sortformer_modules._check_streaming_parameters()
        self._subsampling_factor = subsampling_factor
        self._n_spk = n_spk

    def __call__(self) -> StreamingDiarizer:
        return StreamingDiarizer(
            self._model,
            chunk_len=self._tier_params["chunk_len"],
            subsampling_factor=self._subsampling_factor,
            n_spk=self._n_spk,
        )


class StreamingDiarizer:
    """Encapsulates streaming Sortformer diarization behind a two-method interface."""

    def __init__(
        self,
        model,
        *,
        chunk_len: int = 6,
        subsampling_factor: int = 8,
        n_spk: int = 4,
        preprocessor=None,
        post_processor: Callable | None = None,
    ):
        self._model = model
        self._device = model.device
        self._chunk_len = chunk_len
        self._subsampling_factor = subsampling_factor
        self._n_spk = n_spk
        self._preprocessor = preprocessor
        self._post_processor = post_processor

        chunk_audio_seconds = chunk_len * subsampling_factor * 0.01
        self._chunk_audio_bytes = int(chunk_audio_seconds * SAMPLE_RATE * BYTES_PER_SAMPLE)

        self._pcm_buffer = b""
        self._streaming_state = model.sortformer_modules.init_streaming_state(
            batch_size=1,
            async_streaming=False,
            device=self._device,
        )
        # Initialize as empty accumulator matching what forward_streaming_step expects;
        # NeMo uses torch.zeros((batch, 0, n_spk)) as the seed before the first chunk.
        self._total_preds: torch.Tensor = torch.zeros(
            (1, 0, self._n_spk), device=self._device
        )
        self._total_audio_bytes = 0
        self._processed_chunks = 0

    @property
    def processed_chunks(self) -> int:
        return self._processed_chunks

    def ingest_pcm_chunk(self, pcm: bytes) -> None:
        self._pcm_buffer += pcm
        while len(self._pcm_buffer) >= self._chunk_audio_bytes:
            chunk = self._pcm_buffer[: self._chunk_audio_bytes]
            self._pcm_buffer = self._pcm_buffer[self._chunk_audio_bytes :]
            self._process_chunk(chunk)

    def finalize(self) -> list[SpeakerSegment]:
        if self._pcm_buffer:
            logger.info(
                "streaming_diarizer finalize flush_remainder bytes=%d processed_chunks=%d",
                len(self._pcm_buffer),
                self._processed_chunks,
            )
            padded = self._pcm_buffer + b"\x00" * (self._chunk_audio_bytes - len(self._pcm_buffer))
            self._pcm_buffer = b""
            self._process_chunk(padded)

        if self._total_preds.shape[1] == 0:
            logger.info("streaming_diarizer finalize no_predictions processed_chunks=%d", self._processed_chunks)
            return []

        duration = self._total_audio_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        logger.info(
            "streaming_diarizer finalize predictions_shape=%s duration=%.2fs processed_chunks=%d",
            tuple(self._total_preds.shape),
            duration,
            self._processed_chunks,
        )

        if self._post_processor is not None:
            raw_segments = self._post_processor(self._total_preds, self._n_spk)
            return convert_diarization_segments(raw_segments, duration=duration)

        return self._default_post_process(duration)

    def _default_post_process(self, duration: float) -> list[SpeakerSegment]:
        """Run per-speaker VAD post-processing matching the NeMo model's own approach."""
        started = time.perf_counter()
        from nemo.collections.asr.models.sortformer_diar_models import ts_vad_post_processing
        from nemo.collections.asr.parts.mixins.diarization import load_postprocessing_from_yaml

        cfg_vad_params = load_postprocessing_from_yaml(None)
        # total_preds: (1, n_frames, n_spk) — process each speaker independently
        preds_cpu = self._total_preds.squeeze(0).cpu()  # (n_frames, n_spk)
        subsampling_factor = self._subsampling_factor

        raw_segments: list[tuple[float, float, int]] = []
        for spk_id in range(self._n_spk):
            spk_preds = preds_cpu[:, spk_id]  # (n_frames,)
            ts_mat = ts_vad_post_processing(
                spk_preds,
                cfg_vad_params=cfg_vad_params,
                unit_10ms_frame_count=subsampling_factor,
                bypass_postprocessing=False,
            )
            for start, end in ts_mat.detach().cpu().tolist():
                raw_segments.append((start, end, spk_id))

        segments = convert_diarization_segments(raw_segments, duration=duration)
        logger.info(
            "streaming_diarizer post_process complete elapsed=%.3fs raw_segments=%d segments=%d",
            time.perf_counter() - started,
            len(raw_segments),
            len(segments),
        )
        return segments

    def _get_preprocessor(self):
        if self._preprocessor is not None:
            return self._preprocessor
        from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor

        self._preprocessor = AudioToMelSpectrogramPreprocessor(
            window_size=0.025, normalize="NA", n_fft=512, features=128, pad_to=0,
        ).to(self._device)
        return self._preprocessor

    def _process_chunk(self, pcm: bytes) -> None:
        """Run one chunk through the streaming diarizer.

        The NeMo streaming_state (spkcache + fifo) carries all historical
        left-context internally — we pass only the current chunk's mel frames.
        forward_streaming_step expects (batch, time, features), i.e. time-first,
        which is the opposite of the preprocessor's (batch, features, time) output.
        """
        audio_np = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).to(self._device)

        preprocessor = self._get_preprocessor()
        mel, mel_len = preprocessor(
            input_signal=audio_tensor,
            length=torch.tensor([len(audio_np)], device=self._device),
        )

        # Transpose from (batch, features, time) → (batch, time, features)
        signal_t = mel.transpose(1, 2)

        self._streaming_state, self._total_preds = self._model.forward_streaming_step(
            signal_t,
            mel_len,
            self._streaming_state,
            self._total_preds,
            left_offset=0,
            right_offset=0,
        )

        self._total_audio_bytes += len(pcm)
        self._processed_chunks += 1
        if self._processed_chunks == 1 or self._processed_chunks % 5 == 0:
            logger.info(
                "streaming_diarizer chunk=%d pcm_bytes=%d mel_shape=%s preds_shape=%s",
                self._processed_chunks,
                len(pcm),
                tuple(mel.shape),
                tuple(self._total_preds.shape),
            )
