"""Streaming Sortformer diarizer."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from asr_diar_server.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
from asr_diar_server.backends.nemo import convert_diarization_segments
from asr_diar_server.core.types import SpeakerSegment

_LEFT_CONTEXT_FRAMES = 99

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
            device=model.device,
        )
        self._total_preds: torch.Tensor | None = None
        self._left_context: torch.Tensor | None = None
        self._total_audio_bytes = 0

    def ingest_pcm_chunk(self, pcm: bytes) -> None:
        self._pcm_buffer += pcm
        while len(self._pcm_buffer) >= self._chunk_audio_bytes:
            chunk = self._pcm_buffer[: self._chunk_audio_bytes]
            self._pcm_buffer = self._pcm_buffer[self._chunk_audio_bytes :]
            self._process_chunk(chunk)

    def finalize(self) -> list[SpeakerSegment]:
        if self._pcm_buffer:
            padded = self._pcm_buffer + b"\x00" * (self._chunk_audio_bytes - len(self._pcm_buffer))
            self._pcm_buffer = b""
            self._process_chunk(padded)

        if self._total_preds is None:
            return []

        if self._post_processor is not None:
            raw_segments = self._post_processor(self._total_preds, self._n_spk)
        else:
            from nemo.collections.asr.modules import ts_vad_post_processing

            raw_segments = ts_vad_post_processing(self._total_preds, self._n_spk)

        duration = self._total_audio_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        return convert_diarization_segments(raw_segments, duration=duration)

    def _get_preprocessor(self):
        if self._preprocessor is not None:
            return self._preprocessor
        from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor

        self._preprocessor = AudioToMelSpectrogramPreprocessor(
            window_size=0.025, normalize="NA", n_fft=512, features=128, pad_to=0,
        )
        return self._preprocessor

    def _process_chunk(self, pcm: bytes) -> None:
        audio_np = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).to(self._model.device)

        preprocessor = self._get_preprocessor()
        mel, mel_len = preprocessor(audio_tensor, torch.tensor([len(audio_np)]))

        if self._left_context is None:
            self._left_context = torch.zeros(1, mel.shape[1], _LEFT_CONTEXT_FRAMES)

        signal = torch.cat([self._left_context, mel], dim=-1)
        signal_len = mel_len + _LEFT_CONTEXT_FRAMES

        self._streaming_state, self._total_preds = self._model.forward_streaming_step(
            signal,
            signal_len,
            self._streaming_state,
            self._total_preds,
            left_offset=0,
            right_offset=0,
        )

        self._left_context = mel[:, :, -_LEFT_CONTEXT_FRAMES:]
        self._total_audio_bytes += len(pcm)
