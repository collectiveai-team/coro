"""Streaming Sortformer diarizer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
import time

import numpy as np
import torch

from coro.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
from coro.backends.nemo import convert_diarization_segments
from coro.core.models import SpeakerSegment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LatencyTierParams:
    """Immutable Sortformer streaming parameters for one latency tier."""

    chunk_len: int
    chunk_right_context: int
    fifo_len: int
    spkcache_update_period: int
    spkcache_len: int


LATENCY_TIER_PARAMS: dict[str, LatencyTierParams] = {
    "very-high": LatencyTierParams(
        chunk_len=340,
        chunk_right_context=40,
        fifo_len=40,
        spkcache_update_period=300,
        spkcache_len=188,
    ),
    "high": LatencyTierParams(
        chunk_len=124,
        chunk_right_context=1,
        fifo_len=124,
        spkcache_update_period=124,
        spkcache_len=188,
    ),
    "low": LatencyTierParams(
        chunk_len=6,
        chunk_right_context=7,
        fifo_len=188,
        spkcache_update_period=144,
        spkcache_len=188,
    ),
    "ultra-low": LatencyTierParams(
        chunk_len=3,
        chunk_right_context=1,
        fifo_len=188,
        spkcache_update_period=144,
        spkcache_len=188,
    ),
}


def get_latency_tier_params(tier: str) -> LatencyTierParams:
    """Return the immutable streaming parameters for a latency tier."""
    return LATENCY_TIER_PARAMS[tier]


class StreamingDiarizerFactory:
    """Produces fresh per-request StreamingDiarizer instances bound to a shared model."""

    def __init__(self, model, *, tier: str = "very-high") -> None:
        self._model = model
        self._tier = tier
        self._tier_params = get_latency_tier_params(tier)
        subsampling_factor = getattr(model.sortformer_modules, "subsampling_factor", 8)
        n_spk = getattr(model.sortformer_modules, "n_spk", 4)
        model.sortformer_modules.chunk_len = self._tier_params.chunk_len
        model.sortformer_modules.chunk_right_context = self._tier_params.chunk_right_context
        model.sortformer_modules.fifo_len = self._tier_params.fifo_len
        model.sortformer_modules.spkcache_update_period = self._tier_params.spkcache_update_period
        model.sortformer_modules.spkcache_len = self._tier_params.spkcache_len
        model.sortformer_modules._check_streaming_parameters()
        self._subsampling_factor = subsampling_factor
        self._n_spk = n_spk

    def __call__(self) -> StreamingDiarizer:
        return StreamingDiarizer(
            self._model,
            chunk_len=self._tier_params.chunk_len,
            chunk_right_context=self._tier_params.chunk_right_context,
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
        chunk_right_context: int = 1,
        subsampling_factor: int = 8,
        n_spk: int = 4,
        preprocessor=None,
        post_processor: Callable | None = None,
    ):
        self._model = model
        self._device = model.device
        self._chunk_len = chunk_len
        self._chunk_right_context = chunk_right_context
        self._subsampling_factor = subsampling_factor
        self._n_spk = n_spk
        self._preprocessor = preprocessor
        self._post_processor = post_processor

        chunk_audio_seconds = chunk_len * subsampling_factor * 0.01
        self._chunk_audio_bytes = int(chunk_audio_seconds * SAMPLE_RATE * BYTES_PER_SAMPLE)

        # Right-context PCM: extra audio beyond each chunk boundary so the
        # non-causal transformer has future context for frames near the chunk
        # end.  NeMo's streaming_feat_loader includes
        # chunk_right_context * subsampling_factor mel frames of right context
        # and sets right_offset so the model trims those frames from output.
        right_context_seconds = chunk_right_context * subsampling_factor * 0.01
        self._right_context_bytes = int(right_context_seconds * SAMPLE_RATE * BYTES_PER_SAMPLE)
        self._model_right_context_frames = chunk_right_context * subsampling_factor

        self._pcm_buffer = b""
        self._streaming_state = model.sortformer_modules.init_streaming_state(
            batch_size=1,
            async_streaming=False,
            device=self._device,
        )
        # Initialize as empty accumulator matching what forward_streaming_step expects;
        # NeMo uses torch.zeros((batch, 0, n_spk)) as the seed before the first chunk.
        self._total_preds: torch.Tensor = torch.zeros((1, 0, self._n_spk), device=self._device)
        self._pred_chunks: list[torch.Tensor] = []
        self._total_audio_bytes = 0
        self._processed_chunks = 0

    @property
    def processed_chunks(self) -> int:
        return self._processed_chunks

    def ingest_pcm_chunk(self, pcm: bytes) -> None:
        self._pcm_buffer += pcm
        min_chunk = self._chunk_audio_bytes + self._right_context_bytes
        while len(self._pcm_buffer) >= min_chunk:
            chunk_pcm = self._pcm_buffer[:min_chunk]
            self._pcm_buffer = self._pcm_buffer[self._chunk_audio_bytes :]
            self._total_audio_bytes += self._chunk_audio_bytes
            self._process_chunk(chunk_pcm, right_offset=self._model_right_context_frames)

    def finalize(self) -> list[SpeakerSegment]:
        if self._pcm_buffer:
            remainder_len = len(self._pcm_buffer)
            logger.info(
                "streaming_diarizer finalize flush_remainder bytes=%d processed_chunks=%d",
                remainder_len,
                self._processed_chunks,
            )
            # Process the real remainder without zero-padding.  Padding to a full
            # chunk produces spurious prediction frames for the silent tail that
            # inflate the output frame count and misalign the timeline; the mel
            # trim in _process_chunk (right_offset=0 branch) instead floors the
            # remainder to a whole number of output frames.
            remainder = self._pcm_buffer
            self._pcm_buffer = b""
            self._process_chunk(remainder, right_offset=0)
            self._total_audio_bytes += remainder_len

        total_preds = self._combined_preds()
        if total_preds.shape[1] == 0:
            logger.info(
                "streaming_diarizer finalize no_predictions processed_chunks=%d",
                self._processed_chunks,
            )
            return []

        duration = self._total_audio_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        logger.info(
            "streaming_diarizer finalize predictions_shape=%s duration=%.2fs processed_chunks=%d",
            tuple(total_preds.shape),
            duration,
            self._processed_chunks,
        )

        if self._post_processor is not None:
            raw_segments = self._post_processor(total_preds, self._n_spk)
            return convert_diarization_segments(raw_segments, duration=duration)

        return self._default_post_process(duration, total_preds=total_preds)

    def _combined_preds(self) -> torch.Tensor:
        if self._pred_chunks:
            return torch.cat(self._pred_chunks, dim=1)
        return self._total_preds.cpu()

    def _default_post_process(
        self,
        duration: float,
        *,
        total_preds: torch.Tensor | None = None,
    ) -> list[SpeakerSegment]:
        """Run per-speaker VAD post-processing matching the NeMo model's own approach."""
        started = time.perf_counter()
        from nemo.collections.asr.models.sortformer_diar_models import ts_vad_post_processing
        from nemo.collections.asr.parts.mixins.diarization import load_postprocessing_from_yaml

        # NeMo accepts None to load default post-processing params; stub types str.
        cfg_vad_params = load_postprocessing_from_yaml(None)  # pyrefly: ignore[bad-argument-type]
        # total_preds: (1, n_frames, n_spk) — process each speaker independently
        preds_cpu = (
            (total_preds if total_preds is not None else self._combined_preds()).squeeze(0).cpu()
        )
        subsampling_factor = self._subsampling_factor

        raw_segments: list[tuple[float, float, int]] = []
        for spk_id in range(self._n_spk):
            spk_preds = preds_cpu[:, spk_id]  # (n_frames,)
            ts_mat = ts_vad_post_processing(
                spk_preds,
                # NeMo consumes the PostProcessingParams dataclass; stub types OmegaConf.
                cfg_vad_params=cfg_vad_params,  # pyrefly: ignore[bad-argument-type]
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
            window_size=0.025,
            normalize="NA",
            n_fft=512,
            features=128,
            pad_to=0,
        ).to(self._device)
        return self._preprocessor

    def _process_chunk(self, pcm: bytes, right_offset: int = 0) -> None:
        """Run one chunk through the streaming diarizer.

        The NeMo streaming_state (spkcache + fifo) carries all historical
        left-context internally.  When ``right_offset > 0`` the PCM includes
        extra future audio so the non-causal transformer has right-context for
        frames near the chunk end; ``forward_streaming_step`` trims those
        extra prediction frames via ``right_offset`` (in mel-frame units).
        forward_streaming_step expects (batch, time, features), i.e. time-first,
        which is the opposite of the preprocessor's (batch, features, time) output.
        """
        with torch.inference_mode():
            audio_np = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).to(self._device)

            preprocessor = self._get_preprocessor()
            mel, mel_len = preprocessor(
                input_signal=audio_tensor,
                length=torch.tensor([len(audio_np)], device=self._device),
            )

            # Transpose from (batch, features, time) → (batch, time, features)
            signal_t = mel.transpose(1, 2)

            # Trim mel to an exact target frame count.  Computing the mel
            # spectrogram per-PCM-chunk introduces a boundary edge frame that the
            # batch path (which computes one mel over the full audio) never
            # produces.  Left unchecked this adds ~1 output frame per chunk,
            # accumulating temporal drift across a long recording and degrading
            # DER on later segments.  We trim to a multiple of the subsampling
            # factor so each chunk emits exactly chunk_len prediction frames.
            mel_frames = signal_t.shape[1]
            if right_offset > 0:
                target_frames = self._chunk_len * self._subsampling_factor + right_offset
            else:
                target_frames = (mel_frames // self._subsampling_factor) * self._subsampling_factor
            target_frames = min(target_frames, mel_frames)
            if target_frames < self._subsampling_factor:
                # Too little audio to yield even one output frame; skip.
                return
            signal_t = signal_t[:, :target_frames, :]
            mel_len = torch.tensor([target_frames], device=self._device)

            seed_preds = torch.zeros((1, 0, self._n_spk), device=self._device)
            self._streaming_state, chunk_preds = self._model.forward_streaming_step(
                signal_t,
                mel_len,
                self._streaming_state,
                seed_preds,
                left_offset=0,
                right_offset=right_offset,
            )
        self._pred_chunks.append(chunk_preds.detach().cpu())
        self._total_preds = seed_preds

        self._processed_chunks += 1
        if self._processed_chunks == 1 or self._processed_chunks % 5 == 0:
            logger.info(
                "streaming_diarizer chunk=%d pcm_bytes=%d right_offset=%d mel_shape=%s "
                "chunk_preds_shape=%s stored_pred_chunks=%d",
                self._processed_chunks,
                len(pcm),
                right_offset,
                tuple(mel.shape),
                tuple(chunk_preds.shape),
                len(self._pred_chunks),
            )
