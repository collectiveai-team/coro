"""Streaming Sortformer diarizer."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import torch

from coro.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
from coro.backends.diarization.nemo.latency import (
    LATENCY_TIER_PARAMS,
    LatencyTierParams,
    applied_streaming_params,
    bind_latency_tier,
    get_latency_tier_params,
)
from coro.backends.diarization.nemo.postprocessing import (
    DEFAULT_MAX_SPEAKERS,
    apply_gated_postprocessing,
)
from coro.backends.diarization.segments import convert_diarization_segments
from coro.core.logger import get_logger
from coro.core.models import SpeakerSegment

if TYPE_CHECKING:
    from coro.backends.diarization.nemo.sortformer import (
        MelPreprocessor,
        PostProcessor,
        StreamingSortformerModel,
    )

# The latency-tier names stay importable from here for existing callers.
__all__ = [
    "LATENCY_TIER_PARAMS",
    "LatencyTierParams",
    "NemoStreamingDiarizerFactory",
    "StreamingDiarizer",
    "applied_streaming_params",
    "bind_latency_tier",
    "get_latency_tier_params",
]

logger = get_logger(__name__)


class NemoStreamingDiarizerFactory:
    """Produces fresh per-request StreamingDiarizer instances bound to a shared NeMo model.

    Conforms to the core ``StreamingDiarizerFactory`` protocol; named distinctly so
    the concrete implementation does not collide with that protocol.
    """

    def __init__(
        self,
        model: StreamingSortformerModel,
        *,
        tier: str = "very-high",
        postprocessing_yaml: str | None = None,
        max_speakers: int = DEFAULT_MAX_SPEAKERS,
        preprocessor: MelPreprocessor | None = None,
    ) -> None:
        """Bind ``model`` to ``tier``.

        ``preprocessor`` is shared by every diarizer of the factory; None gives
        each diarizer NeMo's own mel preprocessor, built on first use.
        """
        self._tier = tier
        self._tier_params = get_latency_tier_params(tier)
        # Every request of this factory shares one tier-bound view; building it
        # leaves the shared model untouched, and so does every later call.
        self._model = bind_latency_tier(model, self._tier_params)
        self._model.sortformer_modules._check_streaming_parameters()
        self._postprocessing_yaml = postprocessing_yaml
        self._max_speakers = max_speakers
        self._preprocessor = preprocessor
        self._subsampling_factor = getattr(model.sortformer_modules, "subsampling_factor", 8)
        self._n_spk = getattr(model.sortformer_modules, "n_spk", 4)

    @property
    def tier(self) -> str:
        """The Diarization Latency tier every diarizer of this factory runs at."""
        return self._tier

    @property
    def postprocessing_yaml(self) -> str | None:
        """The resolved Diarization Post-Processing Configuration path, or None."""
        return self._postprocessing_yaml

    @property
    def max_speakers(self) -> int:
        """The Speaker-Count Post-Processing Gate ceiling."""
        return self._max_speakers

    def __call__(self) -> StreamingDiarizer:
        return StreamingDiarizer(
            self._model,
            chunk_len=self._tier_params.chunk_len,
            chunk_right_context=self._tier_params.chunk_right_context,
            subsampling_factor=self._subsampling_factor,
            n_spk=self._n_spk,
            preprocessor=self._preprocessor,
            postprocessing_yaml=self._postprocessing_yaml,
            max_speakers=self._max_speakers,
        )


class StreamingDiarizer:
    """Encapsulates streaming Sortformer diarization behind a two-method interface."""

    def __init__(
        self,
        model: StreamingSortformerModel,
        *,
        chunk_len: int = 6,
        chunk_right_context: int = 1,
        subsampling_factor: int = 8,
        n_spk: int = 4,
        preprocessor: MelPreprocessor | None = None,
        post_processor: PostProcessor | None = None,
        postprocessing_yaml: str | None = None,
        max_speakers: int = DEFAULT_MAX_SPEAKERS,
    ):
        self._model = model
        self._device = model.device
        self._chunk_len = chunk_len
        self._chunk_right_context = chunk_right_context
        self._subsampling_factor = subsampling_factor
        self._n_spk = n_spk
        self._preprocessor = preprocessor
        self._post_processor = post_processor
        self._postprocessing_yaml = postprocessing_yaml
        self._max_speakers = max_speakers

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
        """Run per-speaker VAD post-processing matching the NeMo model's own approach.

        Delegates to the shared Diarization Post-Processing Configuration
        helper so the Streaming Pipeline and the Full-Memory Pipeline cannot
        drift apart in how identical predictions become segments, and so both
        honour the same Speaker-Count Post-Processing Gate. See ADR 0010.
        """
        started = time.perf_counter()

        preds = total_preds if total_preds is not None else self._combined_preds()
        raw_segments = apply_gated_postprocessing(
            preds,
            n_spk=self._n_spk,
            postprocessing_yaml=self._postprocessing_yaml,
            subsampling_factor=self._subsampling_factor,
            max_speakers=self._max_speakers,
        )

        segments = convert_diarization_segments(raw_segments, duration=duration)
        logger.info(
            "streaming_diarizer post_process complete elapsed=%.3fs raw_segments=%d segments=%d",
            time.perf_counter() - started,
            len(raw_segments),
            len(segments),
        )
        return segments

    def _get_preprocessor(self) -> MelPreprocessor:
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
            # When built by NemoStreamingDiarizerFactory, ``self._model`` is a
            # tier-bound view: NeMo reads the tier off it, and nothing here
            # writes to the model shared with other requests.
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
