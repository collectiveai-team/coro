"""Real-model integration smoke test for StreamingPipeline.

This test is **opt-in** and will be skipped unless the environment variable
``CORO_RUN_REAL_MODEL_TESTS=1`` is set.

To run:

    CORO_RUN_REAL_MODEL_TESTS=1 .venv/bin/pytest tests/test_streaming_pipeline_real_model.py -v

Requirements:
- A machine with the model ``nvidia/diar_streaming_sortformer_4spk-v2`` cached
  (or network access to download it on first run).
- The NeMo and faster-whisper packages installed in the environment.

The test loads real models and exercises the end-to-end streaming pipeline
against the warmup audio asset. It catches regressions that mocked-model tests
cannot (feature-shape mismatches, NeMo API drift, post-processing errors).
"""

from __future__ import annotations

import os
import struct
from unittest.mock import patch

import pytest

REAL_MODEL_TESTS = os.environ.get("CORO_RUN_REAL_MODEL_TESTS", "0") == "1"
skip_unless_real = pytest.mark.skipif(
    not REAL_MODEL_TESTS,
    reason=(
        "Skipping real-model test. "
        "Set CORO_RUN_REAL_MODEL_TESTS=1 to run with real model checkpoints."
    ),
)


# ---------------------------------------------------------------------------
# Opt-in real-model test
# ---------------------------------------------------------------------------


@skip_unless_real
@pytest.mark.asyncio
async def test_streaming_pipeline_real_model_transcribes_warmup_audio():
    """End-to-end smoke test: real ASR + real NeMo diarizer + warmup audio."""
    from nemo.collections.asr.models import SortformerEncLabelModel

    from coro.audio import AudioInput
    from coro.backends.faster_whisper import build_asr_adapter
    from coro.backends.nemo_streaming import StreamingDiarizerFactory
    from coro.pipelines.streaming import StreamingPipeline
    from coro.bench.data import WARMUP_AUDIO_PATH

    # Load real models
    asr = build_asr_adapter("openai/whisper-small", device="cpu")
    diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2")
    diar_model.eval()
    factory = StreamingDiarizerFactory(diar_model, tier="very-high")

    pipeline = StreamingPipeline(
        asr=asr,
        streaming_diarizer_factory=factory,
    )

    audio = AudioInput(WARMUP_AUDIO_PATH.read_bytes())
    result = await pipeline.transcribe(audio, language="es")

    EXPECTED_KEYS = {"segments", "word_segments", "transcript", "diarization", "raw_words"}
    assert set(result.keys()) >= EXPECTED_KEYS
    assert len(result["transcript"]) > 0, "Expected non-empty transcript"
    assert len(result["segments"]) > 0, "Expected at least one segment"


@skip_unless_real
def test_streaming_diarizer_frame_count_matches_audio_duration():
    """Streaming diarizer output frames must align with the audio timeline.

    Regression test for the per-chunk-mel drift bug: computing the mel
    spectrogram independently per PCM chunk introduced a boundary edge frame
    (~+1 output frame per chunk) plus a zero-padded finalize chunk, inflating
    the prediction frame count.  Over a long recording this accumulated into
    several seconds of temporal drift and wrecked DER on later segments.

    The total number of predicted frames must equal the audio duration in
    subsampled frames (sample_rate / window_stride / subsampling_factor), with
    at most one frame of rounding slack.
    """
    import numpy as np
    from nemo.collections.asr.models import SortformerEncLabelModel

    from coro.backends.nemo_streaming import StreamingDiarizerFactory

    SAMPLE_RATE = 16000
    SUBSAMPLING = 8
    MEL_STRIDE_S = 0.01  # window_stride
    # Use a multi-chunk duration so cross-chunk drift would be detectable: a
    # very-high tier chunk is ~27.2s, so 95s exercises 3 full chunks + remainder.
    DURATION_S = 95

    model = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2")
    model.eval()
    factory = StreamingDiarizerFactory(model, tier="very-high")
    diar = factory()

    rng = np.random.default_rng(0)
    pcm = (rng.standard_normal(SAMPLE_RATE * DURATION_S) * 0.1 * 32768).astype(np.int16).tobytes()

    # Feed 1-second PCM chunks like the streaming pipeline does.
    one_second = SAMPLE_RATE * 2
    for i in range(0, len(pcm), one_second):
        diar.ingest_pcm_chunk(pcm[i : i + one_second])
    diar.finalize()

    total_frames = diar._combined_preds().shape[1]
    expected_frames = int(DURATION_S / MEL_STRIDE_S / SUBSAMPLING)
    assert abs(total_frames - expected_frames) <= 1, (
        f"Frame drift detected: got {total_frames} prediction frames for "
        f"{DURATION_S}s audio, expected ~{expected_frames}. The per-chunk mel "
        f"trim / no-pad finalize fix prevents accumulating temporal drift."
    )


# ---------------------------------------------------------------------------
# Warmup path with mocked-model (always runs)
# ---------------------------------------------------------------------------


class _FakeASRAdapter:
    async def transcribe_pcm(self, pcm_bytes, *, language=None, prompt=None):
        return []


_FAKE_PCM = struct.pack("<1600h", *([0] * 1600))


async def _fake_stream(path: str, chunk_seconds: float = 1.0):
    yield _FAKE_PCM


@pytest.mark.asyncio
async def test_streaming_pipeline_warmup_mocked():
    """StreamingPipeline.transcribe() can be used as warmup without errors."""
    from coro.audio import AudioInput
    from coro.pipelines.streaming import StreamingPipeline

    pipeline = StreamingPipeline(asr=_FakeASRAdapter())
    audio = AudioInput(_FAKE_PCM)

    with patch(
        "coro.pipelines.streaming.stream_pcm_from_file",
        new=_fake_stream,
    ):
        result = await pipeline.transcribe(audio)

    EXPECTED_KEYS = {"segments", "word_segments", "transcript", "diarization", "raw_words"}
    assert set(result.keys()) >= EXPECTED_KEYS
    assert result["segments"] == []
