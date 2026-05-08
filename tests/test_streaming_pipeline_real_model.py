"""Real-model integration smoke test for StreamingPipeline.

This test is **opt-in** and will be skipped unless the environment variable
``ASR_DIAR_RUN_REAL_MODEL_TESTS=1`` is set.

To run:

    ASR_DIAR_RUN_REAL_MODEL_TESTS=1 .venv/bin/pytest tests/test_streaming_pipeline_real_model.py -v

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

REAL_MODEL_TESTS = os.environ.get("ASR_DIAR_RUN_REAL_MODEL_TESTS", "0") == "1"
skip_unless_real = pytest.mark.skipif(
    not REAL_MODEL_TESTS,
    reason=(
        "Skipping real-model test. "
        "Set ASR_DIAR_RUN_REAL_MODEL_TESTS=1 to run with real model checkpoints."
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

    from asr_diar_server.audio import AudioInput
    from asr_diar_server.backends.faster_whisper import FasterWhisperASRAdapter
    from asr_diar_server.backends.nemo_streaming import StreamingDiarizerFactory
    from asr_diar_server.pipelines.streaming import StreamingPipeline
    from asr_diar_server.app import WARMUP_AUDIO_PATH

    # Load real models
    asr = FasterWhisperASRAdapter(model_name="openai/whisper-small", device="cpu")
    diar_model = SortformerEncLabelModel.from_pretrained(
        "nvidia/diar_streaming_sortformer_4spk-v2"
    )
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
    from asr_diar_server.audio import AudioInput
    from asr_diar_server.pipelines.streaming import StreamingPipeline

    pipeline = StreamingPipeline(asr=_FakeASRAdapter(), diarization=None)
    audio = AudioInput(_FAKE_PCM)

    with patch(
        "asr_diar_server.pipelines.streaming.stream_pcm_from_file",
        new=_fake_stream,
    ):
        result = await pipeline.transcribe(audio)

    EXPECTED_KEYS = {"segments", "word_segments", "transcript", "diarization", "raw_words"}
    assert set(result.keys()) >= EXPECTED_KEYS
    assert result["segments"] == []
