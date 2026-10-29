"""Opt-in real-model check that onnx-canary-split decodes concurrently.

The split-decode ``_decode`` override caches the 16 cross-attention K/V
tensors per thread (``threading.local()``): each ``transcribe_pcm`` call runs
its whole ``recognize_batch`` on one ``asyncio.to_thread`` worker, so two
overlapping windows on one shared adapter instance must each recompute and
read back their own K/V tensors. This test exercises that path with the real
ONNX graphs (no mocks anywhere on the K/V path): two *different* windows are
transcribed sequentially, then concurrently through the same adapter, and each
concurrent result must equal its sequential result.

Opt-in because it loads the default model, ``collectiveai/canary-1b-v2-onnx-split-int8``
from the Hugging Face Hub (about 1.29 GB on the first run, seconds of load
time) -- the ``real_model`` marker, same as ``test_asr_determinism.py``.

To run:

    CORO_RUN_REAL_MODEL_TESTS=1 uv run --extra cpu --group bench pytest \
        tests/backends/asr/test_onnx_canary_split_concurrency.py -v
"""

from __future__ import annotations

import asyncio
import wave
from dataclasses import asdict
from pathlib import Path

import pytest

_MODEL = "collectiveai/canary-1b-v2-onnx-split-int8"
_AUDIO = Path(__file__).resolve().parents[3] / "coro" / "bench" / "data" / "jfk.wav"

_WINDOW_SECONDS = 2.5
_SAMPLE_RATE = 16000
# Two different speech windows of the 11 s clip that never share audio.
_WINDOW_STARTS = (1.0, 6.5)


def _window_pcm(start_seconds: float) -> bytes:
    """Carve one s16le mono 16 kHz window from the wav (no ffmpeg needed)."""
    frames = int(_WINDOW_SECONDS * _SAMPLE_RATE)
    with wave.open(str(_AUDIO)) as wav:
        wav.setpos(int(start_seconds * _SAMPLE_RATE))
        return wav.readframes(frames)


@pytest.fixture(scope="module")
def adapter():
    """Build the real adapter once: auto-sized admission (>= 2 permits)."""
    from coro.backends.asr.onnx_canary_split import build_onnx_canary_split_adapter

    built = build_onnx_canary_split_adapter(
        _MODEL,
        device="cpu",
        quantization="static_qdq_v4_pct_excl",
        decoder_quantization="dynamic_v1_quint8",
    )
    assert built.admission.max_concurrency >= 2
    return built


@pytest.mark.real_model
async def test_concurrent_windows_match_sequential_results(adapter):
    """Two overlapping decodes on one shared instance each keep their own K/V."""
    windows = [_window_pcm(start) for start in _WINDOW_STARTS]

    sequential = [
        [asdict(t) for t in await adapter.transcribe_pcm(window, language="en")]
        for window in windows
    ]
    concurrent_pair, concurrent_solo = await asyncio.gather(
        adapter.transcribe_pcm(windows[0], language="en"),
        adapter.transcribe_pcm(windows[1], language="en"),
    )

    assert [asdict(t) for t in concurrent_pair] == sequential[0]
    assert [asdict(t) for t in concurrent_solo] == sequential[1]
