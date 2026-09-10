"""Opt-in real-model check that onnx-canary-split decodes concurrently.

The split-decode ``_decode`` override caches the 16 cross-attention K/V
tensors per thread (``threading.local()``): each ``transcribe_pcm`` call runs
its whole ``recognize_batch`` on one ``asyncio.to_thread`` worker, so two
overlapping windows on one shared adapter instance must each recompute and
read back their own K/V tensors. This test exercises that path with the real
ONNX graphs (no mocks anywhere on the K/V path): two *different* windows are
transcribed sequentially, then concurrently through the same adapter, and each
concurrent result must equal its sequential result.

Opt-in because it loads the full model from local artifacts (several hundred
MB, seconds of load time) -- same convention as
``test_asr_determinism.py``. Skips unless both the env var is set and the
artifacts/audio exist.

To run:

    CORO_RUN_REAL_MODEL_TESTS=1 uv run --extra cpu --group bench pytest \
        tests/backends/asr/test_onnx_canary_split_concurrency.py -v
"""

from __future__ import annotations

import asyncio
import os
import wave
from dataclasses import asdict
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ARTIFACTS = _REPO_ROOT / ".tmp" / "onnx_export" / "canary_split"
_AUDIO = (
    _REPO_ROOT
    / ".tmp"
    / "bench-runs"
    / "spanish-corpora"
    / "mtedx-longform"
    / "mtedx-HLIJkmy3vy8.wav"
)

REAL_MODEL_TESTS = os.environ.get("CORO_RUN_REAL_MODEL_TESTS", "0") == "1"
_ARTIFACTS_PRESENT = (
    (_ARTIFACTS / "encoder-model.static_qdq_v4_pct_excl.onnx").is_file()
    and (_ARTIFACTS / "decoder_step.dynamic_v1_quint8.onnx").is_file()
    and _AUDIO.is_file()
)
skip_unless_real = pytest.mark.skipif(
    not (REAL_MODEL_TESTS and _ARTIFACTS_PRESENT),
    reason=(
        "Skipping real-model concurrency check. Set CORO_RUN_REAL_MODEL_TESTS=1 "
        "with the .tmp/onnx_export/canary_split artifacts and the mtedx wav "
        "present to run it."
    ),
)

_WINDOW_SECONDS = 2.5
_SAMPLE_RATE = 16000
# Two different speech windows, far enough apart that they never share audio.
_WINDOW_STARTS = (60.0, 120.0)


def _window_pcm(start_seconds: float) -> bytes:
    """Carve one s16le mono 16 kHz window from the mtedx wav (no ffmpeg needed)."""
    frames = int(_WINDOW_SECONDS * _SAMPLE_RATE)
    with wave.open(str(_AUDIO)) as wav:
        wav.setpos(int(start_seconds * _SAMPLE_RATE))
        return wav.readframes(frames)


@pytest.fixture(scope="module")
def adapter():
    """Build the real adapter once: auto-sized admission (>= 2 permits)."""
    from coro.backends.asr.onnx_canary_split import build_onnx_canary_split_adapter

    built = build_onnx_canary_split_adapter(
        str(_ARTIFACTS),
        device="cpu",
        quantization="static_qdq_v4_pct_excl",
        decoder_quantization="dynamic_v1_quint8",
    )
    assert built.admission.max_concurrency >= 2
    return built


@skip_unless_real
async def test_concurrent_windows_match_sequential_results(adapter):
    """Two overlapping decodes on one shared instance each keep their own K/V."""
    windows = [_window_pcm(start) for start in _WINDOW_STARTS]

    sequential = [
        [asdict(t) for t in await adapter.transcribe_pcm(window, language="es")]
        for window in windows
    ]
    concurrent_pair, concurrent_solo = await asyncio.gather(
        adapter.transcribe_pcm(windows[0], language="es"),
        adapter.transcribe_pcm(windows[1], language="es"),
    )

    assert [asdict(t) for t in concurrent_pair] == sequential[0]
    assert [asdict(t) for t in concurrent_solo] == sequential[1]
