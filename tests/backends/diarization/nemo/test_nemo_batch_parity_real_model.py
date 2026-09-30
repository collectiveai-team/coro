"""Real-model parity: direct ``forward`` must match NeMo's ``diarize()`` (ADR 0021).

Opt-in, skipped unless ``CORO_RUN_REAL_MODEL_TESTS=1``. Needs
``nvidia/diar_streaming_sortformer_4spk-v2`` cached (or network access).

The batch adapter bypasses ``diarize()`` because its per-call state on the
shared model races under concurrency (issue #81). That is only acceptable if
the numbers do not move, so this compares the raw activity matrix and the
final segments against ``diarize()`` on the same audio, and checks that
concurrent calls on one model agree with a sequential one. Run it on every
NeMo upgrade.
"""

from __future__ import annotations

import asyncio
import os
import wave

import pytest

REAL_MODEL_TESTS = os.environ.get("CORO_RUN_REAL_MODEL_TESTS", "0") == "1"
pytestmark = pytest.mark.skipif(
    not REAL_MODEL_TESTS,
    reason="Set CORO_RUN_REAL_MODEL_TESTS=1 to run with real model checkpoints.",
)

MODEL = "nvidia/diar_streaming_sortformer_4spk-v2"


@pytest.fixture(scope="module")
def adapter():
    from coro.backends.diarization.nemo.diarization import build_nemo_diarization_adapter

    return build_nemo_diarization_adapter(MODEL, device="cpu")


@pytest.fixture(scope="module")
def pcm():
    from coro.audio import convert_path_to_pcm_bytes
    from coro.bench.data import WARMUP_AUDIO_PATH

    # The 11 s clip tiled to ~99 s, so streaming Sortformer revisions run
    # several chunks and exercise the speaker cache / FIFO, not just one chunk.
    return asyncio.run(convert_path_to_pcm_bytes(str(WARMUP_AUDIO_PATH))) * 9


def _diarize_reference(model, pcm: bytes, tmp_path):
    """What the adapter used to do: temp WAV through NeMo's ``diarize()``."""
    from coro.audio import BYTES_PER_SAMPLE, SAMPLE_RATE

    path = tmp_path / "reference.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(BYTES_PER_SAMPLE)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    lines, preds_list = model.diarize(audio=str(path), batch_size=1, include_tensor_outputs=True)
    if len(lines) == 1 and isinstance(lines[0], list):
        lines = lines[0]
    return lines, preds_list[0]


def test_forward_matches_diarize(adapter, pcm, tmp_path):
    """Same activity matrix and same segments (within NeMo's 2-decimal rounding)."""
    from coro.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
    from coro.backends.diarization.segments import convert_diarization_segments

    direct_preds = adapter._predict(pcm)
    direct = asyncio.run(adapter.diarize_pcm(pcm))
    ref_lines, ref_preds = _diarize_reference(adapter.model, pcm, tmp_path)

    assert direct_preds.shape == ref_preds.shape
    assert (direct_preds - ref_preds.cpu()).abs().max().item() <= 1e-5

    duration = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
    reference = convert_diarization_segments(ref_lines, duration=duration)
    assert [s.speaker for s in direct] == [s.speaker for s in reference]
    # max() over an empty pairing raises, so "no segments on either side" fails too.
    boundary_drift = max(
        abs(a - b)
        for got, want in zip(direct, reference, strict=True)
        for a, b in ((got.start, want.start), (got.end, want.end))
    )
    assert boundary_drift <= 0.006  # NeMo rounds its RTTM lines to 2 decimals


def test_concurrent_calls_on_one_model_match_a_sequential_call(adapter, pcm):
    """The issue #81 scenario on the real model: overlap must not change the result."""

    async def _run():
        sequential = await adapter.diarize_pcm(pcm)
        concurrent = await asyncio.gather(*(adapter.diarize_pcm(pcm) for _ in range(3)))
        return sequential, concurrent

    sequential, concurrent = asyncio.run(_run())

    # The tiled warmup clip is one speaker talking for ~78 s of the ~99 s.
    assert {s.speaker for s in sequential} == {1}
    assert sum(s.end - s.start for s in sequential) > 60.0
    assert concurrent == [sequential] * 3
