"""The streaming diarizer factory must not retune the batch Diarization Adapter.

Both Diarization Flows share one Sortformer model object, and NeMo reads the
streaming parameters (``chunk_len``, ``chunk_right_context``, ``fifo_len``,
``spkcache_update_period``, ``spkcache_len``) off ``model.sortformer_modules``
at call time — the batch flow included, because the streaming Sortformer
revisions set ``streaming_mode=True`` and therefore route batch ``diarize()``
through ``forward_streaming``.

Writing the streaming latency tier onto that shared object permanently, as
construction used to, silently changed what batch diarization did. Any
batch-vs-streaming comparison in one process was invalid. These tests pin the
fix. See ADR 0009.
"""

from __future__ import annotations

import struct
from unittest.mock import MagicMock

import pytest
import torch

from coro.backends.diarization.nemo.streaming import (
    LATENCY_TIER_PARAMS,
    NemoStreamingDiarizerFactory,
    applied_streaming_params,
    get_latency_tier_params,
)

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2

# The five attributes the latency tier overwrites, and therefore the five the
# batch Diarization Adapter can be silently retuned through.
SHARED_STREAMING_ATTRS = (
    "chunk_len",
    "chunk_right_context",
    "fifo_len",
    "spkcache_update_period",
    "spkcache_len",
)

# Deliberately unlike any latency tier, so an accidental write is visible.
BATCH_CONFIG = {
    "chunk_len": 188,
    "chunk_right_context": 1,
    "fifo_len": 0,
    "spkcache_update_period": 188,
    "spkcache_len": 188,
}


class _FakeSortformerModules:
    """Plain object (not a MagicMock) so attribute writes are really observable."""

    def __init__(self):
        for name, value in BATCH_CONFIG.items():
            setattr(self, name, value)
        self.subsampling_factor = 8
        self.n_spk = 4
        self.fc_d_model = 512
        self.checked_with: list[dict] = []

    def _check_streaming_parameters(self):
        self.checked_with.append(_snapshot(self))

    def init_streaming_state(self, *, batch_size, async_streaming, device):
        state = {"step": 0}
        return state


def _snapshot(modules) -> tuple[tuple[str, int], ...]:
    """The shared streaming config, as comparable (name, value) pairs."""
    return tuple((name, getattr(modules, name)) for name in SHARED_STREAMING_ATTRS)


def _pairs(config) -> tuple[tuple[str, int], ...]:
    return tuple((name, config[name]) for name in SHARED_STREAMING_ATTRS)


def _make_model():
    model = MagicMock()
    model.device = torch.device("cpu")
    model.sortformer_modules = _FakeSortformerModules()

    def _forward_streaming_step(
        processed_signal, processed_signal_length, streaming_state, total_preds, **kwargs
    ):
        # Record what NeMo would actually read during the call.
        model.params_seen_during_call.append(_snapshot(model.sortformer_modules))
        chunk_preds = torch.rand(1, 4, 4) * 0.01
        return {"step": 1}, torch.cat([total_preds, chunk_preds], dim=1)

    model.params_seen_during_call = []
    model.forward_streaming_step = MagicMock(side_effect=_forward_streaming_step)
    return model


def _make_preprocessor():
    def _process(*, input_signal, length):
        n_mel_frames = max(1, input_signal.shape[-1] // 160)
        return torch.randn(1, 128, n_mel_frames), torch.tensor([n_mel_frames])

    return MagicMock(side_effect=_process)


def _pcm(n_bytes: int) -> bytes:
    n_samples = n_bytes // BYTES_PER_SAMPLE
    return struct.pack(f"<{n_samples}h", *([1000] * n_samples))


# ---------------------------------------------------------------------------
# AC5: construction leaves the shared model untouched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tier", sorted(LATENCY_TIER_PARAMS))
def test_building_streaming_factory_leaves_batch_config_unchanged(tier):
    """Constructing the streaming factory must not retune the batch adapter."""
    model = _make_model()
    before = _snapshot(model.sortformer_modules)

    NemoStreamingDiarizerFactory(model, tier=tier)

    assert _snapshot(model.sortformer_modules) == before == _pairs(BATCH_CONFIG)


def test_building_streaming_factory_still_validates_the_tier():
    """Validation must still happen — with the tier applied, then rolled back."""
    model = _make_model()

    NemoStreamingDiarizerFactory(model, tier="low")

    tier_params = get_latency_tier_params("low")
    assert model.sortformer_modules.checked_with == [_pairs(vars(tier_params))]
    # ...and the check left no residue on the shared object.
    assert _snapshot(model.sortformer_modules) == _pairs(BATCH_CONFIG)


def test_batch_adapter_sees_its_own_config_after_a_streaming_build():
    """The end-to-end property the bug broke: build streaming, batch is untouched.

    Exercised through the Backend Adapter Factory, which is how the two flows
    actually get wired together at startup.
    """
    from coro.backends.diarization import factory
    from coro.backends.diarization.nemo.diarization import NemoDiarizationAdapter

    model = _make_model()
    batch_adapter = NemoDiarizationAdapter(model)
    before = _snapshot(batch_adapter.model.sortformer_modules)

    factory.build_streaming_diarizer_factory("nemo", batch_adapter, tier="very-high")

    assert _snapshot(batch_adapter.model.sortformer_modules) == before


# ---------------------------------------------------------------------------
# The tier must still actually apply while the model is running
# ---------------------------------------------------------------------------


def test_tier_params_are_applied_during_the_model_call_and_restored_after():
    """Scoped, not removed: NeMo reads these attributes at call time."""
    model = _make_model()
    diarizer = NemoStreamingDiarizerFactory(model, tier="low")()
    diarizer._preprocessor = _make_preprocessor()

    tier_params = get_latency_tier_params("low")
    chunk_bytes = int(tier_params.chunk_len * 8 * 0.01 * SAMPLE_RATE * BYTES_PER_SAMPLE)
    rc_bytes = int(tier_params.chunk_right_context * 8 * 0.01 * SAMPLE_RATE * BYTES_PER_SAMPLE)

    diarizer.ingest_pcm_chunk(_pcm(chunk_bytes + rc_bytes))

    assert model.params_seen_during_call, "model was never called"
    assert model.params_seen_during_call[0] == _pairs(vars(tier_params))
    # Restored the moment the call returned.
    assert _snapshot(model.sortformer_modules) == _pairs(BATCH_CONFIG)


def test_applied_streaming_params_restores_on_exception():
    """A failed model call must not leave the shared model retuned."""
    modules = _FakeSortformerModules()
    before = _snapshot(modules)

    with pytest.raises(RuntimeError), applied_streaming_params(
        modules, get_latency_tier_params("ultra-low")
    ):
        raise RuntimeError("model blew up mid-chunk")

    assert _snapshot(modules) == before
