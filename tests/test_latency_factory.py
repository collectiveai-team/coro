"""Diarization Latency Selection and Streaming Diarizer Factory tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from coro.settings import ServerSettings


class TestLatencyTierSettings:
    def test_default_diarization_latency_is_very_high(self):
        settings = ServerSettings(_env_file=None)
        assert settings.diarization_latency == "very-high"

    @pytest.mark.parametrize("tier", ["very-high", "high", "low", "ultra-low"])
    def test_valid_tier_accepted(self, tier):
        settings = ServerSettings(diarization_latency=tier, _env_file=None)
        assert settings.diarization_latency == tier

    def test_unknown_tier_rejected(self):
        with pytest.raises(ValidationError):
            # Intentionally invalid value to assert strict validation.
            ServerSettings(
                diarization_latency="medium",  # pyrefly: ignore[bad-argument-type]
                _env_file=None,
            )


class TestLatencyTierMapping:
    def test_very_high_params(self):
        from coro.backends.diarization.nemo.streaming import get_latency_tier_params

        params = get_latency_tier_params("very-high")
        assert params.chunk_len == 340
        assert params.chunk_right_context == 40
        assert params.fifo_len == 40
        assert params.spkcache_update_period == 300
        assert params.spkcache_len == 188

    def test_high_params(self):
        from coro.backends.diarization.nemo.streaming import get_latency_tier_params

        params = get_latency_tier_params("high")
        assert params.chunk_len == 124
        assert params.chunk_right_context == 1

    def test_low_params(self):
        from coro.backends.diarization.nemo.streaming import get_latency_tier_params

        params = get_latency_tier_params("low")
        assert params.chunk_len == 6
        assert params.chunk_right_context == 7

    def test_ultra_low_params(self):
        from coro.backends.diarization.nemo.streaming import get_latency_tier_params

        params = get_latency_tier_params("ultra-low")
        assert params.chunk_len == 3

    def test_all_tiers_have_required_fields(self):
        import dataclasses

        from coro.backends.diarization.nemo.streaming import get_latency_tier_params

        required = {
            "chunk_len",
            "chunk_right_context",
            "fifo_len",
            "spkcache_update_period",
            "spkcache_len",
        }
        for tier in ("very-high", "high", "low", "ultra-low"):
            params = get_latency_tier_params(tier)
            assert required == {f.name for f in dataclasses.fields(params)}

    def test_params_are_immutable(self):
        import dataclasses

        from coro.backends.diarization.nemo.streaming import (
            LATENCY_TIER_PARAMS,
            get_latency_tier_params,
        )

        p1 = get_latency_tier_params("very-high")
        # C51 looks for a call inside the block; the raise here comes from an
        # attribute assignment on a frozen dataclass, which it does not model.
        with pytest.raises(dataclasses.FrozenInstanceError):  # falsegreen: ignore
            p1.chunk_len = 999  # type: ignore[misc]
        assert LATENCY_TIER_PARAMS["very-high"].chunk_len == 340


class TestNemoStreamingDiarizerFactory:
    def _make_mock_model(self):
        import torch

        model = MagicMock()
        model.device = torch.device("cpu")
        sortformer_modules = MagicMock()
        sortformer_modules.chunk_len = 6
        sortformer_modules.subsampling_factor = 8
        sortformer_modules.n_spk = 4
        sortformer_modules.fc_d_model = 512
        sortformer_modules.chunk_right_context = 1
        sortformer_modules.fifo_len = 188
        sortformer_modules.spkcache_update_period = 144
        sortformer_modules.spkcache_len = 188
        sortformer_modules.init_streaming_state.return_value = {"step": 0}
        model.sortformer_modules = sortformer_modules
        model.forward_streaming_step = MagicMock(return_value=({"step": 1}, None))
        return model

    def test_factory_validates_the_tier_without_retuning_the_shared_model(self):
        """Construction validates the tier but leaves the shared model as found.

        The tier used to be written onto ``model.sortformer_modules``
        permanently at construction. That object is shared with the batch
        Diarization Adapter, so building the streaming factory silently
        retuned batch diarization. The parameters are now scoped to each model
        call instead; see tests/test_streaming_factory_shared_state.py and
        ADR 0010.
        """
        from coro.backends.diarization.nemo.streaming import NemoStreamingDiarizerFactory

        model = self._make_mock_model()
        before = {
            "chunk_len": model.sortformer_modules.chunk_len,
            "chunk_right_context": model.sortformer_modules.chunk_right_context,
            "fifo_len": model.sortformer_modules.fifo_len,
            "spkcache_update_period": model.sortformer_modules.spkcache_update_period,
            "spkcache_len": model.sortformer_modules.spkcache_len,
        }

        factory = NemoStreamingDiarizerFactory(model, tier="very-high")

        # The tier is held by the factory...
        assert factory._tier_params.chunk_len == 340
        assert factory._tier_params.chunk_right_context == 40
        assert factory._tier_params.fifo_len == 40
        assert factory._tier_params.spkcache_update_period == 300
        assert factory._tier_params.spkcache_len == 188
        # ...validated against NeMo's own constraints...
        model.sortformer_modules._check_streaming_parameters.assert_called_once()
        # ...and not left written onto the model the batch adapter also uses.
        assert model.sortformer_modules.chunk_len == before["chunk_len"]
        assert model.sortformer_modules.chunk_right_context == before["chunk_right_context"]
        assert model.sortformer_modules.fifo_len == before["fifo_len"]
        assert model.sortformer_modules.spkcache_update_period == (before["spkcache_update_period"])
        assert model.sortformer_modules.spkcache_len == before["spkcache_len"]

    def test_factory_produces_distinct_instances(self):
        from coro.backends.diarization.nemo.streaming import NemoStreamingDiarizerFactory

        model = self._make_mock_model()
        factory = NemoStreamingDiarizerFactory(model, tier="low")
        d1 = factory()
        d2 = factory()
        assert d1 is not d2
        assert d1._pcm_buffer == b""
        assert d2._pcm_buffer == b""

    def test_factory_default_tier_is_very_high(self):
        from coro.backends.diarization.nemo.streaming import NemoStreamingDiarizerFactory

        model = self._make_mock_model()
        factory = NemoStreamingDiarizerFactory(model)
        assert factory._tier == "very-high"


class TestRuntimeWiring:
    def test_runtime_has_streaming_factory_fields(self):
        from coro.runtime import RuntimeState

        state = RuntimeState()
        assert state.streaming_diarizer_factory is None
        assert state.diarization_latency is None

    def test_health_reports_latency_when_set(self):
        from coro.runtime import RuntimeState

        state = RuntimeState(diarization_latency="very-high")
        assert state.diarization_latency == "very-high"
