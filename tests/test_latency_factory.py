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
        from coro.backends.nemo_streaming import get_latency_tier_params

        params = get_latency_tier_params("very-high")
        assert params["chunk_len"] == 340
        assert params["chunk_right_context"] == 40
        assert params["fifo_len"] == 40
        assert params["spkcache_update_period"] == 300
        assert params["spkcache_len"] == 188

    def test_high_params(self):
        from coro.backends.nemo_streaming import get_latency_tier_params

        params = get_latency_tier_params("high")
        assert params["chunk_len"] == 124
        assert params["chunk_right_context"] == 1

    def test_low_params(self):
        from coro.backends.nemo_streaming import get_latency_tier_params

        params = get_latency_tier_params("low")
        assert params["chunk_len"] == 6
        assert params["chunk_right_context"] == 7

    def test_ultra_low_params(self):
        from coro.backends.nemo_streaming import get_latency_tier_params

        params = get_latency_tier_params("ultra-low")
        assert params["chunk_len"] == 3

    def test_all_tiers_have_required_keys(self):
        from coro.backends.nemo_streaming import get_latency_tier_params

        required = {
            "chunk_len",
            "chunk_right_context",
            "fifo_len",
            "spkcache_update_period",
            "spkcache_len",
        }
        for tier in ("very-high", "high", "low", "ultra-low"):
            assert required.issubset(get_latency_tier_params(tier).keys())

    def test_params_returns_copy(self):
        from coro.backends.nemo_streaming import (
            LATENCY_TIER_PARAMS,
            get_latency_tier_params,
        )

        p1 = get_latency_tier_params("very-high")
        p1["chunk_len"] = 999
        assert LATENCY_TIER_PARAMS["very-high"]["chunk_len"] == 340


class TestStreamingDiarizerFactory:
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

    def test_factory_applies_tier_params_to_model(self):
        from coro.backends.nemo_streaming import StreamingDiarizerFactory

        model = self._make_mock_model()
        StreamingDiarizerFactory(model, tier="very-high")
        assert model.sortformer_modules.chunk_len == 340
        assert model.sortformer_modules.chunk_right_context == 40
        assert model.sortformer_modules.fifo_len == 40
        assert model.sortformer_modules.spkcache_update_period == 300
        assert model.sortformer_modules.spkcache_len == 188
        model.sortformer_modules._check_streaming_parameters.assert_called_once()

    def test_factory_produces_distinct_instances(self):
        from coro.backends.nemo_streaming import StreamingDiarizerFactory

        model = self._make_mock_model()
        factory = StreamingDiarizerFactory(model, tier="low")
        d1 = factory()
        d2 = factory()
        assert d1 is not d2
        assert d1._pcm_buffer == b""
        assert d2._pcm_buffer == b""

    def test_factory_default_tier_is_very_high(self):
        from coro.backends.nemo_streaming import StreamingDiarizerFactory

        model = self._make_mock_model()
        factory = StreamingDiarizerFactory(model)
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
