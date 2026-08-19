"""Tuned ONNX Runtime session options for the onnx-asr Backend Provider.

Verifies the adapter no longer builds inference sessions at library defaults:
the tuned options object is constructed with the intended values and is handed
to both ``onnx_asr.load_model`` and ``onnx_asr.load_vad``. Critically, the
intra-op thread count stays at the ``0`` sentinel, because any explicit value
disables ONNX Runtime's per-core affinitisation.

No real model is loaded; ``onnx_asr`` module functions are patched.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import onnxruntime as rt
import pytest

from coro.backends.asr.onnx_session import (
    AFFINITISING_INTRA_OP_THREADS,
    SPIN_BACKOFF_MAX,
    SPIN_DURATION_US,
    build_asr_session_options,
)


# ---------------------------------------------------------------------------
# build_asr_session_options
# ---------------------------------------------------------------------------


def test_intra_op_threads_left_at_affinitising_sentinel():
    """The intra-op thread count stays at 0 so ORT keeps per-core affinity."""
    assert AFFINITISING_INTRA_OP_THREADS == 0
    assert build_asr_session_options().intra_op_num_threads == 0


def test_graph_optimization_and_execution_mode_are_explicit():
    """Full graph optimization and sequential execution are asserted, not assumed."""
    options = build_asr_session_options()
    assert options.graph_optimization_level == rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    assert options.execution_mode == rt.ExecutionMode.ORT_SEQUENTIAL
    assert options.enable_mem_pattern is True


def test_thread_pool_spin_settings_are_applied():
    """The spin-tuning config entries ORT benchmarking recommends are set."""
    options = build_asr_session_options()
    assert options.get_session_config_entry("session.intra_op.spin_duration_us") == SPIN_DURATION_US
    assert options.get_session_config_entry("session.intra_op.spin_backoff_max") == SPIN_BACKOFF_MAX


# ---------------------------------------------------------------------------
# build_onnx_asr_adapter wiring
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_onnx_asr(monkeypatch):
    """Install a stub ``onnx_asr`` module and yield its mocked loaders."""
    model = SimpleNamespace()
    model.with_timestamps = lambda: model
    model.with_vad = lambda *_args, **_kwargs: model

    module = SimpleNamespace(
        load_model=MagicMock(return_value=model),
        load_vad=MagicMock(return_value=object()),
    )
    monkeypatch.setitem(sys.modules, "onnx_asr", module)
    return module


def test_load_model_receives_the_tuned_session_options(fake_onnx_asr):
    """``sess_options`` is supplied instead of being left at library defaults."""
    from coro.backends.asr.onnx_asr import build_onnx_asr_adapter

    build_onnx_asr_adapter("nemo-parakeet-tdt-0.6b-v3", device="cpu")

    _, kwargs = fake_onnx_asr.load_model.call_args
    options = kwargs["sess_options"]
    assert isinstance(options, rt.SessionOptions)
    assert options.intra_op_num_threads == AFFINITISING_INTRA_OP_THREADS
    assert options.get_session_config_entry("session.intra_op.spin_duration_us") == SPIN_DURATION_US


def test_vad_session_receives_the_same_tuned_options(fake_onnx_asr):
    """The Silero VAD session is tuned too, not left at defaults."""
    from coro.backends.asr.onnx_asr import build_onnx_asr_adapter

    build_onnx_asr_adapter("nemo-parakeet-tdt-0.6b-v3", device="cpu", vad_enabled=True)

    model_options = fake_onnx_asr.load_model.call_args.kwargs["sess_options"]
    vad_options = fake_onnx_asr.load_vad.call_args.kwargs["sess_options"]
    assert vad_options is model_options


def test_builder_wires_the_configured_admission_policy(fake_onnx_asr):
    """Concurrency settings reach the adapter's admission controller."""
    from coro.backends.asr.onnx_asr import build_onnx_asr_adapter

    adapter = build_onnx_asr_adapter(
        "nemo-parakeet-tdt-0.6b-v3", device="cpu", max_concurrency=5, max_queue_depth=9
    )

    assert adapter.admission.max_concurrency == 5
    assert adapter.admission.max_queue_depth == 9
