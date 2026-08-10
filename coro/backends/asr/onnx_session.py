"""Tuned ONNX Runtime session options for the ONNX ASR Backend Providers.

``onnx_asr.load_model`` and ``onnx_asr.load_vad`` both accept a ``sess_options``
argument. Leaving it unset builds every ``InferenceSession`` at library defaults,
which is what this module exists to stop.

What is set, and why:

* ``graph_optimization_level = ORT_ENABLE_ALL`` and
  ``execution_mode = ORT_SEQUENTIAL`` and ``enable_mem_pattern = True`` are
  asserted explicitly. They currently match ORT's own defaults, so they pin
  behaviour against a future default change rather than changing it today. They
  are the right values regardless: the ASR graphs are deep chains rather than
  branchy ones (sequential wins), and fixed-size ASR windows give static shapes
  (memory patterns apply).
* ``session.intra_op.spin_duration_us`` and ``session.intra_op.spin_backoff_max``
  are the thread-pool spin settings ORT's own benchmarking identifies as the most
  consistent performers, and they are *not* defaults. These are the entries that
  actually change runtime behaviour.

What is deliberately not set:

* ``intra_op_num_threads`` stays at the ``0`` sentinel. Any explicit value
  disables ORT's per-core affinitisation, which costs more than the thread count
  buys back.
* No OpenMP environment tuning. The official ORT CPU wheels use their own Eigen
  thread pool, so ``OMP_NUM_THREADS`` / ``KMP_AFFINITY`` are inert.
* No I/O binding. It optimises host-device transfer and is a no-op on the CPU
  execution provider.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import onnxruntime as rt

# ORT thread-pool spin tuning (microseconds spent spinning before parking, and
# the maximum backoff multiplier). Values from ORT's published benchmark sweep.
SPIN_DURATION_US = "1000"
SPIN_BACKOFF_MAX = "8"

# The ``intra_op_num_threads`` value that keeps per-core affinitisation enabled.
AFFINITISING_INTRA_OP_THREADS = 0


def build_asr_session_options() -> rt.SessionOptions:
    """Build tuned ``SessionOptions`` for ONNX ASR inference sessions.

    Returns:
        A ``SessionOptions`` instance safe to share across every session an
        onnx-asr model builds (ORT copies the options at session construction).

    """
    import onnxruntime as rt

    options = rt.SessionOptions()
    options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    options.enable_mem_pattern = True
    # Left at the sentinel on purpose — see the module docstring.
    options.intra_op_num_threads = AFFINITISING_INTRA_OP_THREADS
    options.add_session_config_entry("session.intra_op.spin_duration_us", SPIN_DURATION_US)
    options.add_session_config_entry("session.intra_op.spin_backoff_max", SPIN_BACKOFF_MAX)
    return options
