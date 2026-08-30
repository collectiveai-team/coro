"""Opt-in real-model check that ASR is a pure function of its inputs.

Any cache keyed on (window PCM, ASR fingerprint, prompt) is only sound if
``transcribe_pcm`` returns identical tokens for identical inputs. This guards
the three ways that can break:

* **Repeat** — the same window twice in a row must agree.
* **Interleaving** — processing other windows in between must not perturb a
  window's result, i.e. the adapter carries no hidden cross-call state.
* **Concurrency** — running windows concurrently through the shared adapter
  must match running them sequentially.

If any of these fail, a cache hit would not reproduce a cache miss, so the
cache could not be transparent and must not be enabled for that backend.

The checks run against the *configured* backend, so pointing ``CORO_BACKEND_ASR``
and ``CORO_MODEL_ASR`` at a deployment's real selection validates that
deployment. Determinism is not portable across devices: a GPU build must be
re-verified rather than assumed from a CPU result.

To run:

    CORO_RUN_REAL_MODEL_TESTS=1 uv run pytest tests/test_asr_determinism.py -v
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import asdict

import pytest

REAL_MODEL_TESTS = os.environ.get("CORO_RUN_REAL_MODEL_TESTS", "0") == "1"
skip_unless_real = pytest.mark.skipif(
    not REAL_MODEL_TESTS,
    reason=(
        "Skipping real-model determinism check. "
        "Set CORO_RUN_REAL_MODEL_TESTS=1 to run with real model checkpoints."
    ),
)

WINDOW_COUNT = 3
PROMPT = "El presidente habla."


def _tokens(token_list) -> list[dict]:
    """Normalise tokens to plain dicts for exact comparison."""
    return [asdict(t) for t in token_list]


async def _transcribe(adapter, window: bytes) -> list[dict]:
    """Transcribe one window with the fixed language and prompt used throughout."""
    return _tokens(await adapter.transcribe_pcm(window, language="en", prompt=PROMPT))


@pytest.fixture(scope="module")
def asr_adapter():
    """Build the configured ASR Adapter once for the whole module."""
    from coro.backends.asr.factory import build_asr_adapter
    from coro.settings import ServerSettings

    return build_asr_adapter(ServerSettings())


@pytest.fixture(scope="module")
def pcm_windows() -> list[bytes]:
    """Split the warmup asset into equal, sample-aligned PCM windows."""
    from coro.audio import convert_path_to_pcm_bytes
    from coro.bench.data import WARMUP_AUDIO_PATH

    pcm = asyncio.run(convert_path_to_pcm_bytes(str(WARMUP_AUDIO_PATH)))
    size = len(pcm) // WINDOW_COUNT
    size -= size % 2
    return [pcm[i * size : (i + 1) * size] for i in range(WINDOW_COUNT)]


@skip_unless_real
@pytest.mark.asyncio
async def test_repeated_transcription_of_a_window_is_identical(asr_adapter, pcm_windows):
    """The same window and prompt must transcribe identically on every call."""
    assert len(pcm_windows) == WINDOW_COUNT

    first_pass = []
    second_pass = []
    for window in pcm_windows:
        first_pass.append(await _transcribe(asr_adapter, window))
        second_pass.append(await _transcribe(asr_adapter, window))

    assert first_pass == second_pass


@skip_unless_real
@pytest.mark.asyncio
async def test_transcription_does_not_depend_on_surrounding_windows(asr_adapter, pcm_windows):
    """Neighbouring calls must not perturb a window's tokens.

    A resumed run replays a different call sequence than the run that populated
    the cache, so any hidden cross-call state inside the adapter would make
    cached tokens diverge from freshly computed ones.
    """
    baseline = await _transcribe(asr_adapter, pcm_windows[0])
    for other in pcm_windows[1:]:
        await _transcribe(asr_adapter, other)
    after = await _transcribe(asr_adapter, pcm_windows[0])

    assert baseline == after, "adapter carries state across calls"


@skip_unless_real
@pytest.mark.asyncio
async def test_concurrent_transcription_matches_sequential(asr_adapter, pcm_windows):
    """Windows sharing the adapter concurrently must match sequential results."""
    assert len(pcm_windows) == WINDOW_COUNT

    sequential = [await _transcribe(asr_adapter, window) for window in pcm_windows]
    concurrent = await asyncio.gather(*(_transcribe(asr_adapter, w) for w in pcm_windows))

    assert sequential == concurrent
