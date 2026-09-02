"""The caching decorator forwards to the model exactly when it must.

"Was the model called?" is the externally observable fact that matters here, so
every assertion is made against a fake inner adapter that records its calls,
rather than against cache internals like row layout or key strings.
"""

from __future__ import annotations

import pytest

from coro.backends.asr.factory import PROVIDER_HONOURS_PROMPT, LazyASRAdapter
from coro.cache.adapter import CachingASRAdapter, adapter_honours_prompt, unwrap_asr_adapter
from coro.cache.store import ASRCacheStore
from coro.core.models import TranscriptToken

_PCM_A = b"\x01\x02" * 800
_PCM_B = b"\x03\x04" * 800


class _RecordingASR:
    """Inner ASR Adapter that records every call it receives."""

    honours_prompt = False

    def __init__(self) -> None:
        self.calls: list[tuple[bytes, str | None, str | None]] = []

    async def transcribe_pcm(self, pcm, *, language=None, prompt=None):
        self.calls.append((pcm, language, prompt))
        return [
            TranscriptToken(start=0.0, end=1.0, text=f" call{len(self.calls)}", probability=0.5)
        ]


@pytest.fixture
def store(tmp_path):
    with ASRCacheStore(str(tmp_path / "cache"), max_bytes=0, ttl_seconds=0.0) as opened:
        yield opened


def _cached(inner, store, *, fingerprint: str = "fp", honours_prompt: bool | None = None):
    return CachingASRAdapter(
        inner, store=store, fingerprint=fingerprint, honours_prompt=honours_prompt
    )


# MARK: Forwarding
@pytest.mark.asyncio
async def test_a_repeated_window_is_not_forwarded(store):
    inner = _RecordingASR()
    adapter = _cached(inner, store)

    first = await adapter.transcribe_pcm(_PCM_A, language="es")
    second = await adapter.transcribe_pcm(_PCM_A, language="es")

    assert len(inner.calls) == 1
    assert second == first


@pytest.mark.asyncio
async def test_a_different_window_is_forwarded(store):
    inner = _RecordingASR()
    adapter = _cached(inner, store)

    await adapter.transcribe_pcm(_PCM_A)
    await adapter.transcribe_pcm(_PCM_B)

    assert len(inner.calls) == 2


@pytest.mark.asyncio
async def test_a_changed_language_is_forwarded(store):
    inner = _RecordingASR()
    adapter = _cached(inner, store)

    await adapter.transcribe_pcm(_PCM_A, language="es")
    await adapter.transcribe_pcm(_PCM_A, language="en")

    assert len(inner.calls) == 2


@pytest.mark.asyncio
async def test_an_equivalent_language_spelling_is_not_forwarded(store):
    inner = _RecordingASR()
    adapter = _cached(inner, store)

    await adapter.transcribe_pcm(_PCM_A, language="es")
    await adapter.transcribe_pcm(_PCM_A, language=" ES ")

    assert len(inner.calls) == 1


@pytest.mark.asyncio
async def test_a_changed_fingerprint_is_forwarded(store):
    """An upgraded backend must never be served results the new build would not produce."""
    inner = _RecordingASR()

    await _cached(inner, store, fingerprint="before").transcribe_pcm(_PCM_A)
    await _cached(inner, store, fingerprint="after").transcribe_pcm(_PCM_A)

    assert len(inner.calls) == 2


@pytest.mark.asyncio
async def test_a_separate_decorator_over_the_same_store_still_hits(store):
    """The cache is what survives the run, not the decorator instance."""
    first_inner = _RecordingASR()
    second_inner = _RecordingASR()

    await _cached(first_inner, store).transcribe_pcm(_PCM_A)
    await _cached(second_inner, store).transcribe_pcm(_PCM_A)

    assert second_inner.calls == []


# MARK: Prompt Capability
@pytest.mark.asyncio
async def test_a_prompt_difference_is_ignored_for_a_prompt_inert_backend(store):
    """The default backend cannot see the prompt, so it must not fragment its keys."""
    inner = _RecordingASR()
    adapter = _cached(inner, store, honours_prompt=False)

    await adapter.transcribe_pcm(_PCM_A, prompt="earlier words")
    await adapter.transcribe_pcm(_PCM_A, prompt="completely different words")

    assert len(inner.calls) == 1


@pytest.mark.asyncio
async def test_a_prompt_difference_is_forwarded_for_a_prompt_honouring_backend(store):
    inner = _RecordingASR()
    adapter = _cached(inner, store, honours_prompt=True)

    await adapter.transcribe_pcm(_PCM_A, prompt="earlier words")
    await adapter.transcribe_pcm(_PCM_A, prompt="completely different words")

    assert len(inner.calls) == 2


@pytest.mark.asyncio
async def test_the_prompt_still_reaches_a_prompt_honouring_backend(store):
    """Excluding the prompt from the key must never mean withholding it from the model."""
    inner = _RecordingASR()
    adapter = _cached(inner, store, honours_prompt=False)

    await adapter.transcribe_pcm(_PCM_A, prompt="earlier words")

    assert inner.calls[0][2] == "earlier words"


def test_the_capability_defaults_to_the_wrapped_adapters_declaration(store):
    class _PromptHonouring(_RecordingASR):
        honours_prompt = True

    assert _cached(_PromptHonouring(), store).honours_prompt is True
    assert _cached(_RecordingASR(), store).honours_prompt is False


def test_an_undeclared_capability_is_read_pessimistically():
    """An unknown adapter costs hit rate rather than correctness."""
    assert adapter_honours_prompt(object()) is True


def test_every_provider_capability_matches_its_adapter_class():
    """The provider table and the adapter classes must not drift apart."""
    from coro.backends.asr.faster_whisper import FasterWhisperASRAdapter
    from coro.backends.asr.nemo import NemoASRAdapter
    from coro.backends.asr.onnx_asr import OnnxAsrASRAdapter
    from coro.backends.asr.onnx_genai import OnnxGenaiASRAdapter

    declared = {
        "onnx-asr": OnnxAsrASRAdapter.honours_prompt,
        "onnx-genai": OnnxGenaiASRAdapter.honours_prompt,
        "nemo": NemoASRAdapter.honours_prompt,
        "faster-whisper": FasterWhisperASRAdapter.honours_prompt,
    }
    assert declared == PROVIDER_HONOURS_PROMPT


# MARK: Counters And Unwrapping
@pytest.mark.asyncio
async def test_hits_and_misses_are_counted(store):
    adapter = _cached(_RecordingASR(), store)

    await adapter.transcribe_pcm(_PCM_A)
    await adapter.transcribe_pcm(_PCM_B)
    await adapter.transcribe_pcm(_PCM_A)

    assert (adapter.hits, adapter.misses) == (1, 2)


def test_unwrapping_reaches_the_real_adapter(store):
    inner = _RecordingASR()
    assert unwrap_asr_adapter(_cached(inner, store)) is inner


def test_unwrapping_an_undecorated_adapter_returns_it_unchanged():
    inner = _RecordingASR()
    assert unwrap_asr_adapter(inner) is inner


# MARK: Deferred Construction
@pytest.mark.asyncio
async def test_a_fully_cached_run_never_builds_the_adapter(store):
    """The point of a cached re-run is skipping the model load, not just inference."""
    built: list[_RecordingASR] = []

    def _build() -> _RecordingASR:
        built.append(_RecordingASR())
        return built[-1]

    # First run populates the cache, so it must build the adapter.
    await _cached(LazyASRAdapter(_build, honours_prompt=False), store).transcribe_pcm(_PCM_A)
    builds_after_cold_run = len(built)

    lazy = LazyASRAdapter(_build, honours_prompt=False)
    await _cached(lazy, store).transcribe_pcm(_PCM_A)

    assert lazy.loaded is False
    assert (builds_after_cold_run, len(built)) == (1, 1)


@pytest.mark.asyncio
async def test_a_missing_window_builds_the_adapter(store):
    lazy = LazyASRAdapter(_RecordingASR, honours_prompt=False)
    await _cached(lazy, store).transcribe_pcm(_PCM_A)
    assert lazy.loaded is True


@pytest.mark.asyncio
async def test_the_adapter_is_built_only_once(store):
    built: list[_RecordingASR] = []

    def _build() -> _RecordingASR:
        built.append(_RecordingASR())
        return built[-1]

    lazy = LazyASRAdapter(_build, honours_prompt=False)
    await lazy.transcribe_pcm(_PCM_A)
    await lazy.transcribe_pcm(_PCM_B)

    assert len(built) == 1
