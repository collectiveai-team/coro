"""Canary word alignment behavior with in-memory and real models."""

from __future__ import annotations

import wave
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

from coro.backends.asr.factory import build_asr_adapter
from coro.backends.asr.onnx_canary_split_adapter import OnnxCanarySplitASRAdapter
from coro.bench.data import WARMUP_AUDIO_PATH
from coro.core.models import TranscriptToken
from coro.settings import ServerSettings


class _SpanishDecoder:
    """Return a Spanish transcript without timing information."""

    def __init__(self):
        self.language_token_ids = {"es": 1}

    def recognize_batch(self, waveforms, waveforms_len, /, **kwargs):
        yield SimpleNamespace(text="Hola, mundo.")


class _SpanishAligner:
    """Return nonuniform timestamps for the Spanish transcript."""

    def align(self, audio, text):
        assert audio == pytest.approx(np.full(16000, 0.5))
        assert text == "Hola, mundo."
        return [TranscriptToken(0.16, 0.24, " Hola,"), TranscriptToken(0.64, 0.80, " mundo.")]


class _FailingAligner:
    """Return no valid alignment for the decoded text."""

    def align(self, audio, text):
        raise ValueError("No valid word alignment")


async def test_aligns_canarys_text_using_the_decoded_audio():
    adapter = OnnxCanarySplitASRAdapter(_SpanishDecoder(), word_aligner=_SpanishAligner())

    tokens = await adapter.transcribe_pcm(b"\x00\x40" * 16000, language="es")

    assert tokens == [
        TranscriptToken(0.16, 0.24, " Hola,"),
        TranscriptToken(0.64, 0.80, " mundo."),
    ]


async def test_alignment_failure_does_not_fall_back_to_interpolation():
    adapter = OnnxCanarySplitASRAdapter(_SpanishDecoder(), word_aligner=_FailingAligner())

    with pytest.raises(ValueError, match="No valid word alignment"):
        await adapter.transcribe_pcm(b"\x00\x40" * 16000, language="es")


@pytest.fixture(scope="module", params=["canary-nemo", "omnilingual-ctc-int8"])
def aligned_canary(request):
    """Load the split decoder with each real alignment model."""
    return build_asr_adapter(
        ServerSettings(
            asr_device="cpu",
            asr_alignment_model=request.param,
            _env_file=None,
        )
    )


@pytest.fixture
def padded_speech():
    """Append two seconds of silence to the warmup speech."""
    with wave.open(str(WARMUP_AUDIO_PATH)) as wav:
        pcm = wav.readframes(wav.getnframes())
        rate = wav.getframerate()
    return pcm + b"\x00\x00" * rate * 2, len(pcm) / (2 * rate) + 2.0


@pytest.mark.real_model
async def test_canary_ctc_aligns_words_before_trailing_silence(aligned_canary, padded_speech):
    """Align a real speech clip without including its trailing silence."""
    pcm, duration = padded_speech

    words = await aligned_canary.transcribe_pcm(pcm, language="en")

    assert " your country" in "".join(word.text for word in words).lower()
    assert words[-1].end < duration - 1.0
    assert all(0 <= word.start < word.end for word in words)
    assert all(left.end <= right.start for left, right in pairwise(words))
