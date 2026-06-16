"""onnx-asr backend conversion + adapter tests using fake onnx-asr objects.

Verify that the onnx-asr adapter:
- Reconstructs words from subword tokens (space-prefixed word starts, as emitted by
  onnx-asr for NeMo Parakeet; also the SentencePiece ``\u2581`` marker variant).
- Synthesises each word's end from the next word's start (final word padded).
- Applies offset_seconds to timestamps.
- Aggregates token log-probabilities into a word probability (exp of mean).
- Returns an empty list when tokens/timestamps are missing.
- Drives ``OnnxAsrASRAdapter.transcribe_pcm`` against a stub model (no real inference).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from coro.backends.onnx_asr import (
    OnnxAsrASRAdapter,
    _LAST_WORD_PAD,
    convert_onnx_asr_result,
    convert_onnx_asr_segments,
)
from coro.core.types import TranscriptToken

_SP = "\u2581"


def _result(tokens, timestamps, logprobs=None):
    return SimpleNamespace(tokens=tokens, timestamps=timestamps, logprobs=logprobs)


# ---------------------------------------------------------------------------
# convert_onnx_asr_result
# ---------------------------------------------------------------------------


def test_reconstructs_words_from_space_prefixed_subwords():
    """Space-prefixed subword tokens (onnx-asr Parakeet format) group into words."""
    result = _result(
        tokens=[" hel", "lo", " world"],
        timestamps=[0.0, 0.2, 0.5],
    )
    tokens = convert_onnx_asr_result(result)
    assert all(isinstance(t, TranscriptToken) for t in tokens)
    assert [t.text for t in tokens] == [" hello", " world"]


def test_reconstructs_words_from_sentencepiece_marker():
    """The SentencePiece marker variant is also supported and normalised to spaces."""
    result = _result(
        tokens=[f"{_SP}hel", "lo", f"{_SP}world"],
        timestamps=[0.0, 0.2, 0.5],
    )
    tokens = convert_onnx_asr_result(result)
    assert [t.text for t in tokens] == [" hello", " world"]


def test_word_text_join_reconstructs_spaced_sentence():
    """Concatenating token text (response-builder convention) yields spaced words."""
    result = _result(
        tokens=[" the", " quick", " fox"],
        timestamps=[0.0, 0.4, 0.8],
    )
    tokens = convert_onnx_asr_result(result)
    assert "".join(t.text for t in tokens) == " the quick fox"


def test_punctuation_token_attaches_to_word():
    """A leading-punctuation token (no space) stays on the current word."""
    result = _result(
        tokens=[" so", ",", " my"],
        timestamps=[0.0, 0.3, 0.6],
    )
    tokens = convert_onnx_asr_result(result)
    assert [t.text for t in tokens] == [" so,", " my"]


def test_word_end_is_next_word_start():
    """Each non-final word ends where the next word begins."""
    result = _result(
        tokens=[" a", " b"],
        timestamps=[1.0, 2.5],
    )
    tokens = convert_onnx_asr_result(result)
    assert tokens[0].start == pytest.approx(1.0)
    assert tokens[0].end == pytest.approx(2.5)


def test_final_word_end_is_padded():
    """The final word, having no successor, is padded by _LAST_WORD_PAD."""
    result = _result(tokens=[" solo"], timestamps=[3.0])
    tokens = convert_onnx_asr_result(result)
    assert tokens[0].start == pytest.approx(3.0)
    assert tokens[0].end == pytest.approx(3.0 + _LAST_WORD_PAD)


def test_applies_offset_seconds():
    """Timestamps are shifted by offset_seconds."""
    result = _result(tokens=[" x", " y"], timestamps=[0.0, 1.0])
    tokens = convert_onnx_asr_result(result, offset_seconds=10.0)
    assert tokens[0].start == pytest.approx(10.0)
    assert tokens[0].end == pytest.approx(11.0)
    assert tokens[1].start == pytest.approx(11.0)


def test_probability_is_exp_mean_logprob():
    """Word probability is exp(mean(subword logprobs))."""
    result = _result(
        tokens=[" hel", "lo"],
        timestamps=[0.0, 0.2],
        logprobs=[math.log(0.5), math.log(0.5)],
    )
    tokens = convert_onnx_asr_result(result)
    assert tokens[0].probability == pytest.approx(0.5)


def test_probability_none_without_logprobs():
    """Probability is None when log-probabilities are unavailable."""
    result = _result(tokens=[" hi"], timestamps=[0.0])
    tokens = convert_onnx_asr_result(result)
    assert tokens[0].probability is None


def test_empty_when_no_tokens():
    """Missing tokens/timestamps (and no text) yield an empty list."""
    assert convert_onnx_asr_result(_result(tokens=[], timestamps=[])) == []
    assert convert_onnx_asr_result(_result(tokens=None, timestamps=None)) == []


def _text_result(text, start=None, end=None):
    """A text-only result (onnx-asr Whisper shape): tokens/timestamps are None."""
    ns = SimpleNamespace(tokens=None, timestamps=None, logprobs=None, text=text)
    if start is not None:
        ns.start = start
    if end is not None:
        ns.end = end
    return ns


def test_text_only_result_spreads_words_across_span():
    """Whisper-shaped result (text, no tokens) yields words spread over the span."""
    tokens = convert_onnx_asr_result(_text_result("and so my fellow"), span_end=4.0)
    assert [t.text for t in tokens] == [" and", " so", " my", " fellow"]
    assert tokens[0].start == pytest.approx(0.0)
    assert tokens[0].end == pytest.approx(1.0)
    assert tokens[3].start == pytest.approx(3.0)
    assert tokens[3].end == pytest.approx(4.0)
    assert "".join(t.text for t in tokens) == " and so my fellow"


def test_text_only_result_without_span_uses_pad_steps():
    """Without a span, words step by _LAST_WORD_PAD so timings stay monotonic."""
    tokens = convert_onnx_asr_result(_text_result("uno dos"))
    assert tokens[0].start == pytest.approx(0.0)
    assert tokens[1].start == pytest.approx(_LAST_WORD_PAD)


def test_text_only_empty_text_yields_no_tokens():
    """A text-only result with blank text yields no tokens."""
    assert convert_onnx_asr_result(_text_result("   "), span_end=5.0) == []


def test_segments_text_only_rebased_to_segment_span():
    """VAD text-only segments rebase word timings onto each segment's [start, end]."""
    segments = [
        _text_result("hola mundo", start=10.0, end=12.0),
        _text_result("adios", start=20.0, end=21.0),
    ]
    tokens = convert_onnx_asr_segments(segments)
    assert [t.text for t in tokens] == [" hola", " mundo", " adios"]
    assert tokens[0].start == pytest.approx(10.0)
    assert tokens[1].start == pytest.approx(11.0)
    assert tokens[2].start == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# OnnxAsrASRAdapter.transcribe_pcm
# ---------------------------------------------------------------------------


class _StubModel:
    """Stub onnx-asr model capturing recognize kwargs and returning a fixed result."""

    def __init__(self):
        self.calls: list[dict] = []

    def recognize(self, audio, **kwargs):
        self.calls.append({"len": len(audio), **kwargs})
        return _result(
            tokens=[" hello", " world"],
            timestamps=[0.0, 0.5],
        )


async def test_adapter_transcribe_pcm_returns_tokens():
    """transcribe_pcm decodes PCM and returns reconstructed TranscriptTokens."""
    model = _StubModel()
    adapter = OnnxAsrASRAdapter(model)
    pcm = np.zeros(16000, dtype=np.int16).tobytes()

    tokens = await adapter.transcribe_pcm(pcm, language="en")

    assert [t.text for t in tokens] == [" hello", " world"]
    assert model.calls[0]["sample_rate"] == 16000
    assert model.calls[0]["language"] == "en"
    assert model.calls[0]["len"] == 16000


async def test_adapter_omits_language_when_absent():
    """No language kwarg is forwarded when language is None."""
    model = _StubModel()
    adapter = OnnxAsrASRAdapter(model)
    pcm = np.zeros(8000, dtype=np.int16).tobytes()

    await adapter.transcribe_pcm(pcm)

    assert "language" not in model.calls[0]


# ---------------------------------------------------------------------------
# VAD-segmented path (convert_onnx_asr_segments + adapter vad_enabled)
# ---------------------------------------------------------------------------


def _segment(start, end, tokens, timestamps, logprobs=None):
    """A TimestampedSegmentResult-like object (segment-relative token timestamps)."""
    return SimpleNamespace(
        start=start, end=end, text="", tokens=tokens, timestamps=timestamps, logprobs=logprobs
    )


def test_convert_segments_rebases_token_times_by_segment_start():
    """Per-segment relative timestamps become absolute via the segment start offset."""
    segments = [
        _segment(0.0, 1.0, [" hello", " world"], [0.0, 0.5]),
        _segment(10.0, 11.0, [" foo", " bar"], [0.0, 0.4]),
    ]
    tokens = convert_onnx_asr_segments(segments)

    assert [t.text for t in tokens] == [" hello", " world", " foo", " bar"]
    # First segment starts at 0.0; second segment's tokens are offset by 10.0s.
    assert tokens[0].start == pytest.approx(0.0)
    assert tokens[2].start == pytest.approx(10.0)
    assert tokens[3].start == pytest.approx(10.4)


def test_convert_segments_empty_iterable_returns_empty():
    assert convert_onnx_asr_segments([]) == []


class _VadStubModel:
    """Stub VAD model: recognize yields an iterator of segment results."""

    def recognize(self, audio, **kwargs):
        return iter(
            [
                _segment(0.0, 1.0, [" hello"], [0.0]),
                _segment(5.0, 6.0, [" world"], [0.0]),
            ]
        )


async def test_adapter_vad_enabled_merges_segments_with_offsets():
    """vad_enabled adapter consumes the segment iterator and re-bases timestamps."""
    adapter = OnnxAsrASRAdapter(_VadStubModel(), vad_enabled=True)
    pcm = np.zeros(16000, dtype=np.int16).tobytes()

    tokens = await adapter.transcribe_pcm(pcm)

    assert [t.text for t in tokens] == [" hello", " world"]
    assert tokens[0].start == pytest.approx(0.0)
    assert tokens[1].start == pytest.approx(5.0)
