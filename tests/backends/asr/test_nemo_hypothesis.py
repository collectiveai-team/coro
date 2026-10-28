"""NeMo hypothesis -> word-token conversion for the nemo ASR backend.

Covers :mod:`coro.backends.asr.nemo_hypothesis`: the defensive handling of
``hyp.timestamp`` shapes, the token-id attribute lookup, and the
frame-index -> seconds conversion of the forced-language path. Hypotheses
are ``SimpleNamespace`` doubles; no NeMo checkpoint is loaded.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from coro.backends.asr.nemo_hypothesis import frame_timestamps_to_seconds, tokens_from_hypothesis

# (window_stride, subsampling_factor) verified against the real checkpoint:
# 5.85s clip, frame indices [7, 9, ..., 72] -> seconds [0.56, 0.72, ..., 5.76].
_REAL_FRAME_CONVERSION = (0.01, 8)

# SentencePiece pieces for the fake hypothesis below.
_PIECES = {11: "▁hola", 12: ",", 13: "▁mundo", 14: "."}


class _FakeTokenizer:
    def ids_to_tokens(self, ids):
        return [_PIECES[i] for i in ids]


class _FakeTensorTimestamp:
    """Duck-types a torch.Tensor closely enough to trip the guard in
    ``tokens_from_hypothesis`` (has ``.numel()``) without a torch dependency
    in this test module. Deliberately not iterable -- the guard must
    short-circuit on the ``numel`` check alone, matching the real bug: the
    original code crashed evaluating ``X or []`` truthiness *before* ever
    reaching iteration.
    """

    def numel(self) -> int:
        return 37


class TestTokensFromHypothesis:
    def test_tensor_timestamp_is_discarded_not_treated_as_seconds(self):
        """A raw per-token frame-index Tensor (verified shape from the real
        checkpoint) must never be treated as flat per-token seconds -- that
        would silently produce garbled word timing. Falls back to
        evenly-spaced text timing instead.
        """
        hyp = SimpleNamespace(
            y_sequence=(11, 12, 13, 14),
            timestamp=_FakeTensorTimestamp(),
            text="hola, mundo.",
        )
        tokens = tokens_from_hypothesis(hyp, _FakeTokenizer(), span_end=1.0)
        assert [t.text for t in tokens] == [" hola,", " mundo."]
        assert tokens[0].start == 0.0

    def test_prefers_y_sequence_over_legacy_y_attribute(self):
        """y_sequence is the real NeMo Hypothesis field; a stale/wrong ``y``
        value (99s aren't in the fake tokenizer's piece table) proves
        ``y_sequence`` -- not ``y`` -- is what actually gets read.
        """
        hyp = SimpleNamespace(
            y_sequence=(11, 12, 13, 14),
            y=(99, 99, 99, 99),
            timestamp=[0.1, 0.2, 0.5, 0.6],
            text="hola, mundo.",
        )
        tokens = tokens_from_hypothesis(hyp, _FakeTokenizer(), span_end=1.0)
        assert [t.text for t in tokens] == [" hola,", " mundo."]

    def test_falls_back_to_legacy_y_attribute_when_y_sequence_absent(self):
        hyp = SimpleNamespace(
            y=(11, 12, 13, 14), timestamp=[0.1, 0.2, 0.5, 0.6], text="hola, mundo."
        )
        tokens = tokens_from_hypothesis(hyp, _FakeTokenizer(), span_end=1.0)
        assert [t.text for t in tokens] == [" hola,", " mundo."]


class TestFrameTimestampsToSeconds:
    def test_converts_frame_indices_using_window_stride_and_subsampling(self):
        """Matches values verified directly against the real checkpoint."""
        seconds = frame_timestamps_to_seconds([7, 9, 10, 72], _REAL_FRAME_CONVERSION)
        assert seconds == pytest.approx([0.56, 0.72, 0.8, 5.76], abs=1e-9)

    def test_returns_none_when_raw_timestamp_is_none(self):
        assert frame_timestamps_to_seconds(None, _REAL_FRAME_CONVERSION) is None

    def test_returns_none_when_frame_conversion_is_none(self):
        assert frame_timestamps_to_seconds([7, 9], None) is None

    def test_unwraps_tensor_like_via_tolist(self):
        class _FakeTensor:
            def tolist(self):
                return [7, 9]

        seconds = frame_timestamps_to_seconds(_FakeTensor(), _REAL_FRAME_CONVERSION)
        assert seconds == pytest.approx([0.56, 0.72], abs=1e-9)

    def test_discards_structured_dict_shape(self):
        """The auto-detection path's {"word": ..., "char": ..., "segment":
        ...} shape must never be misread as a flat frame-index sequence.
        """
        assert frame_timestamps_to_seconds({"word": []}, _REAL_FRAME_CONVERSION) is None
