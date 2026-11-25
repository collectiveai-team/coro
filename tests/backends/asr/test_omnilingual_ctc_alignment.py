"""Omnilingual forced alignment through the adapter's public interface."""

import numpy as np
import pytest

from coro.backends.asr.omnilingual_ctc_alignment import OmnilingualCanaryWordAligner
from coro.core.models import TranscriptToken


class _Emissions:
    """Supply a deterministic frame for each symbol in an expected CTC path."""

    def __init__(self, labels):
        self._labels = labels

    def run(self, output_names, inputs):
        assert output_names == ["logits"]
        assert inputs["x"].shape == (1, 3200)
        return [np.eye(4, dtype=np.float32)[self._labels][None] * 10]


def test_repeated_characters_use_distinct_frames_and_keep_original_punctuation():
    aligner = OmnilingualCanaryWordAligner(
        _Emissions([0, 1, 0, 1, 2, 3, 0]), {"a": 1, " ": 2, "ñ": 3}
    )

    words = aligner.align(np.ones(3200, dtype=np.float32), "Aa, Ñ.")

    assert words == [TranscriptToken(0.02, 0.08, " Aa,"), TranscriptToken(0.10, 0.12, " Ñ.")]


def test_unreachable_transcript_is_not_interpolated():
    aligner = OmnilingualCanaryWordAligner(_Emissions([1]), {"a": 1, " ": 2})

    with pytest.raises(ValueError, match="no complete path"):
        aligner.align(np.ones(3200, dtype=np.float32), "aa")
