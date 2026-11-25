"""Align Canary's transcript with Omnilingual CTC 300M INT8 on CPU."""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from coro.backends.asr.ctc_viterbi import ctc_state_path
from coro.backends.asr.onnx_session import build_asr_session_options
from coro.core.models import TranscriptToken

if TYPE_CHECKING:
    import numpy.typing as npt
    import onnxruntime as ort

OMNILINGUAL_CTC_REVISION = "6abf1ece20cd2308bdb7d13cd78ec1c44fa4c094"
_REPO = "csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12"
_SAMPLE_RATE = 16000
_FRAME_SECONDS = 0.02
_BLANK = 0
_MIN_SAMPLES = 400


class OmnilingualCanaryWordAligner:
    """Score normalized waveforms and align a character sequence on the CPU."""

    def __init__(self, session: ort.InferenceSession, vocabulary: dict[str, int]) -> None:
        self._session = session
        self._vocabulary = vocabulary

    def align(self, audio: npt.NDArray[np.float32], text: str) -> list[TranscriptToken]:
        """Return the original words with boundaries in the decoded audio."""
        words = text.split()
        if not words:
            return []
        if len(audio) < _MIN_SAMPLES:
            raise ValueError("Omnilingual CTC alignment needs at least 25 ms of audio")
        labels, spans = _targets(words, self._vocabulary)
        centered = audio - audio.mean()
        normalized = centered / np.sqrt(np.mean(centered**2) + np.float32(1e-5))
        logits = self._session.run(["logits"], {"x": normalized[None, :]})[0]
        if not isinstance(logits, np.ndarray) or logits.ndim != 3:
            raise ValueError("Omnilingual CTC returned invalid emissions")
        path = ctc_state_path(logits[0], labels, blank_id=_BLANK)
        return _word_boundaries(words, spans, path, len(audio) / _SAMPLE_RATE)


def _targets(
    words: list[str], vocabulary: dict[str, int]
) -> tuple[list[int], list[tuple[int, int]]]:
    """Normalize only the alignment target; output words retain spelling and punctuation."""
    labels: list[int] = []
    spans: list[tuple[int, int]] = []
    for word in words:
        normalized = unicodedata.normalize("NFC", word.casefold().replace("\u2019", "'"))
        chars = [ch for ch in normalized if ch.isalnum() or ch == "'"]
        if not chars or any(ch not in vocabulary for ch in chars):
            raise ValueError("Omnilingual CTC cannot tokenize a transcript word")
        if labels:
            labels.append(vocabulary[" "])
        first = len(labels)
        labels.extend(vocabulary[ch] for ch in chars)
        spans.append((2 * first + 1, 2 * (len(labels) - 1) + 1))
    return labels, spans


def _word_boundaries(
    words: list[str],
    spans: list[tuple[int, int]],
    path: npt.NDArray[np.int64],
    duration: float,
) -> list[TranscriptToken]:
    """Project each target label range onto its first and last visited frames."""
    boundaries = []
    for word, (first, last) in zip(words, spans, strict=True):
        start = float(np.searchsorted(path, first)) * _FRAME_SECONDS
        end = min(duration, float(np.searchsorted(path, last, side="right")) * _FRAME_SECONDS)
        boundaries.append(
            TranscriptToken(start=round(start, 3), end=round(end, 3), text=" " + word)
        )
    return boundaries


def build_omnilingual_word_aligner(*, hf_token: str | None) -> OmnilingualCanaryWordAligner:
    """Load the pinned CPU export and its character vocabulary once."""
    import onnxruntime as ort
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            _REPO,
            revision=OMNILINGUAL_CTC_REVISION,
            allow_patterns=["model.int8.onnx", "tokens.txt"],
            token=hf_token,
        )
    )
    vocabulary = {
        symbol: int(index)
        for line in (snapshot / "tokens.txt").read_text(encoding="utf-8").split("\n")
        if line
        for symbol, index in [line.rsplit(" ", 1)]
    }
    session = ort.InferenceSession(
        snapshot / "model.int8.onnx",
        build_asr_session_options(),
        providers=["CPUExecutionProvider"],
    )
    return OmnilingualCanaryWordAligner(session, vocabulary)
