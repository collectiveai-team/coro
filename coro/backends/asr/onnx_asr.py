"""ONNX ASR Model Integration (onnx-asr backend).

Wraps the ``onnx-asr`` runtime so NeMo Parakeet/Canary-family ONNX models can serve
as a drop-in ASR backend alongside faster-whisper.

onnx-asr's ``.with_timestamps()`` returns parallel flat lists at *token* (subword)
granularity: ``tokens: list[str]`` (decoded subword pieces where a word start is prefixed
by a space; NeMo models may instead use the SentencePiece ``\u2581`` marker),
``timestamps: list[float]`` (one emission time per token, not start/end pairs), and
``logprobs: list[float]``. The pipeline's Project-Owned ``TranscriptToken`` model is
word-level with both ``start`` and ``end``, and ``core/response.py`` groups segments by
punctuation at token boundaries -- so this adapter reconstructs words from subword tokens
and synthesises each word's ``end`` from the next word's start time.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Final, Literal, Protocol, cast

import numpy as np

from coro.backends.asr.concurrency import AdmissionController, build_admission_controller
from coro.backends.asr.onnx_session import build_asr_session_options
from coro.backends.asr.subword_tokens import (
    group_subwords,
    words_from_subword_groups,
    words_from_text,
)
from coro.core.logger import get_logger
from coro.core.models import TranscriptToken

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy.typing as npt

logger = get_logger(__name__)

_SAMPLE_RATE: Final = 16000
# Admission queue depth used when an adapter is built without explicit settings
# (direct construction in tests and tooling); the factory always passes one.
_DEFAULT_QUEUE_DEPTH = 32


class TimestampedRecognizer(Protocol):
    """The slice of an onnx-asr ``.with_timestamps()`` adapter this backend calls.

    ``recognize`` returns one ``TimestampedResult``, or an iterator of
    per-speech-segment results when the adapter wraps a VAD.
    """

    def recognize(
        self,
        waveform: npt.NDArray[np.float32],
        /,
        *,
        sample_rate: Literal[16000],
        language: str | None = ...,
    ) -> object:
        """Recognize one waveform."""
        ...


def convert_onnx_asr_result(
    result: object, *, offset_seconds: float = 0.0, span_end: float | None = None
) -> list[TranscriptToken]:
    """Convert an onnx-asr TimestampedResult into word-level TranscriptTokens.

    NeMo models (Parakeet/Canary) emit parallel ``tokens``/``timestamps`` lists that
    are grouped into words. onnx-asr's Whisper instead leaves those None and only
    fills ``text``; that case falls back to ``words_from_text`` (timings spread over
    ``[offset_seconds, span_end]``).

    Args:
        result: Object with ``tokens``/``timestamps``/``logprobs`` lists or a ``text``.
        offset_seconds: Timestamp offset added to each word's start/end.
        span_end: Absolute end of the result's audio span (used by the text fallback).

    Returns:
        List of TranscriptToken (one per reconstructed word). For the token path each
        word's ``start`` is its first subword's emission time and its ``end`` is the next
        word's start (final word padded by ``LAST_WORD_PAD``); ``probability`` is
        ``exp(mean(logprobs))`` or None.

    """
    tokens = getattr(result, "tokens", None)
    timestamps = getattr(result, "timestamps", None)
    logprobs = getattr(result, "logprobs", None)

    groups = group_subwords(tokens, timestamps, logprobs) if tokens and timestamps else []
    if groups:
        return words_from_subword_groups(groups, offset_seconds=offset_seconds)

    # Text-only result (e.g. onnx-asr Whisper): no token timestamps.
    text = (getattr(result, "text", "") or "").strip()
    return words_from_text(text, offset_seconds, span_end)


def convert_onnx_asr_segments(segments: Iterable[object]) -> list[TranscriptToken]:
    """Convert VAD-segmented onnx-asr results into absolute-timed TranscriptTokens.

    With VAD enabled, ``recognize`` yields one TimestampedSegmentResult per speech
    segment whose token timestamps are *relative to the segment start*. Each
    segment's ``start`` is the absolute offset, so tokens are re-based by it.
    """
    tokens: list[TranscriptToken] = []
    for seg in segments:
        offset = float(getattr(seg, "start", 0.0) or 0.0)
        seg_end = getattr(seg, "end", None)
        span_end = float(seg_end) if seg_end is not None else None
        tokens.extend(convert_onnx_asr_result(seg, offset_seconds=offset, span_end=span_end))
    return tokens


class OnnxAsrASRAdapter:
    """ASRAdapter that wraps an onnx-asr timestamped model.

    Adapter Concurrency Policy: **concurrent**. This adapter holds no lock.
    ``onnxruntime.InferenceSession.run`` is documented thread-safe, and every
    piece of onnx-asr state this adapter touches — the encoder/decoder sessions,
    the resampler's per-rate sessions, the vocabulary maps and the Silero VAD
    session — is built during ``load_model``/``load_vad`` and only read
    afterwards (Silero's LSTM state is a call-local variable, not instance
    state). Serialising here would therefore buy nothing and would cap
    throughput at one request.

    Load is bounded instead by an :class:`AdmissionController`, so total backend
    thread demand stays near the core count and overload is rejected with a
    retry hint rather than queued without limit.
    """

    honours_prompt: bool = False
    """A transducer has no text input port, so the carried prompt cannot reach it.

    This is architectural rather than a gap in the integration, and it was
    confirmed by measurement. The ASR window cache uses it to give this backend
    independent per-window keys, which is the best hit rate available: a missing
    window is purely local, recomputed without disturbing its neighbours.
    """

    def __init__(
        self,
        model: TimestampedRecognizer,
        *,
        vad_enabled: bool = False,
        admission: AdmissionController | None = None,
    ) -> None:
        self._model = model
        self._vad_enabled = vad_enabled
        self._admission = admission or build_admission_controller(
            max_concurrency=0, max_queue_depth=_DEFAULT_QUEUE_DEPTH
        )

    @property
    def admission(self) -> AdmissionController:
        """Admission controller implementing this adapter's concurrency policy."""
        return self._admission

    async def transcribe_pcm(
        self,
        pcm: bytes,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> list[TranscriptToken]:
        """Transcribe raw PCM s16le 16 kHz mono bytes.

        Note:
            ``prompt`` is accepted for protocol compatibility but ignored: onnx-asr's
            ``recognize`` has no ``initial_prompt`` equivalent, so cross-window prompt
            carry does not apply to this backend.

        Raises:
            AsrCapacityError: If the admission queue is full.

        """
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

        def _recognize() -> object:
            if language:
                return self._model.recognize(audio, sample_rate=_SAMPLE_RATE, language=language)
            return self._model.recognize(audio, sample_rate=_SAMPLE_RATE)

        async with self._admission.admit():
            result = await asyncio.to_thread(_recognize)
        if self._vad_enabled:
            # VAD adapter yields an iterator of per-speech-segment results.
            segments = cast("Iterable[object]", result)  # noqa: BTTOOL004 -- a VAD-wrapped onnx-asr adapter returns an iterator of segment results
            return convert_onnx_asr_segments(segments)
        # Non-VAD: a single result spanning the whole clip; pass its duration so the
        # text-only (Whisper) fallback can spread word timings across it.
        duration = len(audio) / _SAMPLE_RATE
        return convert_onnx_asr_result(result, span_end=duration)


def _providers_for_device(device: str) -> list[str] | None:
    """Map an ASR device selector to onnxruntime execution providers.

    Args:
        device: ``"auto"``, ``"cuda"`` or ``"cpu"``.

    Returns:
        A provider list, or None for ``"auto"`` (let onnxruntime choose its default).

    """
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return None


def build_onnx_asr_adapter(
    model_asr: str,
    *,
    device: str = "auto",
    quantization: str | None = None,
    providers: Sequence[str] | None = None,
    vad_enabled: bool = False,
    vad_threshold: float | None = None,
    max_concurrency: int = 0,
    max_queue_depth: int = _DEFAULT_QUEUE_DEPTH,
) -> OnnxAsrASRAdapter:
    """Construct and return an OnnxAsrASRAdapter.

    Args:
        model_asr: onnx-asr model name or HF repo id, e.g. ``"nemo-parakeet-tdt-0.6b-v3"``.
        device: Device selector (``"auto"``, ``"cuda"``, ``"cpu"``) used to derive providers
            when ``providers`` is not given explicitly.
        quantization: onnx-asr quantization selector, e.g. ``None`` or ``"int8"``.
        providers: Explicit onnxruntime providers; overrides ``device`` when supplied.
        vad_enabled: Wrap the model with Silero VAD speech segmentation
            (``onnx_asr.load_vad('silero')``). When True, ``recognize`` yields one
            result per detected speech segment.
        vad_threshold: Optional Silero VAD speech-probability threshold; only applied
            when ``vad_enabled`` is True. ``None`` keeps onnx-asr's default.
        max_concurrency: Adapter Concurrency Policy permit count; 0 auto-sizes
            from the host core count.
        max_queue_depth: Calls allowed to wait for a permit before rejection.

    Returns:
        Initialised adapter ready for use.

    """
    import onnx_asr

    resolved_providers = providers if providers is not None else _providers_for_device(device)
    session_options = build_asr_session_options()
    logger.info(
        "Loading onnx-asr model '%s' (quantization=%s, providers=%s, vad=%s, tuned sess_options).",
        model_asr,
        quantization,
        resolved_providers,
        vad_enabled,
    )
    model = onnx_asr.load_model(
        model_asr,
        quantization=quantization,
        sess_options=session_options,
        providers=resolved_providers,
    )
    if vad_enabled:
        vad = onnx_asr.load_vad(
            "silero", sess_options=session_options, providers=resolved_providers
        )
        segmented = (
            model.with_vad(vad)
            if vad_threshold is None
            else model.with_vad(vad, threshold=vad_threshold)
        )
        recognizer: TimestampedRecognizer = segmented.with_timestamps()
    else:
        recognizer = model.with_timestamps()
    logger.info("onnx-asr model loaded.")
    return OnnxAsrASRAdapter(
        recognizer,
        vad_enabled=vad_enabled,
        admission=build_admission_controller(
            max_concurrency=max_concurrency, max_queue_depth=max_queue_depth
        ),
    )
