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
import logging
import math
import threading

import numpy as np

from coro.core.models import TranscriptToken

logger = logging.getLogger(__name__)

# SentencePiece word-start marker used by NeMo models (Parakeet/Canary/GigaAM).
_SP_SPACE = "\u2581"
# Synthesised duration (seconds) for the final word, which has no following token.
_LAST_WORD_PAD = 0.2
_SAMPLE_RATE = 16000


def _group_subwords(
    tokens: list[str],
    timestamps: list[float],
    logprobs: list[float] | None,
) -> list[dict]:
    """Group subword tokens into words on the word-start marker.

    A token starts a new word when it is prefixed by a space or the SentencePiece
    ``\u2581`` marker; otherwise it continues the current word (this also keeps
    leading punctuation tokens, e.g. ``","``, attached to the preceding word).

    Args:
        tokens: Subword token strings (a leading space or ``\u2581`` prefixes a new word).
        timestamps: One emission time per token, parallel to ``tokens``.
        logprobs: One log-probability per token, parallel to ``tokens``, or None.

    Returns:
        List of ``{"text": str, "start": float, "logprobs": list[float]}`` word groups,
        where ``text`` keeps the leading space of a word start so that
        ``"".join(token.text ...)`` in the response builder reconstructs spaced text.

    """
    groups: list[dict] = []
    n = len(tokens)
    for i in range(n):
        token = tokens[i]
        if not token:
            continue
        ts = float(timestamps[i])
        lp = float(logprobs[i]) if logprobs is not None and i < len(logprobs) else None
        is_word_start = token.startswith((" ", _SP_SPACE))
        piece = token.replace(_SP_SPACE, " ")
        if is_word_start or not groups:
            groups.append({"text": piece, "start": ts, "logprobs": []})
        else:
            groups[-1]["text"] += piece
        if lp is not None:
            groups[-1]["logprobs"].append(lp)
    return groups


def _words_from_text(text: str, start: float, span_end: float | None) -> list[TranscriptToken]:
    """Synthesise word-level tokens from a text-only result (no token timestamps).

    onnx-asr's Whisper exposes ``text`` but leaves ``tokens``/``timestamps`` None, so
    word timings are spread evenly across ``[start, span_end]`` (the VAD segment span,
    or the whole clip). Each word keeps a leading space so the response builder's
    ``"".join(...)`` reconstructs spaced text.
    """
    words = text.split()
    if not words:
        return []
    step = (span_end - start) / len(words) if span_end and span_end > start else _LAST_WORD_PAD
    out: list[TranscriptToken] = []
    for i, word in enumerate(words):
        word_start = start + i * step
        word_end = start + (i + 1) * step
        out.append(
            TranscriptToken(
                start=round(word_start, 3),
                end=round(max(word_end, word_start), 3),
                text=" " + word,
                probability=None,
            )
        )
    return out


def convert_onnx_asr_result(
    result, *, offset_seconds: float = 0.0, span_end: float | None = None
) -> list[TranscriptToken]:
    """Convert an onnx-asr TimestampedResult into word-level TranscriptTokens.

    NeMo models (Parakeet/Canary) emit parallel ``tokens``/``timestamps`` lists that
    are grouped into words. onnx-asr's Whisper instead leaves those None and only
    fills ``text``; that case falls back to ``_words_from_text`` (timings spread over
    ``[offset_seconds, span_end]``).

    Args:
        result: Object with ``tokens``/``timestamps``/``logprobs`` lists or a ``text``.
        offset_seconds: Timestamp offset added to each word's start/end.
        span_end: Absolute end of the result's audio span (used by the text fallback).

    Returns:
        List of TranscriptToken (one per reconstructed word). For the token path each
        word's ``start`` is its first subword's emission time and its ``end`` is the next
        word's start (final word padded by ``_LAST_WORD_PAD``); ``probability`` is
        ``exp(mean(logprobs))`` or None.

    """
    tokens = getattr(result, "tokens", None)
    timestamps = getattr(result, "timestamps", None)
    logprobs = getattr(result, "logprobs", None)

    if tokens and timestamps:
        groups = _group_subwords(tokens, timestamps, logprobs)
        if groups:
            out: list[TranscriptToken] = []
            for i, group in enumerate(groups):
                start = group["start"] + offset_seconds
                if i + 1 < len(groups):
                    end = groups[i + 1]["start"] + offset_seconds
                else:
                    end = group["start"] + _LAST_WORD_PAD + offset_seconds
                end = max(end, start)

                word_logprobs = group["logprobs"]
                probability = (
                    math.exp(sum(word_logprobs) / len(word_logprobs)) if word_logprobs else None
                )

                out.append(
                    TranscriptToken(
                        start=round(start, 3),
                        end=round(end, 3),
                        text=group["text"],
                        probability=probability,
                    )
                )
            return out

    # Text-only result (e.g. onnx-asr Whisper): no token timestamps.
    text = (getattr(result, "text", "") or "").strip()
    return _words_from_text(text, offset_seconds, span_end)


def convert_onnx_asr_segments(segments) -> list[TranscriptToken]:
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
    """ASRAdapter that wraps an onnx-asr timestamped model."""

    def __init__(self, model, *, vad_enabled: bool = False) -> None:
        self._model = model
        self._lock = threading.Lock()
        self._vad_enabled = vad_enabled

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

        """
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

        def _recognize():
            with self._lock:
                kwargs: dict = {"sample_rate": _SAMPLE_RATE}
                if language:
                    kwargs["language"] = language
                return self._model.recognize(audio, **kwargs)

        result = await asyncio.to_thread(_recognize)
        if self._vad_enabled:
            # VAD adapter yields an iterator of per-speech-segment results.
            return convert_onnx_asr_segments(result)
        # Non-VAD: a single result spanning the whole clip; pass its duration so the
        # text-only (Whisper) fallback can spread word timings across it.
        duration = len(audio) / _SAMPLE_RATE
        return convert_onnx_asr_result(result, span_end=duration)


def _providers_for_device(device: str):
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
    providers=None,
    vad_enabled: bool = False,
    vad_threshold: float | None = None,
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

    Returns:
        Initialised adapter ready for use.

    """
    import onnx_asr

    resolved_providers = providers if providers is not None else _providers_for_device(device)
    logger.info(
        "Loading onnx-asr model '%s' (quantization=%s, providers=%s, vad=%s).",
        model_asr,
        quantization,
        resolved_providers,
        vad_enabled,
    )
    model = onnx_asr.load_model(
        model_asr,
        quantization=quantization,
        providers=resolved_providers,
    )
    if vad_enabled:
        vad = onnx_asr.load_vad("silero", providers=resolved_providers)
        vad_options: dict = {}
        if vad_threshold is not None:
            vad_options["threshold"] = vad_threshold
        model = model.with_vad(vad, **vad_options).with_timestamps()
    else:
        model = model.with_timestamps()
    logger.info("onnx-asr model loaded.")
    return OnnxAsrASRAdapter(model, vad_enabled=vad_enabled)
