"""Faster Whisper ML Model Integration."""

from __future__ import annotations

import asyncio
import logging
import threading

import numpy as np

from asr_diar_server.core.types import TranscriptToken

logger = logging.getLogger(__name__)

_NO_SPEECH_THRESHOLD = 0.9


def convert_asr_segments(
    native_segments,
    *,
    offset_seconds: float = 0.0,
) -> list[TranscriptToken]:
    """Convert faster-whisper segment objects to TranscriptTokens.

    Args:
        native_segments: Iterable of segment objects with ``.words`` and
            ``.no_speech_prob`` attributes.
        offset_seconds: Timestamp offset to add to each word's start/end.

    Returns:
        List of TranscriptToken sorted by start time.

    """
    tokens: list[TranscriptToken] = []

    for seg in native_segments:
        if getattr(seg, "no_speech_prob", 0.0) > _NO_SPEECH_THRESHOLD:
            continue
        for word in getattr(seg, "words", []):
            start = round(float(getattr(word, "start", 0.0)) + offset_seconds, 3)
            end = round(float(getattr(word, "end", 0.0)) + offset_seconds, 3)
            text = getattr(word, "word", getattr(word, "text", ""))
            probability = getattr(word, "probability", None)
            tokens.append(TranscriptToken(start=start, end=end, text=text, probability=probability))

    return tokens


def _model_size_from_id(model_id: str) -> str:
    """Extract the Faster Whisper model size token from a model id."""
    base = model_id.split("/")[-1]
    if base.startswith("whisper-"):
        base = base[len("whisper-") :]
    return base


class FasterWhisperASRAdapter:
    """ASRAdapter that wraps a faster-whisper WhisperModel."""

    def __init__(self, model) -> None:
        self._model = model
        self._lock = threading.Lock()

    async def transcribe_pcm(
        self,
        pcm: bytes,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> list[TranscriptToken]:
        """Transcribe raw PCM s16le 16 kHz mono bytes."""
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

        def _transcribe() -> list:
            with self._lock:
                segments, _info = self._model.transcribe(
                    audio,
                    language=language,
                    initial_prompt=prompt,
                    word_timestamps=True,
                )
                return list(segments)

        segments = await asyncio.to_thread(_transcribe)
        return convert_asr_segments(segments)


def build_asr_adapter(
    model_asr: str,
    *,
    device: str = "auto",
    compute_type: str = "default",
) -> FasterWhisperASRAdapter:
    """Construct and return a FasterWhisperASRAdapter.

    Args:
        model_asr: Model identifier, e.g. ``"openai/whisper-medium"`` or
            ``"medium"``.
        device: Faster Whisper device selector, e.g. ``"auto"``, ``"cuda"``,
            or ``"cpu"``.
        compute_type: Faster Whisper compute type selector, e.g. ``"default"``,
            ``"float16"``, or ``"int8"``.

    Returns:
        Initialised adapter ready for use.

    """
    from faster_whisper import WhisperModel

    model_size = _model_size_from_id(model_asr)
    logger.info(
        "Loading ASR model '%s' with faster-whisper size token '%s' on device '%s'.",
        model_asr,
        model_size,
        device,
    )
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    logger.info("ASR model loaded.")
    return FasterWhisperASRAdapter(model)
