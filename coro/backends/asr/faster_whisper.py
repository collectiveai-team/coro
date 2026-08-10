"""Faster Whisper ML Model Integration."""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from coro.backends.asr.concurrency import (
    AdmissionController,
    build_admission_controller,
    resolve_max_concurrency,
)
from coro.core.models import TranscriptToken

logger = logging.getLogger(__name__)

_NO_SPEECH_THRESHOLD = 0.9
# Admission queue depth used when an adapter is built without explicit settings
# (direct construction in tests and tooling); the factory always passes one.
_DEFAULT_QUEUE_DEPTH = 32


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
    """ASRAdapter that wraps a faster-whisper WhisperModel.

    Adapter Concurrency Policy: **concurrent**. This adapter holds no lock.
    faster-whisper documents calling ``WhisperModel.transcribe`` from multiple
    Python threads and exposes ``num_workers`` precisely for that case;
    ``transcribe`` itself mutates no instance state (``last_speech_timestamp`` is
    a call-local in the non-batched path), and the underlying CTranslate2
    ``Whisper`` model is thread-safe. The builder sizes ``num_workers`` to the
    resolved permit count so CTranslate2 actually runs those calls in parallel
    instead of queueing them behind one worker — without it, dropping the lock
    would only overlap the Python-side work.

    Load is bounded by an :class:`AdmissionController` so the product of permits
    and per-worker threads stays near the core count.
    """

    def __init__(self, model, *, admission: AdmissionController | None = None) -> None:
        self._model = model
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

        Raises:
            AsrCapacityError: If the admission queue is full.

        """
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

        def _transcribe() -> list:
            segments, _info = self._model.transcribe(
                audio,
                language=language,
                initial_prompt=prompt,
                word_timestamps=True,
            )
            return list(segments)

        async with self._admission.admit():
            segments = await asyncio.to_thread(_transcribe)
        return convert_asr_segments(segments)


def build_asr_adapter(
    model_asr: str,
    *,
    device: str = "auto",
    compute_type: str = "default",
    max_concurrency: int = 0,
    max_queue_depth: int = _DEFAULT_QUEUE_DEPTH,
) -> FasterWhisperASRAdapter:
    """Construct and return a FasterWhisperASRAdapter.

    Args:
        model_asr: Model identifier, e.g. ``"openai/whisper-medium"`` or
            ``"medium"``.
        device: Faster Whisper device selector, e.g. ``"auto"``, ``"cuda"``,
            or ``"cpu"``.
        compute_type: Faster Whisper compute type selector, e.g. ``"default"``,
            ``"float16"``, or ``"int8"``.
        max_concurrency: Adapter Concurrency Policy permit count; 0 auto-sizes
            from the host core count. Also becomes CTranslate2's ``num_workers``
            so admitted calls genuinely run in parallel. Model weights are shared
            across workers; only per-worker compute buffers are duplicated.
        max_queue_depth: Calls allowed to wait for a permit before rejection.

    Returns:
        Initialised adapter ready for use.

    """
    from faster_whisper import WhisperModel

    model_size = _model_size_from_id(model_asr)
    workers = resolve_max_concurrency(max_concurrency)
    logger.info(
        "Loading ASR model '%s' with faster-whisper size token '%s' on device '%s' "
        "(num_workers=%d).",
        model_asr,
        model_size,
        device,
        workers,
    )
    model = WhisperModel(model_size, device=device, compute_type=compute_type, num_workers=workers)
    logger.info("ASR model loaded.")
    return FasterWhisperASRAdapter(
        model,
        admission=build_admission_controller(
            max_concurrency=max_concurrency, max_queue_depth=max_queue_depth
        ),
    )
