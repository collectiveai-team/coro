"""whisperlivekit ML Model Integration.

Provides pure conversion functions that translate whisperlivekit-native ASR
and diarization objects into Project-Owned Transcript Model types, plus
concrete adapter classes that satisfy the ASRAdapter and DiarizationAdapter
protocols.

Conversion functions are pure (no I/O, no model calls) so they can be
tested in isolation with fake SimpleNamespace objects.

Public surface:
    convert_asr_segments        — native segment/word list → list[TranscriptToken]
    convert_diarization_segments — native diar list → list[SpeakerSegment]
    WhisperLiveKitASRAdapter    — ASRAdapter wrapping TranscriptionEngine.asr
    WhisperLiveKitDiarizationAdapter — DiarizationAdapter wrapping diarization_model
    build_asr_adapter           — factory: settings → WhisperLiveKitASRAdapter
    build_diarization_adapter   — factory: settings → WhisperLiveKitDiarizationAdapter | None
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading

import numpy as np

from asr_diar_server.core.types import SpeakerSegment, TranscriptToken

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_BYTES_PER_SAMPLE = 2  # int16

_NO_SPEECH_THRESHOLD = 0.9


def convert_asr_segments(
    native_segments,
    *,
    offset_seconds: float = 0.0,
) -> list[TranscriptToken]:
    """Convert faster-whisper/whisperlivekit segment objects to TranscriptTokens.

    Args:
        native_segments: Iterable of native segment objects with ``.words`` and
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


def _speaker_to_one_indexed(speaker) -> int:
    """Convert a zero-indexed or string speaker label to 1-indexed int.

    Args:
        speaker: int (0-indexed) or string like "SPEAKER_00" or "0".

    Returns:
        1-indexed integer speaker label.

    """
    if isinstance(speaker, int):
        return speaker + 1
    # numpy integer types
    try:
        import numpy as np

        if isinstance(speaker, np.integer):
            return int(speaker) + 1
    except ImportError:
        pass
    # String: extract first integer and add 1
    match = re.search(r"\d+", str(speaker))
    if match:
        return int(match.group(0)) + 1
    return 1


def convert_diarization_segments(
    native_segments,
    *,
    duration: float,
) -> list[SpeakerSegment]:
    """Convert whisperlivekit diarization segment objects to SpeakerSegments.

    Args:
        native_segments: Iterable of native diarization segment objects with
            ``.start``, ``.end``, and ``.speaker`` attributes.
        duration: Total audio duration in seconds; end times are clamped.

    Returns:
        Deduplicated list of SpeakerSegment sorted by start time.

    """
    timeline: list[SpeakerSegment] = []
    seen: set = set()

    for seg in native_segments:
        start = max(0.0, float(getattr(seg, "start", 0.0) or 0.0))
        end = min(duration, float(getattr(seg, "end", 0.0) or 0.0))
        if end <= start:
            continue
        speaker = _speaker_to_one_indexed(getattr(seg, "speaker", 0))
        key = (round(start, 3), round(end, 3), speaker)
        if key in seen:
            continue
        seen.add(key)
        timeline.append(SpeakerSegment(start=round(start, 3), end=round(end, 3), speaker=speaker))

    timeline.sort(key=lambda s: s.start)
    return timeline


# ---------------------------------------------------------------------------
# Adapter classes
# ---------------------------------------------------------------------------

def _model_size_from_id(model_id: str) -> str:
    """Extract the model size token from a HuggingFace-style model id.

    ``"openai/whisper-medium"`` → ``"medium"``
    ``"medium"``               → ``"medium"``
    """
    base = model_id.split("/")[-1]
    # Strip "whisper-" prefix if present
    if base.startswith("whisper-"):
        base = base[len("whisper-"):]
    return base


class WhisperLiveKitASRAdapter:
    """ASRAdapter that wraps a whisperlivekit ASR backend object.

    The ``asr`` object is obtained from ``TranscriptionEngine.asr`` and
    exposes ``transcribe(audio_np, init_prompt) -> list[Segment]``.

    A per-instance lock serialises calls so multiple concurrent requests
    don't corrupt shared model state (GPU is single-threaded anyway).
    """

    def __init__(self, asr) -> None:
        self._asr = asr
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

        # Override ASR language for this call when a language hint is given.
        asr = self._asr
        original_language = getattr(asr, "original_language", None)
        if language:
            try:
                asr.original_language = language
            except Exception:
                pass

        def _transcribe() -> list:
            with self._lock:
                return asr.transcribe(audio, init_prompt=prompt or "")

        try:
            segments = await asyncio.to_thread(_transcribe)
        finally:
            if language and original_language is not None:
                try:
                    asr.original_language = original_language
                except Exception:
                    pass

        return convert_asr_segments(segments)


class WhisperLiveKitDiarizationAdapter:
    """DiarizationAdapter that wraps a SortformerDiarization shared model.

    Mirrors the batch diarization pattern from ``custom_server._batch_diarize``:
    creates a fresh ``SortformerDiarizationOnline`` per call, feeds 1-second
    chunks, flushes with a silent chunk, then collects segments.
    """

    _CHUNK_SECONDS = 1.0

    def __init__(self, diarization_model) -> None:
        self._shared_model = diarization_model

    async def diarize_pcm(self, pcm: bytes) -> list[SpeakerSegment]:
        """Run batch diarization over full PCM audio."""
        return await asyncio.to_thread(self._diarize_sync, pcm)

    def _diarize_sync(self, pcm: bytes) -> list[SpeakerSegment]:
        from argparse import Namespace

        from whisperlivekit.core import online_diarization_factory

        chunk_samples = int(_SAMPLE_RATE * self._CHUNK_SECONDS)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        duration = len(audio) / _SAMPLE_RATE

        args = Namespace(diarization_backend="sortformer")
        online_diarization = online_diarization_factory(args, self._shared_model)
        raw_segments: list = []

        import asyncio as _asyncio

        loop = _asyncio.new_event_loop()
        try:
            for start in range(0, len(audio), chunk_samples):
                online_diarization.insert_audio_chunk(audio[start : start + chunk_samples])
                new_segs = loop.run_until_complete(online_diarization.diarize())
                if new_segs:
                    raw_segments.extend(new_segs)

            # Flush
            online_diarization.insert_audio_chunk(np.zeros(chunk_samples, dtype=np.float32))
            new_segs = loop.run_until_complete(online_diarization.diarize())
            if new_segs:
                raw_segments.extend(new_segs)
        finally:
            loop.close()
            close = getattr(online_diarization, "close", None)
            if close:
                close()

        return convert_diarization_segments(raw_segments, duration=duration)


# ---------------------------------------------------------------------------
# Adapter factories
# ---------------------------------------------------------------------------

def build_asr_adapter(model_asr: str) -> WhisperLiveKitASRAdapter:
    """Construct and return a WhisperLiveKitASRAdapter.

    Initialises ``TranscriptionEngine`` with the given model, then wraps
    ``engine.asr`` in a ``WhisperLiveKitASRAdapter``.

    Args:
        model_asr: Model identifier, e.g. ``"openai/whisper-medium"`` or
            ``"medium"``.

    Returns:
        Initialised adapter ready for use.

    """
    from whisperlivekit import TranscriptionEngine
    from whisperlivekit.config import WhisperLiveKitConfig

    model_size = _model_size_from_id(model_asr)
    logger.info("Loading ASR model '%s' (size token: '%s') …", model_asr, model_size)
    config = WhisperLiveKitConfig(
        model_size=model_size,
        transcription=True,
        diarization=False,
        # Force LocalAgreement policy so engine.asr is a FasterWhisperASR
        # (or equivalent batch-capable backend) with a working transcribe()
        # method.  The default SimulStreamingASR.transcribe() is a no-op
        # designed for streaming and cannot be used for batch transcription.
        backend_policy="localagreement",
    )
    engine = TranscriptionEngine(config=config)
    logger.info("ASR model loaded.")
    return WhisperLiveKitASRAdapter(engine.asr)


def build_diarization_adapter(model_diarization: str) -> WhisperLiveKitDiarizationAdapter:
    """Construct and return a WhisperLiveKitDiarizationAdapter.

    Args:
        model_diarization: HuggingFace model id for Sortformer, e.g.
            ``"nvidia/diar_sortformer_4spk-v1"``.

    Returns:
        Initialised adapter ready for use.

    """
    from whisperlivekit.diarization.sortformer_backend import SortformerDiarization

    logger.info("Loading diarization model '%s' …", model_diarization)
    shared_model = SortformerDiarization(model_name=model_diarization)
    logger.info("Diarization model loaded.")
    return WhisperLiveKitDiarizationAdapter(shared_model)
