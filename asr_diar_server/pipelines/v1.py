"""v1 Transcription Pipeline — full-memory pipeline.

Orchestrates: audio → full-memory PCM → ASR adapter → optional diarization
adapter → core response builder → WhisperX-Style Response dict.

The route handler passes audio bytes directly; this pipeline keeps the full
PCM in memory and processes it in one shot.  It is the counterpart to v2's
disk-backed chunked pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from asr_diar_server.audio import BYTES_PER_SAMPLE, SAMPLE_RATE, convert_to_pcm_bytes
from asr_diar_server.core.response import build_whisperx_response

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class V1Pipeline:
    """Full-memory Transcription Pipeline for API v1.

    Args:
        asr: ASR Adapter implementing ``transcribe_pcm(pcm, *, language, prompt)``.
        diarization: Optional Diarization Adapter implementing ``diarize_pcm(pcm)``.

    """

    def __init__(self, asr: Any, diarization: Any | None = None) -> None:
        self._asr = asr
        self._diarization = diarization

    async def run(
        self,
        audio_bytes: bytes,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> dict:
        """Transcribe audio and return a WhisperX-Style Response.

        Args:
            audio_bytes: Raw audio in any format supported by ffmpeg.
            language: Optional BCP-47 language hint.
            prompt: Optional initial prompt for the ASR model.

        Returns:
            WhisperX-Style Response dict.

        """
        pcm = await convert_to_pcm_bytes(audio_bytes)
        duration = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)

        tokens = await self._asr.transcribe_pcm(pcm, language=language, prompt=prompt)

        timeline = []
        if self._diarization is not None:
            timeline = await self._diarization.diarize_pcm(pcm)

        return build_whisperx_response(tokens=tokens, speaker_timeline=timeline, duration=duration)

    async def stream(
        self,
        audio_bytes: bytes,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ):
        """Stream transcription events as an async generator.

        Yields OpenAI-Exact SSE event dicts:
            - ``transcript.text.delta`` for incremental text.
            - ``transcript.text.done`` carrying the final JSON result.

        Args:
            audio_bytes: Raw audio in any format supported by ffmpeg.
            language: Optional BCP-47 language hint.
            prompt: Optional initial prompt for the ASR model.

        """
        pcm = await convert_to_pcm_bytes(audio_bytes)
        duration = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)

        # For v1 the full transcription must complete before streaming starts.
        # We emit a single delta with all text then a done event.
        tokens = await self._asr.transcribe_pcm(pcm, language=language, prompt=prompt)
        timeline = []
        if self._diarization is not None:
            timeline = await self._diarization.diarize_pcm(pcm)

        result = build_whisperx_response(
            tokens=tokens, speaker_timeline=timeline, duration=duration
        )

        # Emit one delta covering the full transcript text.
        full_text = " ".join(s.get("text", "") for s in result.get("segments", []))
        if full_text:
            yield {"type": "transcript.text.delta", "delta": full_text}
            await asyncio.sleep(0)  # yield control

        yield {"type": "transcript.text.done", "text": json.dumps(result)}
