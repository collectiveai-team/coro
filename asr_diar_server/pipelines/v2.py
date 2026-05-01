"""v2 Transcription Pipeline — disk-backed chunked pipeline.

Orchestrates: file path → PCM chunk stream → per-window ASR adapter →
optional diarization adapter → core response builder → WhisperX-Style Response.

The v2 pipeline reads audio from a temp file via ffmpeg streaming so the
full audio PCM is never loaded into memory simultaneously.  Windows of 30s
(with 2s overlap) are processed sequentially.

The upload temp file is owned by the API router; this pipeline does not
delete it (the router is responsible for cleanup).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from asr_diar_server.audio import BYTES_PER_SAMPLE, SAMPLE_RATE, stream_pcm_from_file
from asr_diar_server.core.response import build_whisperx_response
from asr_diar_server.core.types import SpeakerSegment, TranscriptToken

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_WINDOW_SECONDS = 30
_OVERLAP_SECONDS = 2
_WINDOW_BYTES = _WINDOW_SECONDS * SAMPLE_RATE * BYTES_PER_SAMPLE
_OVERLAP_BYTES = _OVERLAP_SECONDS * SAMPLE_RATE * BYTES_PER_SAMPLE


class V2Pipeline:
    """Disk-backed chunked Transcription Pipeline for API v2.

    Args:
        asr: ASR Adapter implementing ``transcribe_pcm(pcm, *, language, prompt)``.
        diarization: Optional Diarization Adapter implementing ``diarize_pcm(pcm)``.

    """

    def __init__(self, asr: Any, diarization: Any | None = None) -> None:
        self._asr = asr
        self._diarization = diarization

    async def run_from_path(
        self,
        path: str,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> dict:
        """Transcribe from a file path using the disk-backed pipeline.

        Args:
            path: Filesystem path to the audio file.
            language: Optional BCP-47 language hint.
            prompt: Optional initial prompt propagated across ASR windows.

        Returns:
            WhisperX-Style Response dict.

        """
        all_tokens: list[TranscriptToken] = []
        all_diarization: list[SpeakerSegment] = []
        processed_bytes = 0
        asr_buffer = b""
        asr_offset = 0.0
        init_prompt = prompt or ""
        accepted_until = 0.0

        async for pcm_chunk in stream_pcm_from_file(path, chunk_seconds=1.0):
            processed_bytes += len(pcm_chunk)

            # Feed diarization per 1-second chunk if enabled.
            if self._diarization is not None:
                new_diar = await self._diarization.diarize_pcm(pcm_chunk)
                all_diarization.extend(new_diar)

            asr_buffer += pcm_chunk

            while len(asr_buffer) >= _WINDOW_BYTES:
                window = asr_buffer[:_WINDOW_BYTES]
                offset_seconds = asr_offset
                new_tokens = await self._asr.transcribe_pcm(
                    window, language=language, prompt=init_prompt
                )
                # Deduplicate overlap region.
                overlap_end = offset_seconds + _OVERLAP_SECONDS
                accepted = []
                for token in new_tokens:
                    in_overlap = accepted_until != 0.0 and token.start < overlap_end
                    if in_overlap and any(
                        abs(token.start - t.start) <= 0.25 for t in all_tokens[-100:]
                    ):
                        continue
                    accepted.append(token)
                if accepted:
                    all_tokens.extend(accepted)
                    recent = "".join(t.text for t in all_tokens[-50:])
                    init_prompt = recent[-200:]
                accepted_until = offset_seconds + max(0.0, _WINDOW_SECONDS - _OVERLAP_SECONDS)
                asr_offset += _WINDOW_SECONDS - _OVERLAP_SECONDS
                asr_buffer = asr_buffer[_WINDOW_BYTES - _OVERLAP_BYTES :]

        # Process remaining buffer.
        if asr_buffer:
            new_tokens = await self._asr.transcribe_pcm(
                asr_buffer, language=language, prompt=init_prompt
            )
            all_tokens.extend(new_tokens)

        duration = processed_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        all_tokens.sort(key=lambda t: t.start)
        return build_whisperx_response(
            tokens=all_tokens, speaker_timeline=all_diarization, duration=duration
        )

    async def stream_from_path(
        self,
        path: str,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ):
        """Stream transcription events from a file path.

        Yields OpenAI-Exact SSE event dicts with no package-specific progress events.

        Args:
            path: Filesystem path to the audio file.
            language: Optional BCP-47 language hint.
            prompt: Optional initial prompt for the ASR model.

        """
        result = await self.run_from_path(path, language=language, prompt=prompt)
        full_text = " ".join(s.get("text", "") for s in result.get("segments", []))
        if full_text:
            yield {"type": "transcript.text.delta", "delta": full_text}
            await asyncio.sleep(0)
        yield {"type": "transcript.text.done", "text": json.dumps(result)}
