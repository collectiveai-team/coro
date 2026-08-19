"""Obtaining a Transcript Source from whatever pipeline is configured.

A pipeline that can hand back a lazy transcript exposes ``transcribe_source``;
anything else is run through ``transcribe`` and wrapped. The distinction is
*where the transcript lives*, never how it is rendered — both arms feed the same
response projection, so the two pipelines cannot drift apart in output (ADR
0018).

The fallback is what keeps this open: a pipeline implementing only the batch
protocol, including every test double, still works and simply is not flat.
"""

from __future__ import annotations

from typing import Any

from coro.audio import AudioInput
from coro.core.transcript_source import MemoryTranscriptSource, TranscriptSource


async def transcript_source(
    pipeline: Any,
    audio: AudioInput,
    *,
    language: str | None = None,
    prompt: str | None = None,
) -> TranscriptSource:
    """Run one transcription and return a Transcript Source over its transcript.

    Args:
        pipeline: The Configured Transcription Pipeline.
        audio: The Audio Input to transcribe. The pipeline owns its cleanup.
        language: Optional language hint.
        prompt: Optional initial prompt.

    Returns:
        A Transcript Source the caller must ``close()``. It is backed by the
        Transcript Spill Store when the pipeline offers one, and by an
        in-memory result otherwise.

    """
    build_source = getattr(pipeline, "transcribe_source", None)
    if build_source is not None:
        return await build_source(audio, language=language, prompt=prompt)
    result = await pipeline.transcribe(audio, language=language, prompt=prompt)
    return MemoryTranscriptSource(result)
