"""Skipping diarization for requests whose response carries no speaker.

Diarization costs about a quarter of Transcription Throughput (docs/benchmark.md),
and its result only reaches the wire through speaker fields. A request whose
response renders none of them gets the same body without it, so the endpoints
ask for a pipeline without diarization in that case.

Which responses need speakers is decided by each endpoint, because it depends
on the vendor contract: speakers appear in OpenAI's ``diarized_json`` and in
every SSE done frame, and in Deepgram's ``diarize`` words and ``utterances``
turns (turns are split by speaker even when the labels are hidden).
"""

from __future__ import annotations

from coro.core.protocols import TranscriptionPipeline


def pipeline_for_speakers(
    pipeline: TranscriptionPipeline, *, needs_speakers: bool
) -> TranscriptionPipeline:
    """Return ``pipeline``, or a copy that skips diarization when no speaker is rendered.

    A pipeline that cannot drop its diarization is returned unchanged: the
    response is the same, only slower.
    """
    if needs_speakers:
        return pipeline
    without_diarization = getattr(pipeline, "without_diarization", None)
    if without_diarization is None:
        return pipeline
    return without_diarization()
