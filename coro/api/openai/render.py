"""Incremental renderers for the OpenAI-shaped response formats.

Each renderer builds its response model with empty arrays and a sentinel text,
serialises that envelope, and splices the arrays in one element at a time from a
**Transcript Source**. The models therefore still define the contract — key
order, key names and scalar formatting all come from them — while no array is
ever fully resident (ADR 0018).

The per-element models are the same ones the materialised projection used, so
the elements are identical by construction; only the envelope assembly is new,
and ``tests/test_openai_formats_unchanged.py`` freezes its bytes.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

from coro.api.exceptions import TranscriptionValidationError
from coro.api.json_body import (
    TEXT_SENTINEL,
    array_slot,
    dump_model,
    splice,
    text_slot,
)
from coro.api.openai.formats import ResponseFormat
from coro.api.openai.schemas import (
    DiarizedJsonResponse,
    DiarizedJsonSegment,
    JsonResponse,
    TranscriptionUsage,
    VerboseJsonResponse,
    VerboseJsonSegment,
    VerboseJsonWord,
)
from coro.core.transcript_source import (
    TranscriptSource,
    iter_text_fragments,
    response_duration,
)


def _usage(duration: float) -> TranscriptionUsage:
    return TranscriptionUsage(type="duration", seconds=math.ceil(duration))


def _timed_words(source: TranscriptSource) -> Iterator:
    """Yield the words a verbose response reports, with the raw-word fallback.

    The materialised projection read ``word_segments or raw_words``, falling back
    only when the per-word view is empty. Peeking one item answers "is it empty"
    without materialising either.
    """
    words = source.iter_words()
    first = next(words, None)
    if first is None:
        yield from source.iter_raw_words()
        return
    yield first
    yield from words


def render_json(source: TranscriptSource) -> Iterator[str]:
    """Render the default ``json`` body: the transcript text and its usage."""
    envelope = dump_model(JsonResponse(text=TEXT_SENTINEL, usage=_usage(response_duration(source))))
    yield from splice(envelope, [text_slot(iter_text_fragments(source))])


def render_verbose_json(source: TranscriptSource, *, language: str | None) -> Iterator[str]:
    """Render the ``verbose_json`` body: text, segments and words."""
    duration = response_duration(source)
    envelope = dump_model(
        VerboseJsonResponse(
            duration=duration,
            language=language or "unknown",
            text=TEXT_SENTINEL,
            segments=[],
            words=[],
            usage=_usage(duration),
        )
    )
    segments = (
        dump_model(
            VerboseJsonSegment(
                id=index,
                seek=int(segment.start * 100),
                start=segment.start,
                end=segment.end,
                text=segment.text,
                tokens=[],
                temperature=0.0,
                avg_logprob=0.0,
                compression_ratio=0.0,
                no_speech_prob=0.0,
            )
        )
        for index, segment in enumerate(source.iter_segments())
    )
    words = (
        dump_model(VerboseJsonWord(word=word.word, start=word.start, end=word.end))
        for word in _timed_words(source)
    )
    yield from splice(
        envelope,
        [
            text_slot(iter_text_fragments(source)),
            array_slot("segments", segments),
            array_slot("words", words),
        ],
    )


def render_diarized_json(source: TranscriptSource) -> Iterator[str]:
    """Render the ``diarized_json`` body: text and speaker-annotated segments."""
    duration = response_duration(source)
    envelope = dump_model(
        DiarizedJsonResponse(
            task="transcribe",
            duration=duration,
            text=TEXT_SENTINEL,
            segments=[],
            usage=_usage(duration),
        )
    )
    segments = (
        dump_model(
            DiarizedJsonSegment(
                type="transcript.text.segment",
                id=f"seg_{index + 1:03d}",
                start=segment.start,
                end=segment.end,
                text=segment.text,
                speaker=segment.speaker,
            )
        )
        for index, segment in enumerate(source.iter_segments())
    )
    yield from splice(
        envelope, [text_slot(iter_text_fragments(source)), array_slot("segments", segments)]
    )


def render_for_format(
    response_format: ResponseFormat,
    source: TranscriptSource,
    *,
    language: str | None,
) -> Iterator[str]:
    """Render a transcription in one of the supported response formats.

    Public because the offline command renders through it too: ``coro run`` must
    produce the same body the endpoint would for the same format, or its output
    would silently be a fourth shape nobody documented.

    Args:
        response_format: The requested format.
        source: The transcription's Transcript Source.
        language: Language to report, for the formats that carry one.

    Returns:
        The response body as an iterator of JSON fragments.

    Raises:
        TranscriptionValidationError: If the format is recognised but unsupported.

    """
    match response_format:
        case ResponseFormat.JSON:
            return render_json(source)
        case ResponseFormat.VERBOSE_JSON:
            return render_verbose_json(source, language=language)
        case ResponseFormat.DIARIZED_JSON:
            return render_diarized_json(source)

    raise TranscriptionValidationError(
        f"Unsupported response_format '{response_format}'.",
        param="response_format",
    )
