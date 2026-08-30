"""Incremental renderer for the Deepgram-shaped pre-recorded response.

Same construction as the OpenAI renderers: the envelope comes from
``DeepgramResponse`` itself, built with empty arrays and a sentinel transcript,
and the arrays are spliced in one element at a time from a **Transcript Source**
(ADR 0018).

Two things make this shape harder than the OpenAI ones. The arrays are *nested*
— ``results.channels[0].alternatives[0].words`` — so the envelope is built with a
real channel and alternative carrying no words rather than with an empty
``channels`` list. And ``alternatives[0].confidence`` is the mean over the very
words it precedes, so it is computed by its own pass before the envelope is
serialised, exactly as the duration is.

``exclude_none`` is preserved: an undiarized response carries no ``speaker`` keys
rather than null ones, and ``utterances`` is absent unless requested, which is
what Deepgram itself emits.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from coro.api.deepgram.schemas import (
    MONO_CHANNEL_COUNT,
    PRIMARY_CHANNEL,
    DeepgramAlternative,
    DeepgramChannel,
    DeepgramMetadata,
    DeepgramResponse,
    DeepgramResults,
    DeepgramUtterance,
    deepgram_word,
    speaker_label,
)
from coro.api.json_body import (
    TEXT_SENTINEL,
    array_slot,
    dump_model,
    splice,
    text_slot,
)
from coro.api.utterances import group_words_into_utterances, mean_measured
from coro.core.transcript_source import (
    TranscriptSource,
    iter_text_fragments,
    response_duration,
)


def _mean_word_confidence(source: TranscriptSource) -> float | None:
    """Mean of the words' measured confidences, over its own pass."""
    return mean_measured(word.score for word in source.iter_words())


def _iter_utterances(source: TranscriptSource, *, diarize: bool) -> Iterator[str]:
    """Yield rendered utterances, buffering one speaker turn at a time.

    ``group_words_into_utterances`` groups a materialised sequence; feeding it
    the whole transcript would defeat the point, so runs are closed here as the
    speaker changes and each is grouped on its own. The result is identical
    because a run is complete the moment a different speaker appears.
    """
    run: list[Any] = []
    for word in source.iter_words():
        if run and run[-1].speaker != word.speaker:
            yield from _render_run(run, diarize=diarize)
            run = []
        run.append(word)
    if run:
        yield from _render_run(run, diarize=diarize)


def _render_run(run: list[Any], *, diarize: bool) -> Iterator[str]:
    """Render one same-speaker run through the shared grouping helper."""
    for utterance in group_words_into_utterances(run):
        yield dump_model(
            DeepgramUtterance(
                start=utterance.start,
                end=utterance.end,
                confidence=utterance.confidence,
                channel=PRIMARY_CHANNEL,
                transcript=utterance.text,
                words=[deepgram_word(word, diarize=diarize) for word in utterance.words],
                speaker=speaker_label(utterance.speaker) if diarize else None,
            ),
            exclude_none=True,
        )


def render_deepgram(
    source: TranscriptSource,
    *,
    request_id: str,
    audio_sha256: str,
    created: str,
    asr_model: str,
    asr_backend: str,
    diarize: bool,
    utterances: bool,
) -> Iterator[str]:
    """Render the Deepgram pre-recorded body from a Transcript Source."""
    duration = response_duration(source)
    envelope_response = DeepgramResponse(
        metadata=DeepgramMetadata(
            request_id=request_id,
            sha256=audio_sha256,
            created=created,
            duration=duration,
            channels=MONO_CHANNEL_COUNT,
            models=[asr_model],
            model_info={asr_model: {"name": asr_model, "arch": asr_backend}},
        ),
        results=DeepgramResults(
            channels=[
                DeepgramChannel(
                    alternatives=[
                        DeepgramAlternative(
                            transcript=TEXT_SENTINEL,
                            confidence=_mean_word_confidence(source),
                            words=[],
                        )
                    ]
                )
            ],
            utterances=[] if utterances else None,
        ),
    )
    envelope = dump_model(envelope_response, exclude_none=True)

    words = (
        dump_model(deepgram_word(word, diarize=diarize), exclude_none=True)
        for word in source.iter_words()
    )
    slots = [text_slot(iter_text_fragments(source)), array_slot("words", words)]
    if utterances:
        slots.append(array_slot("utterances", _iter_utterances(source, diarize=diarize)))
    yield from splice(envelope, slots)
