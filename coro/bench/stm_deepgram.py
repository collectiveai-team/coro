"""Adapt a Deepgram-shaped response onto the per-word STM builder.

Deepgram nests its words at ``results.channels[].alternatives[].words[]``,
where :func:`coro.bench.stm.hyp_response_to_stm` looks only at the top-level
``word_segments``/``words``. Extracting them here lets the existing per-word
builder score the vendor shape unchanged, rather than duplicating the
run-grouping logic per wire format.

This module deliberately imports nothing outside :mod:`coro.bench`, matching
the rest of the benchmark package, which stays independent of the serving
layer.
"""

from __future__ import annotations

from typing import Any

from coro.bench.errors import UndiarizedResponseError
from coro.bench.wder import UNKNOWN_SPEAKER_LABEL as UNKNOWN_SPEAKER

# The label is taken from the scorer rather than redeclared: WDER is what gives
# it meaning (an abstention, counted as an error but excluded from
# ``wder_claimed``), so emitting anything else here would silently score as an
# ordinary speaker. The serving layer spells the same sentinel in
# ``coro.api.utterances``; a test pins the two together, since bench stays
# independent of that layer.


def _speaker(word: dict[str, Any]) -> str:
    """Map Deepgram's optional integer speaker onto a coro speaker label.

    Deepgram spells "no speaker" as a null or an omitted key, and coro's
    response projection serializes with ``exclude_none``, so an abstention and
    an undiarized request are indistinguishable on the wire — both arrive with
    no ``speaker``. Either way the word carries no attribution, which is what
    the ``-1`` sentinel already means to the per-word STM path.
    """
    speaker = word.get("speaker")
    return UNKNOWN_SPEAKER if speaker is None else str(speaker)


def deepgram_word_segments(
    response: dict[str, Any],
    *,
    recording_id: str = "?",
) -> list[dict[str, Any]]:
    """Extract per-word entries from a Deepgram-shaped response.

    Reads the top hypothesis of the first channel — coro downmixes uploads to
    mono, so there is exactly one channel and the first alternative is the only
    one served. Returns an empty list for any response that is not this shape,
    so callers can treat it as a "not a Deepgram response" signal.

    Raises:
        UndiarizedResponseError: If the response has words but not one carries
            a speaker. Per word that is indistinguishable from abstention, but
            across a whole response it means diarization never ran, and scoring
            it would report a WDER computed over no attribution at all.

    """
    results = response.get("results")
    if not isinstance(results, dict):
        return []
    channels = results.get("channels")
    if not isinstance(channels, list) or not channels:
        return []
    alternatives = channels[0].get("alternatives")
    if not isinstance(alternatives, list) or not alternatives:
        return []
    words = alternatives[0].get("words")
    if not isinstance(words, list) or not words:
        return []
    if all(word.get("speaker") is None for word in words):
        raise UndiarizedResponseError(recording_id, word_count=len(words))
    return [{**word, "speaker": _speaker(word)} for word in words]
