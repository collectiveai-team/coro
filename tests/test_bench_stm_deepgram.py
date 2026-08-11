"""Scoring a Deepgram-shaped response uses its per-word speakers.

Deepgram nests words at ``results.channels[].alternatives[].words[]``, not at
the top-level ``word_segments``/``words`` that :func:`hyp_response_to_stm`
looks for. Without an adapter the per-word preference misses them and the
segments fallback is taken *silently* — which is the whole failure mode these
tests exist to make loud.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from conftest import FakePipeline, make_app, make_wav
from coro.bench.stm import hyp_response_to_stm, hyp_segments_to_stm


def _deepgram(words: list[dict[str, Any]], **extra: Any) -> Any:
    """Wrap word dicts in Deepgram's channels/alternatives nesting."""
    doc: dict[str, Any] = {
        "metadata": {"request_id": "abc123", "duration": 4.0, "channels": 1},
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": " ".join(str(w["word"]) for w in words),
                            "confidence": 0.9,
                            "words": words,
                        }
                    ]
                }
            ]
        },
    }
    doc.update(extra)
    return doc


def _word(word: str, start: float, end: float, speaker: int | None = None) -> Any:
    item: dict[str, Any] = {"word": word, "start": start, "end": end, "confidence": 0.9}
    if speaker is not None:
        item["speaker"] = speaker
    return item


class TestDeepgramWordsBecomeStm:
    def test_words_group_into_maximal_same_speaker_runs(self):
        response = _deepgram(
            [
                _word("a", 0.0, 1.0, 0),
                _word("b", 1.0, 2.0, 0),
                _word("c", 2.0, 3.0, 1),
                _word("d", 3.0, 4.0, 0),
            ]
        )

        lines = hyp_response_to_stm(response, "rec01").strip().split("\n")

        assert lines == [
            "rec01 1 0 0.000 2.000 a b",
            "rec01 1 1 2.000 3.000 c",
            "rec01 1 0 3.000 4.000 d",
        ]

    def test_absent_speaker_scores_as_the_unknown_sentinel(self):
        """Deepgram spells coro's ``-1`` abstention as an omitted key.

        ``exclude_none`` at serialization drops the null, so an unattributed
        word arrives with no ``speaker`` at all. It must still score as ``-1``,
        the label the per-word path already uses, or abstention silently
        becomes a speaker named ``UNKNOWN``/``None`` that matches no reference.
        """
        response = _deepgram([_word("a", 0.0, 1.0, 0), _word("b", 1.0, 2.0)])

        lines = hyp_response_to_stm(response, "rec01").strip().split("\n")

        assert lines == [
            "rec01 1 0 0.000 1.000 a",
            "rec01 1 -1 1.000 2.000 b",
        ]

    def test_explicit_null_speaker_scores_as_the_unknown_sentinel(self):
        """A client that does not strip nulls must score identically."""
        words = [_word("a", 0.0, 1.0, 0), _word("b", 1.0, 2.0)]
        words[1]["speaker"] = None

        lines = hyp_response_to_stm(_deepgram(words), "rec01").strip().split("\n")

        assert lines == [
            "rec01 1 0 0.000 1.000 a",
            "rec01 1 -1 1.000 2.000 b",
        ]


class TestTheFallbackIsNotSilentlyTaken:
    """Guards against the regression this adapter exists to fix.

    Taking the segments fallback on a Deepgram response fails *silently*: the
    response has no top-level ``segments``, so scoring reports a clean run over
    an empty hypothesis rather than raising. Both tests below fail loudly if
    the vendor surface stops being preferred.
    """

    def test_a_deepgram_response_does_not_score_as_an_empty_hypothesis(self):
        """What the bug looked like: no words found, no segments, no STM, no error."""
        response = _deepgram([_word("a", 0.0, 1.0, 0), _word("b", 1.0, 2.0, 1)])

        assert "segments" not in response
        assert hyp_segments_to_stm(response.get("segments", []), "rec01") == ""
        assert hyp_response_to_stm(response, "rec01") != ""

    def test_per_word_labels_win_over_a_planted_segment_summary(self):
        """A planted summary makes the silent fallback observable.

        A real Deepgram response carries no ``segments``, so this one is
        contrived on purpose: it gives the fallback something plausible to
        return, so that taking it produces a *wrong* STM instead of an empty
        one. The single-speaker summary disagrees with the per-word truth
        exactly where WDER is measured.
        """
        words = [
            _word("a", 0.0, 1.0, 0),
            _word("b", 1.0, 2.0, 0),
            _word("c", 2.0, 3.0, 1),
            _word("d", 3.0, 4.0, 0),
        ]
        segments = [{"start": 0.0, "end": 4.0, "text": "a b c d", "speaker": "0"}]
        response = _deepgram(words, segments=segments)

        result = hyp_response_to_stm(response, "rec01")

        assert result != hyp_segments_to_stm(segments, "rec01")
        assert "rec01 1 1 2.000 3.000 c" in result


@pytest.mark.asyncio
class TestRealListenResponseScoresFromWords:
    """Closes the loop: the served body, not a handwritten fixture.

    The two halves of this change are only useful together — a transport that
    calls ``/v1/listen`` and an STM builder that understands what comes back.
    This exercises the real route so a projection change that moves or renames
    the word list is caught here rather than by a benchmark reporting a
    suspiciously good WDER.
    """

    async def _listen_body(self) -> Any:
        app = make_app(FakePipeline())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/v1/listen?diarize=true&utterances=true",
                content=make_wav(),
                headers={"Content-Type": "audio/wav"},
            )
        return response.json()

    async def test_served_body_yields_per_word_speaker_runs(self):
        body = await self._listen_body()

        lines = hyp_response_to_stm(body, "rec01").strip().split("\n")

        assert lines == [
            "rec01 1 1 0.000 1.000 hola mundo",
            "rec01 1 2 1.200 1.600 si",
            "rec01 1 -1 1.600 2.000 claro",
        ]

    async def test_served_body_carries_no_segments_to_fall_back_on(self):
        """Why preferring the vendor shape is load-bearing, not just tidier."""
        body = await self._listen_body()

        assert "segments" not in body
        assert hyp_segments_to_stm(body.get("segments", []), "rec01") == ""


class TestUnknownSpeakerLabelStaysInSync:
    def test_bench_sentinel_matches_the_serving_layer(self):
        """The label is duplicated to keep bench free of serving-layer imports."""
        from coro.api.utterances import UNKNOWN_SPEAKER_LABEL
        from coro.bench.stm_deepgram import UNKNOWN_SPEAKER

        assert UNKNOWN_SPEAKER == UNKNOWN_SPEAKER_LABEL
