"""Spanish-aware segmentation policy for transcript segment runs."""

from __future__ import annotations

from coro.core.models import TranscriptToken
from coro.core.segmentation import (
    MAX_SEGMENT_SECONDS,
    SegmentAccumulator,
    closes_segment,
    group_tokens_into_runs,
    opens_segment,
    run_span,
)


def _tok(start, end, text):
    return TranscriptToken(start=start, end=end, text=text, probability=1.0)


def _texts(runs):
    return ["".join(t.text for t in run).strip() for run in runs]


# ---------------------------------------------------------------------------
# Terminator policy
# ---------------------------------------------------------------------------


def test_comma_is_not_a_terminator():
    """A comma no longer closes a segment run."""
    assert closes_segment(" cuando,") is False
    runs = group_tokens_into_runs(
        [_tok(0.0, 0.4, " cuando,"), _tok(0.4, 0.8, " llegue"), _tok(0.8, 1.2, " mañana.")]
    )
    assert _texts(runs) == ["cuando, llegue mañana."]


def test_sentence_final_punctuation_closes_a_run():
    for text in (" fin.", " cómo?", " vamos!", " bueno…"):
        assert closes_segment(text) is True
    runs = group_tokens_into_runs([_tok(0.0, 0.4, " uno."), _tok(0.4, 0.8, " dos?")])
    assert _texts(runs) == ["uno.", "dos?"]


# ---------------------------------------------------------------------------
# Spanish opening marks
# ---------------------------------------------------------------------------


def test_opening_mark_pulls_the_boundary_before_it():
    """``¿``/``¡`` begin a sentence, so the split lands before the mark."""
    assert opens_segment(" ¿cómo") is True
    runs = group_tokens_into_runs(
        [
            _tok(0.0, 0.4, " bien"),
            _tok(0.4, 0.8, " gracias"),
            _tok(0.8, 1.2, " ¿y"),
            _tok(1.2, 1.6, " tú?"),
        ]
    )
    assert _texts(runs) == ["bien gracias", "¿y tú?"]


def test_opening_mark_after_a_terminator_does_not_emit_an_empty_run():
    runs = group_tokens_into_runs(
        [_tok(0.0, 0.4, " hola."), _tok(0.4, 0.8, " ¡vaya!"), _tok(0.8, 1.2, " ya.")]
    )
    assert _texts(runs) == ["hola.", "¡vaya!", "ya."]


# ---------------------------------------------------------------------------
# Maximum-duration fallback
# ---------------------------------------------------------------------------


def test_unpunctuated_run_segments_at_the_maximum_duration():
    tokens = [_tok(i * 2.0, (i + 1) * 2.0, f" w{i}") for i in range(12)]
    runs = group_tokens_into_runs(tokens)
    # 12 tokens x 2 s = 24 s, so the 15 s fallback closes exactly one run early:
    # tokens 0-7 (0.0-16.0, the first span to reach the cap) then the remainder.
    assert [len(run) for run in runs] == [8, 4]
    # Compare the closed spans to their expected values, not just their presence.
    assert [run_span(run)[:2] for run in runs] == [(0.0, 16.0), (16.0, 24.0)]
    assert runs[0][-1].end - runs[0][0].start >= MAX_SEGMENT_SECONDS


def test_maximum_duration_fallback_can_be_disabled():
    tokens = [_tok(i * 2.0, (i + 1) * 2.0, f" w{i}") for i in range(12)]
    runs = group_tokens_into_runs(tokens, max_segment_seconds=0.0)
    assert len(runs) == 1


# ---------------------------------------------------------------------------
# Accumulator / batch equivalence
# ---------------------------------------------------------------------------


def test_accumulator_matches_batch_grouping():
    """Incremental accumulation equals whole-list grouping, token by token."""
    tokens = [
        _tok(0.0, 0.4, " hola"),
        _tok(0.4, 0.8, " mundo,"),
        _tok(0.8, 1.2, " ¿qué"),
        _tok(1.2, 1.6, " tal?"),
        _tok(1.6, 2.0, " sin"),
        _tok(2.0, 2.4, " punto"),
    ]
    accumulator = SegmentAccumulator()
    incremental: list[list[TranscriptToken]] = []
    for token in tokens:
        incremental.extend(accumulator.add(token))
    tail = accumulator.flush()
    if tail:
        incremental.append(tail)

    assert _texts(incremental) == _texts(group_tokens_into_runs(tokens))
    assert _texts(incremental) == ["hola mundo,", "¿qué tal?", "sin punto"]


def test_run_span_drops_whitespace_only_runs():
    assert run_span([]) is None
    assert run_span([_tok(0.0, 0.1, "   ")]) is None
