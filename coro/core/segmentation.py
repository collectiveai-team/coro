"""Transcript segmentation policy for the Core Boundary.

Groups Project-Owned transcript tokens into *segment runs* — the contiguous
token spans that later become response segments. The batch response builder and
the Streaming Pipeline's finalizer both drive :class:`SegmentAccumulator`, so
the two paths segment identically by construction.

Spanish-aware rules:

- Only sentence-final punctuation closes a run. The comma is deliberately *not*
  a terminator: Spanish subordinate clauses are comma-heavy, and terminating on
  commas fragments a sentence into very short segments, which makes speaker
  attribution noisier without adding information.
- Spanish opening marks (``¿``/``¡``) begin a sentence. A run is therefore
  closed *before* the token carrying one, so the mark leads its own segment
  instead of being orphaned at the tail of the previous one.
- A maximum run duration bounds long unpunctuated stretches, which the comma
  terminator previously bounded only by accident.
"""

from __future__ import annotations

from coro.core.models import TranscriptToken

SENTENCE_FINAL_PUNCTUATION = ".!?…"
"""Punctuation that terminates a sentence. The comma is intentionally absent."""

OPENING_MARKS = "¿¡"
"""Spanish inverted marks that open a sentence and pull a boundary before them."""

MAX_SEGMENT_SECONDS = 15.0
"""Fallback boundary so an unpunctuated run still segments."""


def closes_segment(text: str) -> bool:
    """Return True when a token's text ends on sentence-final punctuation."""
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in SENTENCE_FINAL_PUNCTUATION


def opens_segment(text: str) -> bool:
    """Return True when a token's text starts with a Spanish opening mark."""
    stripped = text.lstrip()
    return bool(stripped) and stripped[0] in OPENING_MARKS


def run_span(tokens: list[TranscriptToken]) -> tuple[float, float, str] | None:
    """Collapse a run of tokens into a ``(start, end, text)`` span.

    Returns ``None`` when the run is empty or whitespace-only, which is how
    both callers drop runs that carry no transcript.
    """
    if not tokens:
        return None
    text = "".join(t.text for t in tokens)
    if not text.strip():
        return None
    start = min(t.start for t in tokens)
    end = max(t.end for t in tokens)
    if end < start:
        start, end = end, start
    return start, end, text


class SegmentAccumulator:
    """Incrementally group ordered tokens into closed segment runs.

    Tokens are appended one at a time; each :meth:`add` returns the runs the
    new token closed (zero, one, or — when an opening mark closes the previous
    run and the same token also terminates a sentence — two).
    """

    def __init__(self, *, max_segment_seconds: float = MAX_SEGMENT_SECONDS) -> None:
        self._max_segment_seconds = max_segment_seconds
        self._open: list[TranscriptToken] = []

    @property
    def open_tokens(self) -> list[TranscriptToken]:
        """The tokens of the currently open, unterminated run."""
        return self._open

    def add(self, token: TranscriptToken) -> list[list[TranscriptToken]]:
        """Ingest one token and return any runs it closed."""
        if not token.text:
            return []
        closed: list[list[TranscriptToken]] = []
        if self._open and opens_segment(token.text):
            closed.append(self._close())
        self._open.append(token)
        if closes_segment(token.text) or self._exceeds_max_duration():
            closed.append(self._close())
        return [run for run in closed if run]

    def flush(self) -> list[TranscriptToken]:
        """Close and return the trailing open run (possibly empty)."""
        return self._close()

    def _exceeds_max_duration(self) -> bool:
        if self._max_segment_seconds <= 0 or not self._open:
            return False
        span = self._open[-1].end - self._open[0].start
        return span >= self._max_segment_seconds

    def _close(self) -> list[TranscriptToken]:
        run, self._open = self._open, []
        return run


def group_tokens_into_runs(
    tokens: list[TranscriptToken],
    *,
    max_segment_seconds: float = MAX_SEGMENT_SECONDS,
) -> list[list[TranscriptToken]]:
    """Group an entire ordered token list into segment runs."""
    accumulator = SegmentAccumulator(max_segment_seconds=max_segment_seconds)
    runs: list[list[TranscriptToken]] = []
    for token in tokens:
        runs.extend(accumulator.add(token))
    tail = accumulator.flush()
    if tail:
        runs.append(tail)
    return runs
