"""Speaker Boundary Split — cut segments where the speaker timeline changes.

Segments are cut at punctuation, so one segment can span a real turn change
while carrying a single speaker label; every word of the minority speaker is
then silently misattributed. This module finds those turn changes so a segment
can be split at them, upholding the **Single-Speaker Segment Invariant**.

Splits are decided from **Measured Word Start** values alone — which side of a
boundary a word falls on needs its left edge and nothing else. No word ends, no
durations, no forced alignment (ADR 0008).

Both pipelines share this logic but apply it at different moments: the
Full-Memory Pipeline splits during segmentation, where tokens and timeline are
both in hand, and the Streaming Pipeline splits at response assembly against a
persisted segment's stored words, because it commits segments before the
diarizer has produced a timeline at all.
"""

from __future__ import annotations

from dataclasses import dataclass

from coro.core.models import SpeakerSegment

_UNKNOWN_SPEAKER = -1


@dataclass(frozen=True)
class MinimumTurnThreshold:
    """How much speech an interrupting turn needs before it splits a segment.

    A run must clear **both** bounds. Meeting audio is dense with backchannels —
    "mm-hmm", "yeah", "right" — and a one-word backchannel routinely lasts
    longer than the duration bound, so requiring only one of the two would let
    exactly the case this exists to suppress fragment a sentence into three.
    """

    words: int = 2
    seconds: float = 0.4


@dataclass(frozen=True)
class SpeakerRun:
    """A maximal run of consecutive words sharing one speaker.

    ``start`` is inclusive and ``end`` exclusive, indexing the word list the run
    was computed from.
    """

    speaker: int
    start: int
    end: int

    @property
    def word_count(self) -> int:
        """Number of words in the run."""
        return self.end - self.start


def speaker_at_instant(
    instant: float,
    merged: list[SpeakerSegment],
    last_end: float,
) -> int | None:
    """Return the speaker talking at ``instant``, or None if the timeline is silent there.

    ``merged`` must come from ``merge_speaker_timeline``. Instants at or beyond
    the diarization horizon are unknown rather than silent, matching
    ``speaker_for_span``.
    """
    if instant >= last_end:
        return _UNKNOWN_SPEAKER
    for entry in merged:
        if entry.start <= instant < entry.end:
            return entry.speaker
    return None


def _labels_for_starts(
    starts: list[float],
    merged: list[SpeakerSegment],
    last_end: float,
) -> list[int] | None:
    """Label each word start with its speaker, carrying labels across silence.

    A word starting in a gap between turns belongs to the turn it continues, so
    gaps inherit the previous label; a leading gap inherits the first known one.
    Returns None when the timeline resolves no word at all.
    """
    labels: list[int | None] = [speaker_at_instant(s, merged, last_end) for s in starts]
    if all(label is None for label in labels):
        return None

    last_seen: int | None = None
    for i, label in enumerate(labels):
        if label is None:
            labels[i] = last_seen
        else:
            last_seen = label

    first_known = next(label for label in labels if label is not None)
    return [first_known if label is None else label for label in labels]


def _consecutive_runs(labels: list[int]) -> list[SpeakerRun]:
    """Collapse a per-word label list into maximal same-speaker runs."""
    runs: list[SpeakerRun] = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append(SpeakerRun(speaker=labels[start], start=start, end=i))
            start = i
    return runs


def _merge_adjacent(runs: list[SpeakerRun]) -> list[SpeakerRun]:
    """Coalesce neighbouring runs that ended up with the same speaker."""
    merged: list[SpeakerRun] = []
    for run in runs:
        if merged and merged[-1].speaker == run.speaker:
            merged[-1] = SpeakerRun(run.speaker, merged[-1].start, run.end)
        else:
            merged.append(run)
    return merged


def _run_seconds(run: SpeakerRun, starts: list[float], span_end: float) -> float:
    """Return a run's duration, measured start-to-start.

    The run ends where the next one begins; the final run ends with the
    segment. Word ends are deliberately not consulted — see the module
    docstring.
    """
    end = starts[run.end] if run.end < len(starts) else span_end
    return end - starts[run.start]


def _absorb_short_turns(
    runs: list[SpeakerRun],
    starts: list[float],
    span_end: float,
    threshold: MinimumTurnThreshold,
) -> list[SpeakerRun]:
    """Fold runs below the Minimum Turn Threshold into a neighbour.

    A sub-threshold run is not a turn, so its words stay with the surrounding
    speaker: they join the preceding run, or the following one when the segment
    opens on a backchannel.
    """
    while len(runs) > 1:
        short = next(
            (
                i
                for i, run in enumerate(runs)
                if run.word_count < threshold.words
                or _run_seconds(run, starts, span_end) < threshold.seconds
            ),
            None,
        )
        if short is None:
            break
        if short > 0:
            keep = runs[short - 1]
            runs[short - 1] = SpeakerRun(keep.speaker, keep.start, runs[short].end)
        else:
            keep = runs[short + 1]
            runs[short + 1] = SpeakerRun(keep.speaker, runs[short].start, keep.end)
        runs.pop(short)
        runs = _merge_adjacent(runs)
    return runs


def speaker_runs(
    starts: list[float],
    span_end: float,
    merged: list[SpeakerSegment],
    last_end: float,
    threshold: MinimumTurnThreshold,
) -> list[SpeakerRun]:
    """Partition word starts into the turns a segment should be split at.

    Returns an empty list when the segment should be left alone — no timeline,
    too few words, an unresolvable timeline, or a single surviving turn. Callers
    can therefore treat "no runs" as "keep current behaviour" and stay
    byte-identical to the pre-split output wherever no split is warranted.
    """
    if not merged or len(starts) < 2:
        return []
    labels = _labels_for_starts(starts, merged, last_end)
    if labels is None:
        return []
    runs = _absorb_short_turns(
        _consecutive_runs(labels),
        starts,
        span_end,
        threshold,
    )
    return runs if len(runs) > 1 else []
