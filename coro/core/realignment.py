"""Flicker correction for per-word speaker labels.

Per-word attribution splits a segment run at every speaker change, which takes
the diarization timeline at face value. A timeline that changes hands for a
fraction of a second manufactures *flicker*: a stretch of words whose speaker
differs from both neighbours, where those two neighbours agree. Measured over
six AMI meetings, splitting on the raw change tripled it — 11.6% of segments to
27.6% — and doubled one-word segments.

This module corrects that, and the shape of the correction is the whole point.

**Only sandwiched islands are absorbed.** An island of one speaker flanked on
both sides by the *same* other speaker is the exact pattern that defines
flicker, and it is the only pattern touched here. A clean two-way turn
(``A B``) is left alone, because nothing about it says "error": it is what a
genuine turn looks like. This is a width-3 mode filter over the speaker
sequence — the degenerate case of the median filtering used to post-process
frame-level diarization (Medennikov et al., *Target-Speaker Voice Activity
Detection*, 2020) — applied on the word axis.

The alternative, and the first thing tried here, was the punctuation-bounded
majority vote from MahmoudAshraf/whisper-diarization: relabel every word that
disagrees with its sentence's majority. It was measured and rejected. That rule
flattens *any* sentence containing a mid-sentence speaker change, and on this
workload 62% of what it flattened were clean two-way turns rather than flicker.
It made DER speaker-error worse than the pre-attribution baseline on 6 of 6
clips. Its upstream design target is different from ours — it exists to repair
turn boundaries landing a few hundred milliseconds inside a sentence, after a
punctuation-restoration model has run, on two-speaker audio. See ADR 0008.

Both rules are bounded by sentences, and that part is kept: punctuation is
evidence of a genuine turn, so a change that coincides with it is never
second-guessed.
"""

from __future__ import annotations

from dataclasses import dataclass

from coro.core.models import TranscriptToken
from coro.core.segmentation import closes_segment, opens_segment
from coro.core.speakers import UNKNOWN_SPEAKER, SpeakerAttribution

_AttributedToken = tuple[TranscriptToken, SpeakerAttribution]


@dataclass(frozen=True)
class _Island:
    """A maximal run of consecutive words sharing one real speaker."""

    speaker: int
    indices: tuple[int, ...]
    duration: float

    def merged_with(self, *others: _Island) -> _Island:
        """Return one island covering ``self`` and ``others``, keeping this speaker."""
        indices = self.indices
        duration = self.duration
        for other in others:
            indices += other.indices
            duration += other.duration
        return _Island(speaker=self.speaker, indices=indices, duration=duration)


def realign_speaker_flicker(attributed: list[_AttributedToken]) -> list[_AttributedToken]:
    """Absorb sandwiched speaker islands into the speaker flanking them.

    Within each punctuation-bounded sentence, an island of one speaker whose
    immediate neighbours are both the *same* other speaker is relabelled to
    that flanking speaker — but only when the island holds less total word
    duration than the two flanks combined. Duration, not word count: one long
    word is stronger evidence than several short ones, and a long island
    between two brief ones is not a blip.

    That duration test is a comparison between measured quantities, not a
    tuned threshold, so the rule has no free parameter. A clean two-way turn is
    never touched, because it is not sandwiched.

    :data:`~coro.core.speakers.UNKNOWN_SPEAKER` words are transparent: they
    neither break an island nor form one, they contribute no duration, and they
    are never relabelled. Abstention must not be overwritten by a correction it
    cannot participate in.

    Args:
        attributed: Tokens paired with their per-word attribution, in order.

    Returns:
        A new list, same length and order, with flicker speakers corrected.

    """
    if not attributed:
        return []

    realigned = list(attributed)
    for lo, hi in _sentence_bounds(attributed):
        for index, speaker in _absorbed_indices(_speaker_islands(attributed, lo, hi)):
            token, attribution = attributed[index]
            realigned[index] = (
                token,
                SpeakerAttribution(speaker=speaker, overlap=attribution.overlap),
            )
    return realigned


def _absorbed_indices(islands: list[_Island]) -> list[tuple[int, int]]:
    """Return ``(token index, new speaker)`` for every sandwiched island.

    Absorbing an island joins it to both flanks, so the result is re-tested
    against the next island along. That keeps an alternating ``A B A B A`` run
    from producing contradictory relabels, and terminates because every
    absorption shortens the list.
    """
    relabels: list[tuple[int, int]] = []
    position = 1
    while position < len(islands) - 1:
        left, island, right = islands[position - 1 : position + 2]
        sandwiched = left.speaker == right.speaker
        outweighed = island.duration < left.duration + right.duration
        if not (sandwiched and outweighed):
            position += 1
            continue
        relabels.extend((index, left.speaker) for index in island.indices)
        islands[position - 1 : position + 2] = [left.merged_with(island, right)]
    return relabels


def _speaker_islands(attributed: list[_AttributedToken], lo: int, hi: int) -> list[_Island]:
    """Group ``attributed[lo:hi]`` into maximal same-speaker islands, skipping unknowns."""
    islands: list[_Island] = []
    for index in range(lo, hi):
        token, attribution = attributed[index]
        if attribution.speaker == UNKNOWN_SPEAKER:
            continue
        current = _Island(
            speaker=attribution.speaker, indices=(index,), duration=token.duration()
        )
        if islands and islands[-1].speaker == attribution.speaker:
            islands[-1] = islands[-1].merged_with(current)
        else:
            islands.append(current)
    return islands


def _sentence_bounds(attributed: list[_AttributedToken]) -> list[tuple[int, int]]:
    """Split ``attributed`` into ``[start, stop)`` index ranges, one per sentence."""
    bounds: list[tuple[int, int]] = []
    start = 0
    total = len(attributed)
    for index in range(1, total + 1):
        at_end = index == total
        prev_closes = closes_segment(attributed[index - 1][0].text)
        next_opens = not at_end and opens_segment(attributed[index][0].text)
        if at_end or prev_closes or next_opens:
            bounds.append((start, index))
            start = index
    return bounds
