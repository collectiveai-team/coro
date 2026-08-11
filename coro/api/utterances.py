"""Utterance grouping shared by the vendor-native response projections.

Speech vendors carry a *speaker-turn* view (``utterances``) alongside a
per-word view. An utterance is a maximal run of consecutive words sharing one
speaker.

The grouping deliberately reads ``word_segments`` rather than ``segments``.
Per-word speakers are the truth the vendor formats exist to expose. Segments are
sentence-shaped and carry the duration-weighted majority of their own words
(ADR 0014), so a segment can span a speaker turn and is not a valid utterance;
deriving turns from the per-word view is what keeps this projection correct. The
two views cannot disagree because only one of them is a source.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from coro.api.schemas import TranscriptWord

UNKNOWN_SPEAKER_LABEL = "-1"
"""Speaker label for words the diarization timeline does not support."""


@dataclass
class Utterance:
    """A maximal run of consecutive words attributed to one speaker."""

    speaker: str
    words: list[TranscriptWord] = field(default_factory=list)

    @property
    def start(self) -> float:
        """Start of the first word, in seconds."""
        return min(word.start for word in self.words)

    @property
    def end(self) -> float:
        """End of the last word, in seconds."""
        return max(word.end for word in self.words)

    @property
    def text(self) -> str:
        """The utterance transcript, rebuilt from its words."""
        return " ".join(word.word for word in self.words).strip()

    @property
    def confidence(self) -> float | None:
        """Mean of the words' ASR confidences, or ``None`` if none were measured."""
        return mean_confidence(self.words)


def group_words_into_utterances(words: Sequence[TranscriptWord]) -> list[Utterance]:
    """Group consecutive words into maximal same-speaker utterances.

    Args:
        words: Per-word entries in transcript order, each carrying the speaker
            assigned by word-level attribution.

    Returns:
        Utterances in transcript order. An empty input yields an empty list.
        Words are never reordered, so a speaker that recurs later in the
        transcript produces a separate utterance rather than being merged.

    """
    utterances: list[Utterance] = []
    for word in words:
        if utterances and utterances[-1].speaker == word.speaker:
            utterances[-1].words.append(word)
            continue
        utterances.append(Utterance(speaker=word.speaker, words=[word]))
    return utterances


def mean_measured(values: Iterable[float | None]) -> float | None:
    """Return the mean of the values that were actually measured.

    ``None`` entries are *excluded from the average*, not counted as any value:
    averaging in a ``0.0`` would understate the entries that were measured, and
    averaging in a ``1.0`` would overstate them.

    Returns ``None`` when nothing was measured — including for an empty
    iterable. An aggregate over no evidence is absent, not ``0.0``, which is a
    claim of zero confidence in something (ADR 0015 rule 3).

    Shared by every confidence aggregate so the batch and live surfaces cannot
    drift apart on what an unmeasured word does to a mean.
    """
    measured = [value for value in values if value is not None]
    if not measured:
        return None
    return sum(measured) / len(measured)


def mean_confidence(words: Sequence[TranscriptWord]) -> float | None:
    """Return the mean of the words' measured ASR confidences, or ``None``."""
    return mean_measured(word.score for word in words)
