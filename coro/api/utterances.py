"""Utterance grouping shared by the vendor-native response projections.

Speech vendors carry a *speaker-turn* view (``utterances``) alongside a
per-word view. An utterance is a maximal run of consecutive words sharing one
speaker.

The grouping deliberately reads ``word_segments`` rather than ``segments``.
Per-word speakers are the truth the vendor formats exist to expose, and
deriving turns from them keeps this projection correct no matter what shape
``segments`` takes — sentence-shaped with a majority speaker, or split at every
speaker change. The two views cannot disagree because only one of them is a
source.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from coro.api.schemas import WhisperWord

UNKNOWN_SPEAKER_LABEL = "-1"
"""Speaker label for words the diarization timeline does not support."""


@dataclass
class Utterance:
    """A maximal run of consecutive words attributed to one speaker."""

    speaker: str
    words: list[WhisperWord] = field(default_factory=list)

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
    def confidence(self) -> float:
        """Mean of the words' ASR confidences."""
        return mean_confidence(self.words)


def group_words_into_utterances(words: Sequence[WhisperWord]) -> list[Utterance]:
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


def mean_confidence(words: Sequence[WhisperWord]) -> float:
    """Return the arithmetic mean of the words' ASR confidences.

    Returns ``0.0`` for an empty sequence — an aggregate over no evidence, not
    a claim of zero confidence in something.
    """
    if not words:
        return 0.0
    return sum(word.score for word in words) / len(words)
