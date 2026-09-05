"""Synthetic code-switch corpus models.

Declarative descriptions of the FLEURS language configs spliced together to
build a synthetic Spanish/English code-switch eval corpus (see
`.scratch/auto-vs-forced-language-wer/PRD.md`), the named splice presets, and
the per-item ground truth recorded for each synthetic item. Behaviour that
consumes these models lives in ``bench.code_switch``.

Unlike the Spanish Workload Set (one clip = one item, one language), every
item here is spliced together from several source clips, so its ground truth
is a list of :class:`CodeSwitchSegment` -- the exact source corpus, language
and offset of every spliced-in clip, known by construction rather than by
human annotation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from coro.bench.models.spanish import CorpusManifest


@dataclass(frozen=True)
class CodeSwitchCorpus:
    """One FLEURS language config used as a carrier or switch source.

    ``key`` is the token recorded against every segment spliced from this
    corpus (``CodeSwitchSegment.source_corpus``), so the source of a segment
    is always recoverable from the item's own manifest.
    """

    key: str
    name: str
    hf_dataset: str
    hf_config: str
    hf_split: str
    licence: str
    licence_url: str
    homepage: str
    language: str
    """ISO 639-1 code of the speech in this corpus (e.g. ``"es"``, ``"en"``)."""
    id_column: str
    text_column: str
    audio_column: str = "audio"
    notes: str = ""


@dataclass(frozen=True)
class CodeSwitchPreset:
    """A named synthetic code-switch corpus: splice pattern, sources, sizing."""

    key: str
    pattern: str
    """``"sparse"`` (few switch-language clips amid a longer carrier run) or
    ``"paragraph"`` (whole clips alternated every step)."""
    carrier_corpus: str
    switch_corpus: str
    carrier_clips_per_item: int
    switch_clips_per_item: int
    items: int
    gap_seconds: float = 1.0
    description: str = ""


@dataclass
class CodeSwitchSegment:
    """Exact ground truth of one spliced-in clip within a synthetic item.

    Recorded at construction time, so no human annotation is needed to know
    which language and which source clip produced any span of the
    concatenated audio.
    """

    source_corpus: str
    source_item_id: str
    language: str
    start_seconds: float
    end_seconds: float
    text: str


@dataclass
class CodeSwitchItem:
    """One materialised synthetic item: its clip and per-segment ground truth."""

    item_id: str
    audio_seconds: float
    segments: list[CodeSwitchSegment] = field(default_factory=list)


@dataclass
class CodeSwitchWorkloadManifest:
    """The manifest written beside a materialised code-switch preset."""

    preset: str
    pattern: str
    description: str
    gap_seconds: float
    items_requested: int
    source_corpora: list[CorpusManifest] = field(default_factory=list)
    items: list[CodeSwitchItem] = field(default_factory=list)
