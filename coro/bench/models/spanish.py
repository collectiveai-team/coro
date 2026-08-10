"""Spanish Workload Set models.

Declarative descriptions of the freely-licensed public Spanish corpora the
Spanish Workload Set is built from, the named presets that group them, and the
published-WER calibration targets used to prove the harness is free of
systematic error. Behaviour that consumes these models lives in
``bench.spanish`` and ``bench.calibration``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SpanishCorpus:
    """One freely-licensed public Spanish corpus available to the workload set.

    ``key`` is also the item-id prefix every **Workload Item** materialised from
    this corpus carries, so the corpus of an item is recoverable from its id
    alone (and therefore from a ``quality/`` artifact filename). Keys must not
    contain ``-``.
    """

    key: str
    name: str
    hf_dataset: str
    hf_config: str
    hf_split: str
    licence: str
    licence_url: str
    homepage: str
    role: str
    id_column: str
    text_column: str
    audio_column: str = "audio"
    normalized_text_column: str | None = None
    notes: str = ""
    # Every corpus here is read/parliamentary single-speaker material, so it
    # yields no meaningful DER. Diarization stays AMI-driven (see PRD).
    single_speaker: bool = True


@dataclass(frozen=True)
class SpanishPreset:
    """A named Spanish Workload Set: which corpora, and how many items each."""

    key: str
    corpora: tuple[str, ...]
    items_per_corpus: int
    description: str


@dataclass
class SpanishItem:
    """One materialised Workload Item: its clip, duration and reference text."""

    item_id: str
    audio_seconds: float
    reference_text: str
    corpus_normalized_text: str | None = None


@dataclass
class CorpusManifest:
    """Licence and provenance record for one materialised corpus."""

    key: str
    name: str
    role: str
    licence: str
    licence_url: str
    homepage: str
    hf_dataset: str
    hf_config: str
    hf_split: str
    single_speaker: bool
    notes: str
    requested: int
    materialised: int
    items: list[SpanishItem] = field(default_factory=list)


@dataclass
class SpanishWorkloadManifest:
    """The ``corpora.json`` written beside a materialised Spanish preset."""

    preset: str
    description: str
    items_per_corpus: int
    corpora: list[CorpusManifest] = field(default_factory=list)


@dataclass(frozen=True)
class CorpusFetchPlan:
    """Download footprint of materialising one corpus."""

    corpus: str
    rows: int
    row_groups: int
    download_bytes: int


@dataclass
class CorpusStats:
    """Aggregated scoring counters for one corpus in a benchmark run."""

    corpus: str
    errors: int = 0
    length: int = 0
    items: int = 0


@dataclass(frozen=True)
class CalibrationTarget:
    """A published WER figure for one (ASR Model Selection, corpus) pair.

    Only externally published, citable figures belong here; they are the sole
    end-to-end proof that decoding, windowing, STM conversion, normalization and
    scoring are free of systematic error.
    """

    model_id: str
    corpus: str
    published_wer: float
    source: str
    source_url: str


@dataclass
class CalibrationOutcome:
    """The calibration verdict for one corpus in one benchmark run."""

    corpus: str
    status: str
    n_items: int = 0
    scored_wer: float | None = None
    published_wer: float | None = None
    deviation: float | None = None
    margin: float | None = None
    model_id: str | None = None
    source: str | None = None
    source_url: str | None = None
    detail: str = ""


@dataclass
class CalibrationReport:
    """Run-level calibration result written to ``quality/calibration.json``."""

    model_id: str | None = None
    metric: str = ""
    margin: float = 0.0
    failed: bool = False
    outcomes: list[CalibrationOutcome] = field(default_factory=list)
