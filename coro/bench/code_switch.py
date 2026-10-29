"""Synthetic code-switch corpus: splicing FLEURS clips for the auto-vs-forced study.

Built for `.scratch/auto-vs-forced-language-wer/PRD.md` (issue #64's original,
never-answered question: does forced-language decoding measurably reduce
substitution errors versus auto-detection on audio that genuinely mixes
languages?). Every existing Spanish benchmark corpus in this repo (FLEURS,
VoxPopuli, MLS, `bench.spanish`) is clean and single-language by construction
-- no clip anywhere in the repo's history actually contains more than one
language. This module builds one by splicing together independent FLEURS
``es_419`` ("carrier") and ``en_us`` ("switch") clips.

Because every synthetic item is assembled from clips whose individual
transcripts and languages are already known, the exact language and text of
every spliced-in span is recorded at construction time
(:class:`~coro.bench.models.code_switch.CodeSwitchSegment`) -- this is what
lets the eval skip human annotation entirely (see the PRD's explicit decision
not to build a human-adjudicated FLSR metric this round).

Two splice patterns, both useful and answering different sub-questions (do
not pool their results -- see the PRD's Further Notes):

- ``sparse``: one or few short switch-language clips inserted, evenly
  spaced, amid a longer run of carrier-language clips. The closest synthetic
  analogue to an occasional embedded foreign term or phrase.
- ``paragraph``: whole carrier/switch clips alternated one-for-one. Simulates
  a full topic- or speaker-driven language change.

Mirrors ``bench.spanish``'s shape deliberately (same registry-of-dataclasses
plus idempotent ``materialize_*``/manifest pattern) so this doesn't grow a
second, differently-shaped way to materialise a benchmark corpus. The FLEURS
source-corpus provenance is recorded with the same
:class:`~coro.bench.models.spanish.CorpusManifest` block ``bench.spanish``
already uses, not a parallel type.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from coro.bench.code_switch_presets import (
    CODE_SWITCH_CORPORA,
    CODE_SWITCH_PRESETS,
    resolve_code_switch_preset,
)
from coro.bench.models.code_switch import (
    CodeSwitchCorpus,
    CodeSwitchItem,
    CodeSwitchPreset,
    CodeSwitchSegment,
    CodeSwitchWorkloadManifest,
)
from coro.bench.models.spanish import CorpusManifest
from coro.bench.stm import hyp_segments_to_stm
from coro.bench.utils.audio_clips import concat_wav_clips, transcode_bytes_to_wav
from coro.bench.utils.hf_parquet import iter_parquet_rows, resolve_shard_urls

MANIFEST_NAME = "code-switch.json"
LICENCES_NAME = "LICENCES.md"

__all__ = [
    "CODE_SWITCH_CORPORA",
    "CODE_SWITCH_PRESETS",
    "LICENCES_NAME",
    "MANIFEST_NAME",
    "build_sequence",
    "materialize_code_switch_corpus",
    "render_licences",
    "resolve_code_switch_preset",
]


def _paragraph_sequence[T](carrier: list[T], switch: list[T]) -> list[tuple[str, T]]:
    sequence: list[tuple[str, T]] = []
    for carrier_row, switch_row in zip(carrier, switch, strict=False):
        sequence.append(("carrier", carrier_row))
        sequence.append(("switch", switch_row))
    sequence.extend(("carrier", row) for row in carrier[len(switch) :])
    sequence.extend(("switch", row) for row in switch[len(carrier) :])
    return sequence


def _sparse_sequence[T](carrier: list[T], switch: list[T]) -> list[tuple[str, T]]:
    if not switch:
        return [("carrier", row) for row in carrier]
    if not carrier:
        return [("switch", row) for row in switch]
    # A monotonically increasing threshold (half a step ahead of the
    # per-switch carrier ratio) spaces insertions evenly without ever
    # mapping two switch rows to the same slot -- a plain
    # round()-to-an-index scheme can collide when there are few carrier
    # clips per switch clip, silently dropping a row.
    ratio = len(carrier) / len(switch)
    threshold = ratio / 2
    switch_index = 0
    sequence: list[tuple[str, T]] = []
    for position, carrier_row in enumerate(carrier, start=1):
        sequence.append(("carrier", carrier_row))
        if switch_index < len(switch) and position >= threshold:
            sequence.append(("switch", switch[switch_index]))
            switch_index += 1
            threshold += ratio
    sequence.extend(("switch", row) for row in switch[switch_index:])
    return sequence


def build_sequence[T](
    pattern: str,
    carrier: list[T],
    switch: list[T],
) -> list[tuple[str, T]]:
    """Interleave carrier/switch clip rows into one splice order per ``pattern``.

    ``"paragraph"`` alternates every single clip, carrier first (``c0, s0,
    c1, s1, ...``), padding with whichever side runs out. ``"sparse"``
    distributes ``switch`` rows evenly amid the carrier run (e.g. one switch
    clip among four carrier clips lands in the middle), so a handful of
    insertions read as scattered rather than clustered at one end.
    """
    if pattern == "paragraph":
        return _paragraph_sequence(carrier, switch)
    if pattern == "sparse":
        return _sparse_sequence(carrier, switch)
    raise ValueError(f"Unknown code-switch pattern {pattern!r}; known patterns: sparse, paragraph")


@dataclass(frozen=True)
class _SourceClip:
    """One fetched source row: its corpus id, transcript and encoded audio."""

    item_id: str
    text: str
    audio_bytes: bytes


def _fetch_rows(corpus: CodeSwitchCorpus, count: int) -> list[_SourceClip]:
    """Fetch up to ``count`` non-empty (text, audio) rows from one FLEURS config."""
    if count <= 0:
        return []
    urls = resolve_shard_urls(corpus.hf_dataset, corpus.hf_config, corpus.hf_split)
    columns = [corpus.audio_column, corpus.id_column, corpus.text_column]

    rows: list[_SourceClip] = []
    for row in iter_parquet_rows(urls, limit=count, columns=columns):
        text = str(row.get(corpus.text_column) or "").strip()
        audio = row.get(corpus.audio_column)
        audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
        if not text or not audio_bytes:
            continue
        rows.append(
            _SourceClip(item_id=str(row.get(corpus.id_column)), text=text, audio_bytes=audio_bytes)
        )
    return rows


def _materialize_item(
    item_id: str,
    sequence: list[tuple[str, _SourceClip]],
    preset: CodeSwitchPreset,
    out_dir: Path,
) -> CodeSwitchItem | None:
    """Splice one synthetic item's clips together and write its clip + Reference STM."""
    parts_dir = out_dir / f".{item_id}.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    clip_paths: list[Path] = []
    try:
        for slot, (_role, row) in enumerate(sequence):
            clip_path = parts_dir / f"{slot:02d}.wav"
            transcode_bytes_to_wav(row.audio_bytes, clip_path)
            clip_paths.append(clip_path)

        wav_path = out_dir / f"{item_id}.wav"
        spans = concat_wav_clips(clip_paths, wav_path, gap_seconds=preset.gap_seconds)
    finally:
        for clip_path in clip_paths:
            clip_path.unlink(missing_ok=True)
        parts_dir.rmdir()

    segments = []
    for (role, row), (start, end) in zip(sequence, spans, strict=True):
        corpus_key = preset.carrier_corpus if role == "carrier" else preset.switch_corpus
        segments.append(
            CodeSwitchSegment(
                source_corpus=corpus_key,
                source_item_id=row.item_id,
                language=CODE_SWITCH_CORPORA[corpus_key].language,
                start_seconds=round(start, 3),
                end_seconds=round(end, 3),
                text=row.text,
            )
        )

    stm_text = hyp_segments_to_stm(
        [
            {"start": seg.start_seconds, "end": seg.end_seconds, "text": seg.text, "speaker": "1"}
            for seg in segments
        ],
        item_id,
    )
    if not stm_text:
        wav_path.unlink(missing_ok=True)
        return None
    (out_dir / f"{item_id}.ref.stm").write_text(stm_text, encoding="utf-8")

    return CodeSwitchItem(item_id=item_id, audio_seconds=round(spans[-1][1], 3), segments=segments)


def _corpus_manifest(
    corpus: CodeSwitchCorpus, role: str, requested: int, materialised: int
) -> CorpusManifest:
    """Build the licence and provenance record for one source corpus."""
    return CorpusManifest(
        key=corpus.key,
        name=corpus.name,
        role=role,
        licence=corpus.licence,
        licence_url=corpus.licence_url,
        homepage=corpus.homepage,
        hf_dataset=corpus.hf_dataset,
        hf_config=corpus.hf_config,
        hf_split=corpus.hf_split,
        single_speaker=True,
        notes=corpus.notes,
        requested=requested,
        materialised=materialised,
    )


def render_licences(manifest: CodeSwitchWorkloadManifest) -> str:
    """Render the per-source-corpus licence record shipped beside the clips."""
    lines = [
        "# Code-switch corpus — source licences",
        "",
        f"Preset: `{manifest.preset}` ({manifest.pattern})",
        "",
        "| Source corpus | Role | Licence | Source | Rows fetched |",
        "|---|---|---|---|---:|",
    ]
    for block in manifest.source_corpora:
        lines.append(
            f"| {block.name} | {block.role} | "
            f"[{block.licence}]({block.licence_url}) | "
            f"[{block.hf_dataset}]({block.homepage}) | {block.materialised} |"
        )
    lines += [
        "",
        f"{len(manifest.items)} synthetic items materialised by splicing the "
        "source corpora above with silence gaps; every item's manifest entry "
        "records the exact source clip, language and offset of each spliced "
        "segment, so no human language annotation is needed to score it.",
        "",
    ]
    return "\n".join(lines)


def _manifest_is_current(
    manifest: Mapping[str, object], preset: CodeSwitchPreset, items: int
) -> bool:
    return (
        manifest.get("preset") == preset.key
        and manifest.get("pattern") == preset.pattern
        and manifest.get("items_requested") == items
    )


def materialize_code_switch_corpus(
    preset_key: str,
    root: Path,
    *,
    items: int | None = None,
    no_download: bool = False,
) -> Path:
    """Materialise a code-switch preset and return its ``--clips-dir``.

    Idempotent: an existing manifest for the same preset and item count is
    reused without re-fetching or re-splicing. Every synthetic item is a
    ``(<preset>-<NNN>.wav, <preset>-<NNN>.ref.stm)`` pair, directly consumable
    by the existing Quality Benchmark path unchanged (same shape
    ``materialize_spanish_workload_set`` already produces), plus
    ``code-switch.json`` (per-item segment ground truth) and ``LICENCES.md``.

    Args:
        preset_key: A key of :data:`CODE_SWITCH_PRESETS`.
        root: Directory holding one subdirectory per materialised preset.
        items: Override the preset's item count.
        no_download: Fail instead of fetching anything over the network.

    Returns:
        The clips directory for the preset.

    """
    preset = resolve_code_switch_preset(preset_key)
    count = items if items is not None else preset.items
    out_dir = root / preset.key
    manifest_path = out_dir / MANIFEST_NAME

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _manifest_is_current(manifest, preset, count):
            return out_dir

    if no_download:
        raise RuntimeError(
            f"Code-switch preset {preset.key!r} is not materialised under {out_dir} "
            "(and --no-download was set). Re-run without --no-download."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    carrier_corpus = CODE_SWITCH_CORPORA[preset.carrier_corpus]
    switch_corpus = CODE_SWITCH_CORPORA[preset.switch_corpus]

    carrier_needed = count * preset.carrier_clips_per_item
    switch_needed = count * preset.switch_clips_per_item
    carrier_rows = _fetch_rows(carrier_corpus, carrier_needed)
    switch_rows = _fetch_rows(switch_corpus, switch_needed)

    items_out: list[CodeSwitchItem] = []
    for index in range(count):
        c_slice = carrier_rows[
            index * preset.carrier_clips_per_item : (index + 1) * preset.carrier_clips_per_item
        ]
        s_slice = switch_rows[
            index * preset.switch_clips_per_item : (index + 1) * preset.switch_clips_per_item
        ]
        if (
            len(c_slice) < preset.carrier_clips_per_item
            or len(s_slice) < preset.switch_clips_per_item
        ):
            break  # ran out of source rows (some may have been skipped for empty text)

        sequence = build_sequence(preset.pattern, c_slice, s_slice)
        item = _materialize_item(f"{preset.key}-{index:03d}", sequence, preset, out_dir)
        if item is not None:
            items_out.append(item)

    manifest = CodeSwitchWorkloadManifest(
        preset=preset.key,
        pattern=preset.pattern,
        description=preset.description,
        gap_seconds=preset.gap_seconds,
        items_requested=count,
        source_corpora=[
            _corpus_manifest(carrier_corpus, "carrier", carrier_needed, len(carrier_rows)),
            _corpus_manifest(switch_corpus, "switch", switch_needed, len(switch_rows)),
        ],
        items=items_out,
    )
    manifest_path.write_text(
        json.dumps(asdict(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / LICENCES_NAME).write_text(render_licences(manifest), encoding="utf-8")
    return out_dir
