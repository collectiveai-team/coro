"""Spanish Workload Set: public-corpus resolution and materialisation.

Mirrors ``bench.ami`` for Spanish. A named preset selects one or more
freely-licensed public corpora, materialises them into a ``--clips-dir`` of
``(<item_id>.wav, <item_id>.ref.stm)`` pairs, and records the licence and
provenance of every corpus in a manifest next to the clips. The resulting
directory is consumed by the existing **Quality Benchmark** path unchanged.

Every corpus here is single-speaker, so the Spanish Workload Set validates ASR
quality only and yields no meaningful DER; diarization decisions stay AMI-driven.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from coro.bench.models.spanish import (
    CorpusFetchPlan,
    CorpusManifest,
    SpanishCorpus,
    SpanishItem,
    SpanishPreset,
    SpanishWorkloadManifest,
)
from coro.bench.stm import hyp_segments_to_stm
from coro.bench.utils.audio_clips import transcode_bytes_to_wav, wav_duration_seconds
from coro.bench.utils.hf_parquet import iter_parquet_rows, plan_fetch, resolve_shard_urls

MANIFEST_NAME = "corpora.json"
LICENCES_NAME = "LICENCES.md"

SPANISH_CORPORA: dict[str, SpanishCorpus] = {
    "voxpopuli": SpanishCorpus(
        key="voxpopuli",
        name="VoxPopuli (Spanish, ASR test split)",
        hf_dataset="facebook/voxpopuli",
        hf_config="es",
        hf_split="test",
        licence="CC0-1.0",
        licence_url="https://creativecommons.org/publicdomain/zero/1.0/",
        homepage="https://huggingface.co/datasets/facebook/voxpopuli",
        role="primary",
        id_column="audio_id",
        text_column="raw_text",
        normalized_text_column="normalized_text",
        notes=(
            "Spontaneous European Parliament speech at 16 kHz. Ships both a raw "
            "and a normalized transcript field; the Reference STM carries the raw "
            "text so the harness derives its own normalized lane."
        ),
    ),
    "fleurs": SpanishCorpus(
        key="fleurs",
        name="FLEURS (es_419, test split)",
        hf_dataset="google/fleurs",
        hf_config="es_419",
        hf_split="test",
        licence="CC-BY-4.0",
        licence_url="https://creativecommons.org/licenses/by/4.0/",
        homepage="https://huggingface.co/datasets/google/fleurs",
        role="calibration",
        id_column="id",
        text_column="raw_transcription",
        normalized_text_column="transcription",
        notes="Read speech. Calibration set: published WER exists for common ASR models.",
    ),
    "mls": SpanishCorpus(
        key="mls",
        name="Multilingual LibriSpeech (Spanish, test split)",
        hf_dataset="facebook/multilingual_librispeech",
        hf_config="spanish",
        hf_split="test",
        licence="CC-BY-4.0",
        licence_url="https://creativecommons.org/licenses/by/4.0/",
        homepage="https://huggingface.co/datasets/facebook/multilingual_librispeech",
        role="calibration",
        id_column="id",
        text_column="transcript",
        notes=(
            "Read audiobook speech. Its transcripts are already lowercase and "
            "unpunctuated, so the raw and normalized lanes nearly coincide."
        ),
    ),
}

SPANISH_PRESETS: dict[str, SpanishPreset] = {
    "voxpopuli": SpanishPreset(
        key="voxpopuli",
        corpora=("voxpopuli",),
        items_per_corpus=50,
        description="Primary Spanish workload: CC0 spontaneous parliamentary speech.",
    ),
    "fleurs": SpanishPreset(
        key="fleurs",
        corpora=("fleurs",),
        items_per_corpus=50,
        description="Calibration only: FLEURS es_419.",
    ),
    "mls": SpanishPreset(
        key="mls",
        corpora=("mls",),
        items_per_corpus=50,
        description="Calibration only: Multilingual LibriSpeech Spanish.",
    ),
    "calibration": SpanishPreset(
        key="calibration",
        corpora=("fleurs", "mls"),
        items_per_corpus=50,
        description="Both calibration sets; proves the harness against published WER.",
    ),
    "all": SpanishPreset(
        key="all",
        corpora=("voxpopuli", "fleurs", "mls"),
        items_per_corpus=50,
        description="Primary workload plus both calibration sets.",
    ),
}

_UNSAFE_ID = re.compile(r"[^A-Za-z0-9._]+")


def corpus_of_item(item_id: str) -> str:
    """Return the corpus key encoded in a Spanish workload item id."""
    return item_id.split("-", 1)[0]


def _safe_id(raw: Any) -> str:
    """Return a filesystem- and STM-safe token derived from a corpus row id."""
    text = Path(str(raw)).stem if isinstance(raw, str) and "." in str(raw) else str(raw)
    return _UNSAFE_ID.sub("_", text).strip("_") or "item"


def _corpus_columns(corpus: SpanishCorpus) -> list[str]:
    columns = [corpus.audio_column, corpus.id_column, corpus.text_column]
    if corpus.normalized_text_column:
        columns.append(corpus.normalized_text_column)
    return list(dict.fromkeys(columns))


def preset_fetch_plan(
    preset_key: str,
    *,
    items_per_corpus: int | None = None,
) -> list[CorpusFetchPlan]:
    """Return the per-corpus download footprint of materialising a preset.

    Reads only Parquet footers, so it is safe to call before committing to a
    large fetch.
    """
    preset = resolve_spanish_preset(preset_key)
    count = items_per_corpus or preset.items_per_corpus
    plans: list[CorpusFetchPlan] = []
    for key in preset.corpora:
        corpus = SPANISH_CORPORA[key]
        urls = resolve_shard_urls(corpus.hf_dataset, corpus.hf_config, corpus.hf_split)
        plan = plan_fetch(urls, limit=count, columns=_corpus_columns(corpus))
        plans.append(
            CorpusFetchPlan(
                corpus=key,
                rows=plan.rows,
                row_groups=plan.row_groups,
                download_bytes=plan.download_bytes,
            )
        )
    return plans


def resolve_spanish_preset(preset_key: str) -> SpanishPreset:
    """Return the named Spanish preset, or fail with the known preset names."""
    try:
        return SPANISH_PRESETS[preset_key]
    except KeyError:
        known = ", ".join(sorted(SPANISH_PRESETS))
        raise ValueError(f"Unknown Spanish preset {preset_key!r}; known presets: {known}") from None


def _materialize_corpus(
    corpus: SpanishCorpus,
    out_dir: Path,
    count: int,
) -> list[SpanishItem]:
    """Fetch, transcode and write ``count`` workload items for one corpus."""
    urls = resolve_shard_urls(corpus.hf_dataset, corpus.hf_config, corpus.hf_split)
    columns = _corpus_columns(corpus)

    entries: list[SpanishItem] = []
    seen: set[str] = set()

    for row in iter_parquet_rows(urls, limit=count, columns=columns):
        text = str(row.get(corpus.text_column) or "").strip()
        audio = row.get(corpus.audio_column) or {}
        audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
        if not text or not audio_bytes:
            continue

        item_id = f"{corpus.key}-{_safe_id(row.get(corpus.id_column))}"
        if item_id in seen:
            continue
        seen.add(item_id)

        wav_path = out_dir / f"{item_id}.wav"
        transcode_bytes_to_wav(audio_bytes, wav_path)
        duration = wav_duration_seconds(wav_path)

        stm_text = hyp_segments_to_stm(
            [{"start": 0.0, "end": duration, "text": text, "speaker": "1"}],
            item_id,
        )
        if not stm_text:
            wav_path.unlink(missing_ok=True)
            continue
        (out_dir / f"{item_id}.ref.stm").write_text(stm_text, encoding="utf-8")

        normalized = None
        if corpus.normalized_text_column:
            normalized = str(row.get(corpus.normalized_text_column) or "").strip()
        entries.append(
            SpanishItem(
                item_id=item_id,
                audio_seconds=round(duration, 3),
                reference_text=text,
                corpus_normalized_text=normalized,
            )
        )

    return entries


def _corpus_manifest(
    corpus: SpanishCorpus,
    requested: int,
    entries: list[SpanishItem],
) -> CorpusManifest:
    """Build the licence and provenance record for one materialised corpus."""
    return CorpusManifest(
        key=corpus.key,
        name=corpus.name,
        role=corpus.role,
        licence=corpus.licence,
        licence_url=corpus.licence_url,
        homepage=corpus.homepage,
        hf_dataset=corpus.hf_dataset,
        hf_config=corpus.hf_config,
        hf_split=corpus.hf_split,
        single_speaker=corpus.single_speaker,
        notes=corpus.notes,
        requested=requested,
        materialised=len(entries),
        items=entries,
    )


def render_licences(manifest: SpanishWorkloadManifest) -> str:
    """Render the per-corpus licence record shipped beside the clips."""
    lines = [
        "# Spanish Workload Set — corpus licences",
        "",
        f"Preset: `{manifest.preset}`",
        "",
        "| Corpus | Role | Licence | Source | Items |",
        "|---|---|---|---|---:|",
    ]
    for block in manifest.corpora:
        lines.append(
            f"| {block.name} | {block.role} | "
            f"[{block.licence}]({block.licence_url}) | "
            f"[{block.hf_dataset}]({block.homepage}) | {block.materialised} |"
        )
    lines += [
        "",
        "All corpora above are single-speaker, so this workload set validates ASR",
        "quality only and yields no meaningful DER. Diarization quality is measured",
        "on AMI.",
        "",
    ]
    return "\n".join(lines)


def _manifest_is_current(manifest: dict[str, Any], preset: SpanishPreset, count: int) -> bool:
    if manifest.get("preset") != preset.key or manifest.get("items_per_corpus") != count:
        return False
    present = {block["key"] for block in manifest.get("corpora", [])}
    return present == set(preset.corpora)


def materialize_spanish_workload_set(
    preset_key: str,
    root: Path,
    *,
    items_per_corpus: int | None = None,
    no_download: bool = False,
) -> Path:
    """Materialise a Spanish preset and return its ``--clips-dir``.

    Idempotent: an existing manifest for the same preset and item count is
    reused without re-fetching. Audio and Reference STM files are written as
    ``(<item_id>.wav, <item_id>.ref.stm)`` pairs, and licences are recorded in
    ``LICENCES.md`` and ``corpora.json``.

    Args:
        preset_key: A key of :data:`SPANISH_PRESETS`.
        root: Directory holding one subdirectory per materialised preset.
        items_per_corpus: Override the preset's item count.
        no_download: Fail instead of fetching anything over the network.

    Returns:
        The clips directory for the preset.

    """
    preset = resolve_spanish_preset(preset_key)
    count = items_per_corpus or preset.items_per_corpus
    out_dir = root / preset.key
    manifest_path = out_dir / MANIFEST_NAME

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _manifest_is_current(manifest, preset, count):
            return out_dir

    if no_download:
        raise RuntimeError(
            f"Spanish preset {preset.key!r} is not materialised under {out_dir} "
            "(and --no-download was set). Re-run without --no-download."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = SpanishWorkloadManifest(
        preset=preset.key,
        description=preset.description,
        items_per_corpus=count,
    )
    for key in preset.corpora:
        corpus = SPANISH_CORPORA[key]
        entries = _materialize_corpus(corpus, out_dir, count)
        manifest.corpora.append(_corpus_manifest(corpus, count, entries))

    manifest_path.write_text(
        json.dumps(asdict(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / LICENCES_NAME).write_text(render_licences(manifest), encoding="utf-8")
    return out_dir
