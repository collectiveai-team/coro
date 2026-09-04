r"""Build a multi-language calibration corpus for static-QDQ encoder quantization.

See ``README.md`` in this package for the full write-up (sources, scope
notes, and what changed from the original `.tmp/` script). This docstring
stays short on purpose -- it is what ``--help`` shows.

Usage:
    uv run --extra recipes --extra cpu -m coro.recipes.calibration_corpus
    uv run --extra recipes --extra cpu -m coro.recipes.calibration_corpus \\
        --out-dir recipe-artifacts/calibration_corpus --per-language-target 8
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
from pathlib import Path

import soundfile as sf

from coro.recipes.paths import RECIPE_ARTIFACTS_DIR

MIN_S, MAX_S = 3.0, 15.0
# (lang_code, hf_config) pairs streamed remotely -- Spanish is harvested
# separately from a local FLEURS snapshot (see harvest_local_fleurs_es).
REMOTE_LANGUAGES = [("en", "en_us"), ("fr", "fr_fr"), ("de", "de_de"), ("pt", "pt_br")]


def clip_duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def _resolve_fleurs_es_snapshot() -> Path:
    """Resolve the FLEURS es_419 test shard via huggingface_hub, downloading if needed.

    Portable across hosts/environments -- no hardcoded snapshot hash, unlike
    the original script this recipe replaces.
    """
    from huggingface_hub import snapshot_download

    snapshot_dir = Path(
        snapshot_download(
            "google/fleurs",
            repo_type="dataset",
            allow_patterns=["data/es_419/audio/test.tar.gz"],
        )
    )
    return snapshot_dir / "data" / "es_419"


def harvest_local_fleurs_es(out_dir: Path, target: int) -> list[dict]:
    """Extract a handful of clips from the FLEURS es_419 test shard.

    Spans a spread of durations without extracting the whole archive to disk.
    """
    fleurs_es_dir = _resolve_fleurs_es_snapshot()
    tar_path = fleurs_es_dir / "audio" / "test.tar.gz"
    out_lang_dir = out_dir / "es"
    out_lang_dir.mkdir(parents=True, exist_ok=True)
    picked: list[dict] = []
    print(f"scanning {tar_path} for es clips (target={target}, {MIN_S}-{MAX_S}s)...")
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if len(picked) >= target:
                break
            if not member.isfile() or not member.name.endswith(".wav"):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            data = f.read()
            tmp_path = out_lang_dir / f"_probe_{Path(member.name).name}"
            tmp_path.write_bytes(data)
            dur = clip_duration(tmp_path)
            if MIN_S <= dur <= MAX_S:
                final_path = out_lang_dir / f"es-{Path(member.name).stem}.wav"
                tmp_path.rename(final_path)
                picked.append(
                    {
                        "path": str(final_path),
                        "lang": "es",
                        "duration_s": round(dur, 2),
                        "source": "fleurs-es_419",
                    }
                )
            else:
                tmp_path.unlink()
    print(f"  picked {len(picked)} es clips, durations={[p['duration_s'] for p in picked]}")
    return picked


def harvest_remote_fleurs(
    lang_code: str, hf_config: str, out_dir: Path, target: int, overfetch: int = 60
) -> list[dict]:
    from coro.bench.utils.audio_clips import transcode_bytes_to_wav
    from coro.bench.utils.hf_parquet import iter_parquet_rows, resolve_shard_urls

    out_lang_dir = out_dir / lang_code
    out_lang_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"streaming google/fleurs config={hf_config!r} for lang={lang_code!r} (target={target})..."
    )
    t0 = time.time()
    urls = resolve_shard_urls("google/fleurs", hf_config, "test")
    picked: list[dict] = []
    for row in iter_parquet_rows(urls, limit=overfetch, columns=["id", "audio"]):
        if len(picked) >= target:
            break
        audio = row.get("audio") or {}
        data = audio.get("bytes")
        if not data:
            continue
        wav_path = out_lang_dir / f"{lang_code}-{row['id']}.wav"
        transcode_bytes_to_wav(data, wav_path)
        dur = clip_duration(wav_path)
        if MIN_S <= dur <= MAX_S:
            picked.append(
                {
                    "path": str(wav_path),
                    "lang": lang_code,
                    "duration_s": round(dur, 2),
                    "source": f"fleurs-{hf_config}",
                }
            )
        else:
            wav_path.unlink()
    print(
        f"  picked {len(picked)} {lang_code} clips in {time.time() - t0:.1f}s, "
        f"durations={[p['duration_s'] for p in picked]}"
    )
    return picked


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(line_buffering=True)  # pyrefly: ignore[missing-attribute]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=RECIPE_ARTIFACTS_DIR / "calibration_corpus",
        help="Directory to write <lang>-<id>.wav clips and the manifest into.",
    )
    parser.add_argument(
        "--per-language-target", type=int, default=8, help="Clips to harvest per language."
    )
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    manifest += harvest_local_fleurs_es(args.out_dir, args.per_language_target)

    for lang_code, hf_config in REMOTE_LANGUAGES:
        try:
            manifest += harvest_remote_fleurs(
                lang_code, hf_config, args.out_dir, args.per_language_target
            )
        except Exception as exc:
            print(f"  WARNING: failed to fetch {lang_code} ({hf_config}): {exc}")

    manifest_path = args.out_dir / "calibration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    total_s = sum(m["duration_s"] for m in manifest)
    by_lang: dict[str, int] = {}
    for m in manifest:
        by_lang[m["lang"]] = by_lang.get(m["lang"], 0) + 1
    print(f"\nTotal clips: {len(manifest)}, total duration: {total_s:.1f}s, by language: {by_lang}")
    print(f"Manifest written to {manifest_path}")
