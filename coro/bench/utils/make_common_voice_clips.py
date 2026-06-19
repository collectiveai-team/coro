#!/usr/bin/env python3

r"""Materialize Common Voice clips into a WER benchmark ``--clips-dir``.

Common Voice (https://commonvoice.mozilla.org, CC0) ships single-speaker read
speech with human-validated transcripts — a free target for measuring ASR WER
(it has no speaker turns, so DER is not meaningful). This reads a language
split's ``<split>.tsv`` (``path`` + ``sentence`` columns), transcodes each
referenced ``clips/<path>`` to 16 kHz mono WAV, and writes a single-speaker
reference STM, producing the (<stem>.wav, <stem>.ref.stm) pairs --clips-dir
consumes.

    python -m coro.bench.utils.make_common_voice_clips \
        --cv-dir cv-corpus-17.0/es --split test --limit 25 --out-dir cv-es-clips

Requires ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import wave
from pathlib import Path

from coro.bench.stm import hyp_segments_to_stm


def read_cv_rows(tsv_path: Path, *, limit: int | None = None) -> list[tuple[str, str]]:
    """Return ``(path, sentence)`` pairs from a Common Voice split TSV.

    Rows with an empty ``path`` or ``sentence`` are skipped. ``limit`` caps the
    number of returned rows (None = all).
    """
    rows: list[tuple[str, str]] = []
    with tsv_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for record in reader:
            clip = (record.get("path") or "").strip()
            sentence = (record.get("sentence") or "").strip()
            if not clip or not sentence:
                continue
            rows.append((clip, sentence))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def transcode_to_wav(src: Path, dst: Path) -> None:
    """Transcode any audio file to a 16 kHz mono WAV via ffmpeg."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)],
        check=True,
    )


def wav_duration_seconds(path: Path) -> float:
    """Return the duration of a WAV file in seconds."""
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
    return frames / rate if rate else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize Common Voice clips into a WER --clips-dir.",
    )
    parser.add_argument(
        "--cv-dir", type=Path, required=True, help="Common Voice language dir (has clips/ + *.tsv)."
    )
    parser.add_argument("--split", default="test", help="TSV split name (default: test).")
    parser.add_argument("--limit", type=int, default=None, help="Max clips (default: all).")
    parser.add_argument("--out-dir", type=Path, default=Path("cv-clips"))
    args = parser.parse_args()

    tsv_path = args.cv_dir / f"{args.split}.tsv"
    clips_dir = args.cv_dir / "clips"
    rows = read_cv_rows(tsv_path, limit=args.limit)

    written = 0
    for clip, sentence in rows:
        stem = Path(clip).stem
        wav_dst = args.out_dir / f"{stem}.wav"
        stm_dst = args.out_dir / f"{stem}.ref.stm"

        transcode_to_wav(clips_dir / clip, wav_dst)
        duration = wav_duration_seconds(wav_dst)
        stm_text = hyp_segments_to_stm(
            [{"start": 0.0, "end": duration, "text": sentence, "speaker": "1"}],
            stem,
        )
        stm_dst.write_text(stm_text, encoding="utf-8")
        written += 1

    print(f"wrote {written} clip(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
