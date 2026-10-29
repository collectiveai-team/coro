"""STM text helpers for Quality Benchmark scoring.

Reads Reference and Hypothesis STM files for the checks around MeetEval: the
Normalized Metric Lane copy, diarization-only detection and the
degenerate-diarization check.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from coro.bench.models.quality import DiarizationSanity
from coro.bench.normalizers import BasicTextNormalizer
from coro.bench.stm import DIARIZATION_ONLY_TEXT

# Diacritic-preserving by construction: `remove_diacritics` is left at its
# False default because stripping diacritics collapses distinct Spanish words
# (esta/está, ano/año) and silently understates error. See ADR 0011.
BASIC_NORMALIZER = BasicTextNormalizer()


def normalize_transcript_text(text: str) -> str:
    """Apply the Normalized Metric Lane protocol to one transcript text field.

    Uses the vendored Whisper Basic Text Normalizer: bracketed and parenthesised
    content removed, lowercased, symbols and punctuation mapped to spaces,
    whitespace collapsed, and diacritics preserved.

    Args:
        text: Transcript text as it appears in an STM line.

    Returns:
        The normalized transcript text.

    """
    return BASIC_NORMALIZER(text)


def _stm_lines(path: Path) -> Iterator[str]:
    """Yield STM lines from a file, or from every ``.stm`` in a directory.

    meeteval's multifile API accepts a directory of per-session STMs, so the
    checks around it have to as well — passing one used to raise
    ``IsADirectoryError`` from inside :func:`coro.bench.quality.score_item`,
    which is why ``coro-bench-diar``'s combined-across-sessions DER could never
    run.
    """
    paths = sorted(path.glob("*.stm")) if path.is_dir() else [path]
    for stm_path in paths:
        yield from stm_path.read_text().splitlines()


def is_diarization_only_stm(path: Path) -> bool:
    """Return True when every STM line's text is the diarization-only sentinel.

    Such references (e.g. VoxConverse RTTM converted via rttm_to_stm) carry
    speaker turns but no transcript, so only DER is meaningful. Accepts either
    a single STM or a directory of them.
    """
    saw_line = False
    for line in _stm_lines(path):
        parts = line.strip().split(maxsplit=5)
        if len(parts) < 6:
            continue
        saw_line = True
        if parts[5].strip() != DIARIZATION_ONLY_TEXT:
            return False
    return saw_line


def _count_stm_speakers(path: Path) -> int:
    """Count distinct speaker labels (column 3) in an STM file or directory."""
    speakers: set[str] = set()
    for line in _stm_lines(path):
        parts = line.strip().split(maxsplit=5)
        if len(parts) >= 3:
            speakers.add(parts[2])
    return len(speakers)


def diarization_sanity(ref_stm_path: Path, hyp_stm_path: Path) -> DiarizationSanity:
    """Flag degenerate diarization: a single hyp speaker against multi-speaker ref.

    A diarization-invariant metric like ORC-WER stays low even when every word
    collapses onto one speaker, so this check surfaces the failure the WER
    headline would otherwise hide.
    """
    ref_speakers = _count_stm_speakers(ref_stm_path)
    hyp_speakers = _count_stm_speakers(hyp_stm_path)
    degenerate = hyp_speakers <= 1 and ref_speakers > 1
    return DiarizationSanity(
        ref_speakers=ref_speakers,
        hyp_speakers=hyp_speakers,
        degenerate=degenerate,
    )


def write_normalized_stm(src: Path, dst: Path) -> int:
    """Write an STM file with only the transcript text field normalized.

    Applied identically to the Reference STM and the Hypothesis STM so the
    Normalized Metric Lane compares like with like. The source file is never
    modified: normalization happens on a scratch copy at scoring time, so the
    Raw Metric Lane stays recoverable from the artifacts.

    Lines whose text normalizes to nothing (e.g. a segment that was only a
    ``[non-speech]`` marker) carry no words to score and are dropped.

    Args:
        src: STM file to read.
        dst: STM file to write.

    Returns:
        The number of scoreable lines written.

    """
    lines: list[str] = []
    for line in src.read_text().splitlines():
        parts = line.strip().split(maxsplit=5)
        if len(parts) < 6:
            continue
        text = normalize_transcript_text(parts[5])
        if not text:
            continue
        lines.append(" ".join([*parts[:5], text]))
    dst.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)
