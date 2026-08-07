"""Transcript text schemas the MeetEval Metric Set is reported under.

A WER is only meaningful next to the text conventions it was computed with, so
the Quality Benchmark scores every metric under more than one schema:

``unpunctuated``
    Strips ASCII punctuation and collapses whitespace. Case, contractions and
    fillers survive. Coro's long-standing variant, kept so historical runs stay
    comparable. Named for what it does: it was once called ``normalized``, which
    invited the reading that it applied the conventions published numbers use —
    it does not, it does not even lowercase.

``whisper_english``
    The Whisper ``EnglishTextNormalizer`` conventions, used by the Open ASR
    Leaderboard and therefore by the published numbers on the model cards of the
    ASR backends coro runs. Lowercases, removes punctuation, expands
    contractions, standardises numbers and spellings, and deletes the fillers
    ``um``/``uh``/``hmm``/``mm``. Named for the transform rather than for the
    leaderboard that popularised it, because the leaderboard can change its
    conventions and because the name should show this is English-only — it
    cannot be applied to coro's Spanish deployment.

Only this module knows how text is rewritten; ``bench.quality`` scores whatever
schemas the registry lists.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
import re
import string

TextSchema = Callable[[str], str]

_PUNCTUATION_TRANS = str.maketrans("", "", string.punctuation)


def strip_punctuation(text: str) -> str:
    """Remove ASCII punctuation and collapse repeated whitespace."""
    no_punctuation = text.translate(_PUNCTUATION_TRANS)
    return re.sub(r"\s+", " ", no_punctuation).strip()


@lru_cache(maxsize=1)
def english_normalizer():
    """Return the shared Whisper English normalizer.

    Construction reads a spelling-mapping table, so it is cached rather than
    repeated per segment.
    """
    from whisper_normalizer.english import EnglishTextNormalizer

    return EnglishTextNormalizer()


def whisper_english_text(text: str) -> str:
    """Normalize text the way published English ASR results are scored."""
    return english_normalizer()(text).strip()


# Report order, not an arbitrary mapping: schemas are printed in this sequence.
TEXT_SCHEMAS: tuple[tuple[str, TextSchema], ...] = (
    ("unpunctuated", strip_punctuation),
    ("whisper_english", whisper_english_text),
)


def write_schema_stm(src: Path, dst: Path, normalize: TextSchema) -> None:
    """Copy an STM, rewriting only its transcript column with ``normalize``.

    Segments whose text normalizes to nothing — a reference turn of pure
    backchannel, say — are dropped rather than written as empty, which scorers
    would otherwise treat as a real (empty) turn.
    """
    lines: list[str] = []
    for line in src.read_text().splitlines():
        parts = line.strip().split(maxsplit=5)
        if len(parts) < 6:
            continue
        text = normalize(parts[5])
        if not text:
            continue
        lines.append(" ".join([*parts[:5], text]))
    dst.write_text("\n".join(lines) + ("\n" if lines else ""))
