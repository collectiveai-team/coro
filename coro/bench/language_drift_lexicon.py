"""Spanish-collision-checked word lists behind the language-drift metric.

:mod:`coro.bench.language_drift` reads these tables; they are kept apart
because each entry is a reviewed linguistic decision, not code.
"""

from __future__ import annotations

ORTHOGRAPHIC_MARKERS: dict[str, tuple[str, ...]] = {
    "pt": ("ção", "ções", "ã", "õ", "lh"),
    "it": ("gli", "zione"),
    "ca": ("ny", "·l"),
}
"""Letter sequences rare-to-impossible in Spanish, grouped by the language
they mark. A zero-dependency sanity check on the langid library's verdicts —
low recall by design (real Portuguese/Italian/Catalan text need not contain
any of these), but a false positive here is a strong independent signal."""

FOREIGN_FUNCTION_WORDS: dict[str, frozenset[str]] = {
    "en": frozenset(
        {
            "the",
            "and",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "at",
            "in",
            "on",
            "of",
            "for",
            "with",
            "from",
            "to",
            "by",
            "that",
            "this",
            "these",
            "those",
            "who",
            "whom",
            "which",
            "what",
            "because",
            "although",
            "though",
            "while",
            "when",
            "where",
            "how",
            "not",
            "do",
            "does",
            "did",
            "doing",
            "done",
            "have",
            "has",
            "had",
            "having",
            "will",
            "would",
            "shall",
            "should",
            "can",
            "could",
            "may",
            "might",
            "must",
            "very",
            "also",
            "but",
            "or",
            "if",
            "than",
            "she",
            "they",
            "we",
            "you",
            "your",
            "his",
            "her",
            "their",
            "our",
            "my",
            "it",
            "its",
        }
    ),
}
"""Closed-class (article/auxiliary/conjunction/pronoun) words with no Spanish
reading, checked individually against common Spanish vocabulary to exclude
false-positive-prone overlaps: "no", "as", "he", "so", "si", "mas", "la", "un"
etc. are all valid Spanish words and are deliberately left out, even though
their English near-equivalents ("no", "as", "he", "so") would otherwise be
tempting additions.

English only for now: the mTEDx long-form reproduction (findings.md, session
9) found only English injections, not Portuguese/Italian/Catalan, on a real
22-minute Spanish talk. Extending to those languages needs the same
word-by-word Spanish-collision check done by hand, not just porting a generic
stopword list -- e.g. Portuguese "com"/"mas"/"seu" and Catalan "amb"/"molt"
need individual review against Spanish before they could be added safely.
"""
