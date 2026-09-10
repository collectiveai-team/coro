"""Language Drift Metric — issue #64's acceptance criterion.

The problem this measures: with no forced-language port, an ASR model
auto-detects language independently per 30 s window
(``coro.pipelines.windowing.DEFAULT_WINDOW_SECONDS``). On monolingual Spanish
audio, closely related languages (Portuguese, Italian, Catalan) can be
misdetected in a single window, producing hypothesis text that silently
switches language mid-transcript even though the audio never does. A 30-minute
recording is ~60 independent language decisions, so
``P(>=1 drift) = 1 - (1-p)^N`` approaches certainty as audio gets longer — the
standard short-form Spanish benchmark corpora (FLEURS, VoxPopuli, MLS test
splits) cannot exhibit this because every clip fits inside one window.

Two complementary signals, both reference-free at the word-window level:

1. **Sliding-window language-ID** over the raw hypothesis text. Single-word
   langid is unreliable (proper nouns, loanwords, short function words all
   look foreign in isolation), so windows of ``window_size`` words are
   classified together. A window is "foreign" when its detected language is
   not Spanish. This gives ``foreign_word_rate`` and a *positional profile* —
   foreign rate as a function of position in the transcript — which is the
   view that makes drift visible as a step function rather than noise.
2. **Reference-anchored substitution rate**, for when a reference transcript
   is available (as it is for a benchmark corpus). Word-level alignment
   (``kaldialign``, already used transitively for corpus tooling) separates
   *substitutions* (hypothesis word differs from reference) from *matches*.
   A hypothesis word that happens to be a foreign proper noun but was
   transcribed *correctly* is a match, not a substitution, so it does not
   inflate the drift signal — only look at ``foreign_substitution_rate`` to
   avoid penalising legitimate foreign-language content (place names, loan
   words) that both reference and hypothesis agree on.

A small set of **orthographic markers** — letter sequences rare-to-impossible
in Spanish (``ção``, ``ã``, ``lh`` for Portuguese; ``gli``, ``zione`` for
Italian; ``ny``, ``·l`` for Catalan) — corroborates the langid library
independently: low recall, near-zero false positives on genuine Spanish text.

A fourth signal, :func:`function_word_hits`, exists for a specific blind spot
in the window-level approach: a single injected function word (e.g. an
English "in" standing in for "en") does not shift a whole 7-word window's
langid verdict, since six of seven words are still correct Spanish, so
``classify_windows`` misses it while a per-word check against a curated,
collision-free closed-class word list catches it directly.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
import re

import kaldialign
from lingua import Language, LanguageDetectorBuilder

from coro.bench.normalizers.basic import BasicTextNormalizer

EPS = "<eps>"
"""Alignment gap symbol passed to ``kaldialign.align``."""

DEFAULT_WINDOW_WORDS = 7
"""Sliding-window size for language-ID. 5-10 words is the range single-word
langid cannot discriminate at but a full window reliably can."""

DETECT_LANGUAGES = (
    Language.SPANISH,
    Language.PORTUGUESE,
    Language.ITALIAN,
    Language.CATALAN,
    Language.FRENCH,
    Language.ENGLISH,
)
"""Candidate languages for the detector.

Restricting the candidate set (rather than the library's full ~75 languages)
is what gives ``lingua`` its discrimination power between close Romance
languages — the mechanism this metric depends on. Galician is the audio's
next-closest neighbour after Portuguese but is not one of ``lingua``'s
supported languages as of 2.2.0; Portuguese is a reasonable proxy for it
orthographically.
"""

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

_normalize = BasicTextNormalizer(remove_diacritics=False)
"""Lowercases and strips punctuation while keeping diacritics, which the
Portuguese/Italian/Catalan orthographic markers above depend on."""


@lru_cache(maxsize=1)
def _detector():
    """Build (once) the language detector restricted to ``DETECT_LANGUAGES``."""
    return LanguageDetectorBuilder.from_languages(*DETECT_LANGUAGES).build()


def _tokenize(text: str) -> list[str]:
    """Split normalized text on whitespace, dropping empty tokens."""
    return [w for w in _normalize(text).split(" ") if w]


@dataclass(frozen=True)
class WindowVerdict:
    """Language-ID verdict for one sliding word-window of hypothesis text."""

    start_word: int
    """Index of this window's first word in the full hypothesis word list."""

    end_word: int
    """Index one past this window's last word (half-open, like ``range``)."""

    language: str | None
    """Detected language's ISO 639-1 code, or ``None`` if undetermined."""

    is_foreign: bool
    """Whether the detected language is not Spanish."""

    @property
    def word_count(self) -> int:
        return self.end_word - self.start_word


@dataclass(frozen=True)
class DriftReport:
    """Full language-drift measurement for one hypothesis transcript."""

    windows: list[WindowVerdict] = field(default_factory=list)
    total_words: int = 0
    foreign_word_rate: float = 0.0
    """Fraction of hypothesis words that fall in a foreign-classified window."""

    language_histogram: Counter[str] = field(default_factory=Counter)
    """Count of foreign windows by detected language code."""

    foreign_substitution_rate: float | None = None
    """Fraction of reference-aligned *substitution* words whose surrounding
    window was classified foreign. ``None`` when no reference was supplied."""

    substitution_count: int = 0
    """Total substitution words found against the reference (0 if no
    reference was supplied)."""

    orthographic_marker_hits: Counter[str] = field(default_factory=Counter)
    """Raw count of each language's orthographic markers found verbatim in
    the (normalized) hypothesis text — independent of the langid library."""

    function_word_hits: Counter[str] = field(default_factory=Counter)
    """Count of individual foreign function-word hits by language, from
    :func:`function_word_hits`. Catches single-word injections that a whole
    ``window_size``-word langid window is too coarse to see."""

    function_word_rate: float = 0.0
    """Fraction of hypothesis words that are a foreign function word."""


def classify_windows(
    hyp_words: list[str], *, window_size: int = DEFAULT_WINDOW_WORDS
) -> list[WindowVerdict]:
    """Run language-ID over non-overlapping windows of ``hyp_words``.

    The final window absorbs any remainder shorter than ``window_size`` rather
    than being classified alone, since a 1-2 word tail window is exactly the
    unreliable single-word case this metric exists to avoid.
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if not hyp_words:
        return []

    detector = _detector()
    windows: list[WindowVerdict] = []
    start = 0
    n = len(hyp_words)
    while start < n:
        end = start + window_size
        if n - end < window_size:  # remainder would be a short trailing window
            end = n
        chunk = " ".join(hyp_words[start:end])
        detected = detector.detect_language_of(chunk)
        code = detected.iso_code_639_1.name.lower() if detected else None
        windows.append(
            WindowVerdict(
                start_word=start,
                end_word=end,
                language=code,
                is_foreign=code is not None and code != "es",
            )
        )
        start = end
    return windows


def foreign_word_rate(windows: list[WindowVerdict], total_words: int) -> float:
    """Fraction of ``total_words`` covered by a foreign-classified window."""
    if total_words == 0:
        return 0.0
    foreign_words = sum(w.word_count for w in windows if w.is_foreign)
    return foreign_words / total_words


def language_histogram(windows: list[WindowVerdict]) -> Counter[str]:
    """Count foreign windows by detected language code."""
    return Counter(w.language for w in windows if w.is_foreign and w.language)


def orthographic_marker_hits(text: str) -> Counter[str]:
    """Count verbatim occurrences of each language's markers in ``text``.

    Case-insensitive; diacritics preserved (matches ``ORTHOGRAPHIC_MARKERS``'
    accented entries literally, so callers should not strip diacritics first).
    """
    lowered = text.lower()
    hits: Counter[str] = Counter()
    for lang, markers in ORTHOGRAPHIC_MARKERS.items():
        count = sum(len(re.findall(re.escape(marker), lowered)) for marker in markers)
        if count:
            hits[lang] = count
    return hits


def function_word_hits(hyp_words: list[str]) -> list[tuple[int, str, str]]:
    """Find hypothesis words that are unambiguous foreign function words.

    Complementary to :func:`classify_windows`: a lone injected function word
    does not shift a whole window's langid verdict when the other
    ``window_size - 1`` words are correct Spanish, so this checks words
    individually against :data:`FOREIGN_FUNCTION_WORDS` instead of
    statistically classifying a window.

    Returns:
        List of ``(word_index, word, language_code)`` triples, indices being
        positions in the same tokenized word list :func:`classify_windows`
        consumes.

    """
    hits: list[tuple[int, str, str]] = []
    for idx, word in enumerate(hyp_words):
        for lang, words in FOREIGN_FUNCTION_WORDS.items():
            if word in words:
                hits.append((idx, word, lang))
                break
    return hits


def substitution_words(reference: str, hypothesis: str) -> list[tuple[int, str]]:
    """Word-align ``hypothesis`` to ``reference`` and return substitutions only.

    Insertions and deletions are excluded: an inserted foreign word with no
    reference counterpart is a different failure mode (hallucination) from a
    substitution (the model chose a foreign word *instead of* the correct
    Spanish one), and a deletion has no hypothesis word to attribute language
    to at all. Correct matches (including correctly-transcribed foreign proper
    nouns) are excluded by construction.

    Returns:
        List of ``(hyp_word_index, hyp_word)`` pairs, indices being positions
        in the tokenized (whitespace-split, gap-free) hypothesis word list —
        i.e. valid indices into the same list ``classify_windows`` consumes.

    """
    ref_words = _tokenize(reference)
    hyp_words = _tokenize(hypothesis)
    pairs = kaldialign.align(ref_words, hyp_words, EPS)

    subs: list[tuple[int, str]] = []
    hyp_idx = 0
    for ref_w, hyp_w in pairs:
        if hyp_w != EPS:
            if ref_w != EPS and ref_w != hyp_w:
                subs.append((hyp_idx, hyp_w))
            hyp_idx += 1
    return subs


def _window_at(windows: list[WindowVerdict], word_idx: int) -> WindowVerdict | None:
    for w in windows:
        if w.start_word <= word_idx < w.end_word:
            return w
    return None


def compute_drift_report(
    hypothesis: str,
    *,
    reference: str | None = None,
    window_size: int = DEFAULT_WINDOW_WORDS,
) -> DriftReport:
    """Compute the full language-drift report for one hypothesis transcript.

    Args:
        hypothesis: The ASR output text to inspect.
        reference: Optional ground-truth text. When given, adds
            ``foreign_substitution_rate`` — reference-anchored, robust to
            correctly-transcribed foreign proper nouns.
        window_size: Sliding-window size in words for language-ID.

    Returns:
        A :class:`DriftReport` with the positional profile (``windows``),
        aggregate rate, language histogram, and orthographic sanity check.

    """
    hyp_words = _tokenize(hypothesis)
    windows = classify_windows(hyp_words, window_size=window_size)
    fn_hits = function_word_hits(hyp_words)

    foreign_sub_rate = None
    n_subs = 0
    if reference is not None:
        subs = substitution_words(reference, hypothesis)
        n_subs = len(subs)
        if n_subs:
            foreign = sum(
                1
                for idx, _word in subs
                if (w := _window_at(windows, idx)) is not None and w.is_foreign
            )
            foreign_sub_rate = foreign / n_subs
        else:
            foreign_sub_rate = 0.0

    return DriftReport(
        windows=windows,
        total_words=len(hyp_words),
        foreign_word_rate=foreign_word_rate(windows, len(hyp_words)),
        language_histogram=language_histogram(windows),
        foreign_substitution_rate=foreign_sub_rate,
        substitution_count=n_subs,
        orthographic_marker_hits=orthographic_marker_hits(hypothesis),
        function_word_hits=Counter(lang for _idx, _word, lang in fn_hits),
        function_word_rate=len(fn_hits) / len(hyp_words) if hyp_words else 0.0,
    )
