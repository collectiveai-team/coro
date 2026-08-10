"""Basic Text Normalizer, vendored from OpenAI Whisper.

Source: ``whisper/normalizers/basic.py`` from https://github.com/openai/whisper
(MIT License, Copyright (c) 2022 OpenAI). Vendored verbatim in behaviour rather
than added as a dependency, per ADR 0011: it is ~60 lines of permissively
licensed code and Whisper ships it inside a package that also pulls in Torch.

This is the normalizer OpenAI applied to every non-English language when
reporting Whisper WER. It removes bracketed and parenthesised content,
lowercases, maps symbols and punctuation to spaces, and collapses whitespace.

``remove_diacritics`` defaults to ``False`` and the Quality Benchmark never
sets it: for Spanish, stripping diacritics collapses distinct words
(``esta``/``está``, ``ano``/``año``) and silently understates error.

The English-specific ``EnglishTextNormalizer`` is deliberately not vendored --
its spelling and contraction maps are English-only and would corrupt Spanish
reference and hypothesis text. See ADR 0011.
"""

from __future__ import annotations

import re
import unicodedata

# Non-ASCII letters that are not separated by NFKD normalization, mapped to
# their ASCII expansions. Only consulted when ``remove_diacritics`` is True.
ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}


def remove_symbols_and_diacritics(s: str, keep: str = "") -> str:
    """Replace markers, symbols and punctuation with a space, dropping diacritics.

    Diacritics are dropped by decomposing with NFKD and discarding combining
    marks (Unicode category ``Mn``), plus the manual ``ADDITIONAL_DIACRITICS``
    mappings for letters NFKD does not decompose.

    Args:
        s: Text to clean.
        keep: Characters to pass through untouched.

    Returns:
        The cleaned text.

    """
    return "".join(
        c
        if c in keep
        else ADDITIONAL_DIACRITICS[c]
        if c in ADDITIONAL_DIACRITICS
        else ""
        if unicodedata.category(c) == "Mn"
        else " "
        if unicodedata.category(c)[0] in "MSP"
        else c
        for c in unicodedata.normalize("NFKD", s)
    )


def remove_symbols(s: str) -> str:
    """Replace markers, symbols and punctuation with a space, keeping diacritics.

    NFKC composition keeps accented letters as single letter code points, so
    they survive the category filter that removes combining marks.

    Args:
        s: Text to clean.

    Returns:
        The cleaned text, with diacritics preserved.

    """
    return "".join(
        " " if unicodedata.category(c)[0] in "MSP" else c for c in unicodedata.normalize("NFKC", s)
    )


class BasicTextNormalizer:
    """Whisper's language-agnostic text normalizer.

    Args:
        remove_diacritics: Strip diacritics as well as symbols. Must stay
            ``False`` for Spanish scoring; see the module docstring.
        split_letters: Insert spaces between grapheme clusters, as Whisper does
            for languages without whitespace word boundaries (e.g. Chinese,
            Japanese, Thai). Requires the ``regex`` package.

    """

    def __init__(self, remove_diacritics: bool = False, split_letters: bool = False):
        self.clean = remove_symbols_and_diacritics if remove_diacritics else remove_symbols
        self.split_letters = split_letters

    def __call__(self, s: str) -> str:
        """Normalize one string.

        Args:
            s: Text to normalize.

        Returns:
            The normalized text.

        """
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = self.clean(s).lower()

        if self.split_letters:
            import regex

            s = " ".join(regex.findall(r"\X", s, regex.U))

        # Replace any successive whitespace characters with a single space.
        return re.sub(r"\s+", " ", s).strip()
