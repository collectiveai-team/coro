"""Canonical BCP-47 spelling of the language hint every request surface accepts."""

from __future__ import annotations

import pytest

from coro.core.language import canonical_language


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("es", "es"),
        ("ES", "es"),
        ("es-AR", "es-AR"),
        ("es-ar", "es-AR"),
        ("ES_ar", "es-AR"),
        ("  es-US  ", "es-US"),
        ("zh-hans-cn", "zh-Hans-CN"),
        ("es-419", "es-419"),
    ],
)
def test_equivalent_spellings_share_one_canonical_form(raw, expected):
    assert canonical_language(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_absent_or_blank_language_is_none(raw):
    assert canonical_language(raw) is None
