"""Canonical spelling of the BCP-47 language hint carried by every request surface."""

from __future__ import annotations


def canonical_language(language: str | None) -> str | None:
    """Return the BCP-47 canonical spelling of a language hint.

    Surfaces receive the same hint spelled many ways (``ES``, ``es_ar``,
    ``" es-AR "``). Backends match it against vocabularies keyed in canonical
    form (``es``, ``es-US``, ``zh-Hans``), so the surface fixes the spelling once
    and each backend only decides which supported language that maps to. Only
    the spelling changes: nothing here validates that the tag names a real
    language, which stays the backend's call.

    Args:
        language: Language hint as supplied by a client, or None.

    Returns:
        The primary subtag lowercased, a two-letter subtag (region) uppercased,
        a four-letter subtag (script) title-cased and any other subtag
        lowercased, joined with hyphens; None when absent or blank.

    """
    if language is None:
        return None
    tag = language.strip().replace("_", "-")
    if not tag:
        return None
    primary, *subtags = tag.split("-")
    parts = [primary.lower()]
    for subtag in subtags:
        if len(subtag) == 2 and subtag.isalpha():
            parts.append(subtag.upper())
        elif len(subtag) == 4 and subtag.isalpha():
            parts.append(subtag.title())
        else:
            parts.append(subtag.lower())
    return "-".join(parts)
