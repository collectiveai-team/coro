"""The ``response_format`` vocabulary for ``POST /v1/audio/transcriptions``.

Separate from the route so the incremental renderers can name a format without
importing the router, which imports them.
"""

from __future__ import annotations

from enum import StrEnum


class ResponseFormat(StrEnum):
    """All OpenAI response_format values this server recognises.

    Exactly the values OpenAI's own ``AudioResponseFormat`` defines, minus the
    ones it defines that coro does not implement — the text output formats are
    recognised so they fail with an OpenAI-style 400 (param ``response_format``)
    rather than a generic validation error.

    coro-invented spellings are deliberately absent. ``json_verbose`` and
    ``dirized_json`` were once accepted as typo-tolerant aliases; a misspelling
    that silently succeeds trains clients to depend on it, and the server can
    never afterwards tell a typo from an intent (ADR 0018).
    """

    JSON = "json"
    VERBOSE_JSON = "verbose_json"
    DIARIZED_JSON = "diarized_json"

    # Unsupported OpenAI formats (recognised but not implemented → 400)
    TEXT = "text"
    SRT = "srt"
    VTT = "vtt"
    TSV = "tsv"


# JSON-like formats this server actually renders (vs. the recognised-but-
# unsupported text outputs above).
JSON_LIKE_FORMATS = frozenset(
    {
        ResponseFormat.JSON,
        ResponseFormat.VERBOSE_JSON,
        ResponseFormat.DIARIZED_JSON,
    }
)
