"""The ``response_format`` vocabulary for ``POST /v1/audio/transcriptions``.

Separate from the route so the incremental renderers can name a format without
importing the router, which imports them.
"""

from __future__ import annotations

from enum import StrEnum


class ResponseFormat(StrEnum):
    """All OpenAI response_format values this server recognises.

    JSON-like formats are implemented; ``json_verbose`` is a typo-tolerant alias
    of ``verbose_json``. The text output formats are recognised so they fail with
    an OpenAI-style 400 (param ``response_format``) rather than a generic
    validation error.
    """

    JSON = "json"
    VERBOSE_JSON = "verbose_json"
    JSON_VERBOSE = "json_verbose"
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
        ResponseFormat.JSON_VERBOSE,
        ResponseFormat.DIARIZED_JSON,
    }
)
