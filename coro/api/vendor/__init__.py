"""Vendor-native boundary schemas and response projections.

Each vendor gets its own endpoint implementing that vendor's own contract, so
the OpenAI-compatible surface is never extended with values OpenAI does not
define. Each module owns one vendor's boundary models and the projection that
fills them. See ADR 0010 for the fidelity policy these are held to.
"""

from coro.api.vendor.deepgram import (
    DeepgramErrorResponse,
    DeepgramResponse,
    deepgram_response,
)
from coro.api.vendor.utterances import Utterance, group_words_into_utterances

__all__ = [
    "DeepgramErrorResponse",
    "DeepgramResponse",
    "Utterance",
    "deepgram_response",
    "group_words_into_utterances",
]
