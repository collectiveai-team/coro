"""Vendor-shaped transcription response projections.

Opt-in ``response_format`` values that expose per-word speaker labels, which no
OpenAI-compatible format has a slot for. Each module owns one vendor's boundary
models and the projection that fills them. See ADR 0010 for the fidelity policy
these projections are held to.
"""

from coro.api.vendor.assemblyai import AssemblyAIResponse, assemblyai_response
from coro.api.vendor.deepgram import DeepgramResponse, deepgram_response
from coro.api.vendor.utterances import Utterance, group_words_into_utterances

__all__ = [
    "AssemblyAIResponse",
    "DeepgramResponse",
    "Utterance",
    "assemblyai_response",
    "deepgram_response",
    "group_words_into_utterances",
]
