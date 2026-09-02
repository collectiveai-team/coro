"""Shared subword-to-word token grouping for ASR backends.

Both the ``onnx-asr`` and ``nemo`` backends decode transducer/CTC models that
emit *subword* (SentencePiece-style) tokens with one timestamp per token, but
the project-owned ``TranscriptToken`` contract is word-level (one ``start``/
``end`` pair per word). This module holds that shared reconstruction logic so
neither backend imports the other's internals -- each backend's module stays
self-contained and this one has no ASR-runtime dependency of its own (no
``onnxruntime``, no ``torch``, no ``nemo_toolkit``).

A token starts a new word when it is prefixed by a space or the SentencePiece
``\u2581`` marker; otherwise it continues the current word.
"""

from __future__ import annotations

from coro.core.models import TranscriptToken

# SentencePiece word-start marker used by NeMo models (Parakeet/Canary/GigaAM).
SP_SPACE = "\u2581"
# Synthesised duration (seconds) for the final word, which has no following token.
LAST_WORD_PAD = 0.2


def group_subwords(
    tokens: list[str],
    timestamps: list[float],
    logprobs: list[float] | None,
) -> list[dict]:
    """Group subword tokens into words on the word-start marker.

    A token starts a new word when it is prefixed by a space or the SentencePiece
    ``\u2581`` marker; otherwise it continues the current word (this also keeps
    leading punctuation tokens, e.g. ``","``, attached to the preceding word).

    Args:
        tokens: Subword token strings (a leading space or ``\u2581`` prefixes a new word).
        timestamps: One emission time per token, parallel to ``tokens``.
        logprobs: One log-probability per token, parallel to ``tokens``, or None.

    Returns:
        List of ``{"text": str, "start": float, "logprobs": list[float]}`` word groups,
        where ``text`` keeps the leading space of a word start so that
        ``"".join(token.text ...)`` in the response builder reconstructs spaced text.

    """
    groups: list[dict] = []
    n = len(tokens)
    for i in range(n):
        token = tokens[i]
        if not token:
            continue
        ts = float(timestamps[i])
        lp = float(logprobs[i]) if logprobs is not None and i < len(logprobs) else None
        is_word_start = token.startswith((" ", SP_SPACE))
        piece = token.replace(SP_SPACE, " ")
        if is_word_start or not groups:
            groups.append({"text": piece, "start": ts, "logprobs": []})
        else:
            groups[-1]["text"] += piece
        if lp is not None:
            groups[-1]["logprobs"].append(lp)
    return groups


def words_from_text(text: str, start: float, span_end: float | None) -> list[TranscriptToken]:
    """Synthesise word-level tokens from a text-only result (no token timestamps).

    Used when a decoder exposes ``text`` but no per-token timestamps (e.g.
    onnx-asr's Whisper path, or a NeMo hypothesis with no timestamp data), so
    word timings are spread evenly across ``[start, span_end]`` (the VAD
    segment span, or the whole clip). Each word keeps a leading space so the
    response builder's ``"".join(...)`` reconstructs spaced text.
    """
    words = text.split()
    if not words:
        return []
    step = (span_end - start) / len(words) if span_end and span_end > start else LAST_WORD_PAD
    out: list[TranscriptToken] = []
    for i, word in enumerate(words):
        word_start = start + i * step
        word_end = start + (i + 1) * step
        out.append(
            TranscriptToken(
                start=round(word_start, 3),
                end=round(max(word_end, word_start), 3),
                text=" " + word,
                probability=None,
            )
        )
    return out
