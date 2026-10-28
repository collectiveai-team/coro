"""Forced-language resolution and auto-LID probe for the onnx-canary-split backend.

See ``coro/backends/asr/onnx_canary_split.py``'s module docstring (Forced-language
hardening and Auto-LID sections) for the design these functions implement.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from coro.backends.asr.errors import AsrUnsupportedLanguageError
from coro.backends.asr.nemo import resolve_target_language
from coro.cache.fingerprint import normalise_language

# Matches exactly the two-letter language switch tokens in Canary's vocab
# (`<|en|>`, `<|es|>`, ...) -- deliberately narrower than any `<|...|>` token,
# so control tokens (`<|pnc|>`, `<|noitn|>`, `<|emo:...|>`) are excluded and
# the supported set is whatever this checkpoint's vocab actually ships.
_LANGUAGE_TOKEN_RE = re.compile(r"<\|([a-z]{2})\|>")


def resolve_canary_language(language: str | None, language_tokens: dict[str, int]) -> str | None:
    """Resolve a request language to a Canary ``<|xx|>`` vocab code.

    Composed from the two normalisers the codebase already has (no third one
    is added): :func:`coro.cache.fingerprint.normalise_language` handles the
    spelling (strip / lowercase / underscore-to-hyphen), then NeMo's
    :func:`~coro.backends.asr.nemo.resolve_target_language` does the
    exact-then-primary-subtag match against the vocab-derived token map --
    so ``es-US``, ``es_US`` and ``" ES "`` all resolve to ``es``.

    Args:
        language: Request language, or None/blank to leave unresolved (the
            caller applies its fallback).
        language_tokens: The ``code -> token id`` map derived from the loaded
            vocab; also the supported set the error names.

    Returns:
        The base-subtag code whose ``<|xx|>`` token keys the decode prefix.

    Raises:
        AsrUnsupportedLanguageError: If the language matches no ``<|xx|>``
            token in the loaded vocab.

    """
    normalised = normalise_language(language)
    try:
        return resolve_target_language(normalised, language_tokens)
    except ValueError as exc:
        raise AsrUnsupportedLanguageError(normalised, supported_languages=language_tokens) from exc


def _partial_prompt_lid(asr: Any, pcm: bytes) -> str | None:
    """Run NeMo's Canary2 partial-prompt LID probe on one PCM window.

    NeMo's ``Canary2PromptFormatter`` documents a ``user_partial`` role
    (``<|startofcontext|><|startoftranscript|>``) used for exactly two
    decoder steps to retrieve the emotion and source-language tokens.
    ``.scratch/canary-default/findings-lid-probe.md`` (ticket 01) measured
    this against the split-decode INT8/INT8 checkpoint: 100% accuracy across
    64 probed speech windows (48 mTEDx es, 4 each en/fr/de/pt), step 1 always
    ``<|emo:undefined|>`` (no emotion capability), step 2 the language.

    Deliberately runs its own encoder pass rather than reusing a concurrent
    transcription call's -- keeping :meth:`OnnxCanarySplitASRAdapter.detect_language`
    a self-contained call (no `transcribe_pcm` return-type/contract change,
    see that method's docstring) costs one extra encoder pass on the *first*
    one or two windows of a request needing detection, never a sustained
    per-window cost once :class:`~coro.pipelines.windowing.LanguageState`
    goes sticky.

    Args:
        asr: The loaded split-decode ``NemoConformerAED`` subclass instance.
        pcm: Raw PCM s16le 16 kHz mono bytes for one window.

    Returns:
        The emitted language code (e.g. ``"es"``) when the second step's
        token matches this checkpoint's ``<|xx|>`` language-token shape, else
        ``None`` -- not observed on real audio in the probe (even silence
        emitted some language token), but handled defensively.

    """
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    waveforms = audio[None, :]
    waveforms_len = np.array([len(audio)], dtype=np.int64)
    features, features_lens = asr._preprocessor(waveforms, waveforms_len)
    encoder_embeddings, encoder_mask = asr._encode(features, features_lens)

    prefix = np.array(
        [
            [
                asr._tokens[" "],
                asr._tokens["<|startofcontext|>"],
                asr._tokens["<|startoftranscript|>"],
            ]
        ],
        dtype=np.int64,
    )
    shapes = {x.name: x.shape for x in asr._decoder.get_inputs()}
    decoder_mems = np.empty(
        (shapes["decoder_mems"][0], 1, 0, shapes["decoder_mems"][3]), dtype=np.float32
    )

    batch_tokens = prefix
    next_token_id = -1
    for _ in range(2):
        logits, decoder_mems = asr._decode(
            batch_tokens, encoder_embeddings, encoder_mask, decoder_mems
        )
        next_token_id = int(np.argmax(logits[:, -1], axis=-1)[0])
        batch_tokens = np.concatenate([batch_tokens, [[next_token_id]]], axis=-1)

    match = _LANGUAGE_TOKEN_RE.fullmatch(asr._vocab[next_token_id])
    return match.group(1) if match else None
