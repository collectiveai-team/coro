"""NeMo RNNT hypothesis -> word-level TranscriptToken conversion.

Turns the hypotheses NeMo's ``transcribe()`` returns into the project-owned
``TranscriptToken`` contract for the ``nemo`` backend: frame indices become
seconds, subword pieces become words. See ``coro/backends/asr/nemo.py``'s
module docstring for the timestamp shapes each ``transcribe()`` path yields.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Protocol

from coro.backends.asr.subword_tokens import (
    group_subwords,
    words_from_subword_groups,
    words_from_text,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from coro.core.models import TranscriptToken


class SubwordTokenizer(Protocol):
    """A NeMo ``TokenizerSpec``'s id -> SentencePiece piece mapping."""

    def ids_to_tokens(self, ids: list[int]) -> Sequence[str]:
        """Return the subword piece of each token id."""
        ...


def frame_timestamps_to_seconds(
    raw_timestamp: object, frame_conversion: tuple[float, int] | None
) -> list[float] | None:
    """Convert the forced-language path's raw per-token frame indices to seconds.

    Verified against the real checkpoint (5.85s clip, 37 emitted tokens):
    ``hyp.timestamp`` on this path is a flat per-token frame-index
    Tensor/sequence -- one index per ``hyp.y_sequence`` entry, monotonically
    non-decreasing, e.g. ``[7, 9, 10, ..., 72]`` -- not seconds, and not the
    structured ``{"word": ..., "char": ..., "segment": ...}`` dict NeMo's
    normal timestamp post-processing produces on the auto-detection path.

    Args:
        raw_timestamp: ``hyp.timestamp`` as returned by the forced-language
            ``transcribe()`` call -- a Tensor, a plain sequence of frame
            indices, None, or (defensively) a structured dict.
        frame_conversion: ``(window_stride, subsampling_factor)`` from
            :func:`~coro.backends.asr.nemo_checkpoint.resolve_frame_conversion_factors`,
            or None.

    Returns:
        A plain list of per-token seconds, or None when ``raw_timestamp``
        is absent/unconvertible or ``frame_conversion`` is unavailable --
        callers must degrade to text-fallback timing rather than treat
        unconverted frame indices as seconds.

    """
    if raw_timestamp is None or frame_conversion is None:
        return None
    tolist = getattr(raw_timestamp, "tolist", None)
    if tolist is not None:  # torch.Tensor, no top-level torch import
        raw_timestamp = tolist()
    if isinstance(raw_timestamp, dict):  # auto-path's structured shape, not this path's
        return None
    if not isinstance(raw_timestamp, Iterable):
        return None
    try:
        frame_indices = [float(f) for f in raw_timestamp]
    except TypeError:
        return None
    window_stride, subsampling_factor = frame_conversion
    return [f * window_stride * subsampling_factor for f in frame_indices]


def _per_token_seconds(raw_timestamp: object) -> list[float]:
    """Return ``hyp.timestamp`` as flat per-token seconds, or ``[]`` when untrusted.

    A torch.Tensor (duck-typed via ``numel``, no top-level torch import) or
    the auto-path's structured dict is not a flat per-token seconds
    sequence, so it is discarded rather than misread.
    """
    if hasattr(raw_timestamp, "numel") or isinstance(raw_timestamp, dict):
        return []
    if not raw_timestamp:
        return []
    if not isinstance(raw_timestamp, Iterable):
        msg = f"{type(raw_timestamp).__name__!r} object is not iterable"
        raise TypeError(msg)
    return [float(t) for t in raw_timestamp]


def _subword_pieces(hyp: object, tokenizer: SubwordTokenizer | None) -> list[str]:
    """Decode a hypothesis's token ids into SentencePiece pieces."""
    token_ids = getattr(hyp, "y_sequence", None)
    if token_ids is None:
        token_ids = getattr(hyp, "y", None)  # legacy/test-fixture attribute name
    token_ids = list(token_ids.tolist()) if hasattr(token_ids, "tolist") else list(token_ids or ())
    if tokenizer is None or not token_ids:
        return []
    return list(tokenizer.ids_to_tokens(token_ids))


def tokens_from_hypothesis(
    hyp: object, tokenizer: SubwordTokenizer | None, *, span_end: float
) -> list[TranscriptToken]:
    """Convert one NeMo RNNT hypothesis, timed by its own ``hyp.timestamp``.

    See :func:`tokens_from_timed_hypothesis`.
    """
    return tokens_from_timed_hypothesis(
        hyp, getattr(hyp, "timestamp", None), tokenizer, span_end=span_end
    )


def tokens_from_timed_hypothesis(
    hyp: object,
    raw_timestamp: object,
    tokenizer: SubwordTokenizer | None,
    *,
    span_end: float,
) -> list[TranscriptToken]:
    """Convert one NeMo RNNT hypothesis into word-level TranscriptTokens.

    ``raw_timestamp`` carries one emission
    time (in seconds) per predicted subword token (SentencePiece pieces using
    the ``▁`` word-start marker), so words are reconstructed with the same
    grouping as the onnx-asr converter: a word's ``start`` is its first
    subword's time, its ``end`` the next word's start (final word padded by
    ``LAST_WORD_PAD``). No per-word probability source exists on the
    hypothesis, so ``probability`` stays None rather than being manufactured.
    When the checkpoint emits no timestamps, timings are spread evenly over
    the clip span via the shared text fallback.

    The forced-language path (:class:`~coro.backends.asr.nemo.NemoASRAdapter`)
    passes timestamps already converted from frame indices to seconds by
    :func:`frame_timestamps_to_seconds` (or None when conversion factors were
    unavailable) -- so a raw ``Tensor`` should not normally appear here. It
    is still guarded defensively (rather than assumed impossible): a Tensor
    is treated as untrusted and discarded, as is a dict (the structured
    ``{"word": ..., "char": ..., "segment": ...}`` shape NeMo's normal
    timestamp post-processing produces on the auto-detection path, which this
    model-agnostic helper does not parse) -- both would otherwise crash or
    silently produce garbled word timing. Falling back to evenly-spaced text
    timing is the safe choice in either case.
    """
    timestamps = _per_token_seconds(raw_timestamp)
    pieces = _subword_pieces(hyp, tokenizer) if timestamps else []
    if pieces and len(pieces) == len(timestamps):
        return words_from_subword_groups(group_subwords(pieces, timestamps, None))

    text = (getattr(hyp, "text", "") or "").strip()
    return words_from_text(text, 0.0, span_end or None)
