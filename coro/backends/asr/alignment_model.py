"""Select the required alignment model independently of the Canary decoder."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

if TYPE_CHECKING:
    from collections.abc import Sequence

    from coro.backends.asr.canary_ctc_alignment import CanaryWordAligner
    from coro.settings.selection import CanaryAlignmentModel


def build_alignment_model(
    model: CanaryAlignmentModel, *, providers: Sequence[str] | None, hf_token: str | None
) -> CanaryWordAligner:
    """Load the chosen alignment adapter; never substitute another model on failure."""
    match model:
        case "canary-nemo":
            from coro.backends.asr.canary_ctc_alignment import build_canary_word_aligner

            return build_canary_word_aligner(providers=providers, hf_token=hf_token)
        case "omnilingual-ctc-int8":
            from coro.backends.asr.omnilingual_ctc_alignment import build_omnilingual_word_aligner

            return build_omnilingual_word_aligner(hf_token=hf_token)
        case _:
            assert_never(model)
