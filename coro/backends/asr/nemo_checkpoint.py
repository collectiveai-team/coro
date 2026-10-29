"""Checkpoint introspection for the nemo ASR backend.

Resolves what the ``nemo`` backend needs to know about a loaded NeMo ASR
checkpoint: which prompt-dictionary key a request language forces, which
``override_config`` dataclass its ``transcribe()`` accepts, and the factors
that turn its per-token frame indices into seconds. See
``coro/backends/asr/nemo.py``'s module docstring for the NeMo quirks these
helpers work around.
"""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING, Protocol

from coro.backends.asr.errors import AsrUnsupportedLanguageError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class NemoTranscriber(Protocol):
    """The slice of a NeMo ASR model the ``nemo`` adapter calls.

    ``transcribe`` stays a ``Callable[..., ...]`` because NeMo's signature
    differs per model class (the Prompt subclass adds its own
    ``override_config`` type), and :func:`resolve_override_config_type` reads
    that signature at runtime instead.
    """

    @property
    def transcribe(self) -> Callable[..., Sequence[object] | None]:
        """NeMo's ``transcribe`` method."""
        ...


def resolve_target_language(language: str | None, prompt_dictionary: dict[str, int]) -> str | None:
    """Resolve a request language to a checkpoint prompt-dictionary key.

    Args:
        language: Request language (e.g. ``"es"``, ``"es-US"``, ``"es-ES"``),
            or None to leave the model's default behaviour (auto detection
            where the checkpoint supports it).
        prompt_dictionary: The checkpoint's target-language -> prompt-id map.

    Returns:
        The dictionary key to pass as ``target_lang``, or None when no
        language was requested.

    Raises:
        AsrUnsupportedLanguageError: If the language matches no dictionary key.
            Listing the supported keys turns a silent English fallback (the
            failure mode this backend exists to avoid) into an explicit error
            that every request surface reports as a 400 naming the supported set.

    """
    if not language:
        return None
    if language in prompt_dictionary:
        return language
    primary = language.split("-")[0]
    for key in prompt_dictionary:
        if key.split("-")[0] == primary:
            return key
    raise AsrUnsupportedLanguageError(language, supported_languages=prompt_dictionary)


def resolve_override_config_type(model: NemoTranscriber) -> type | None:
    """Discover a model's ``transcribe(override_config=...)`` dataclass type.

    NeMo checkpoints with prompt conditioning declare their own
    ``TranscribeConfig`` subclass (e.g. ``HybridRNNTCTCPromptTranscribeConfig``)
    as the ``override_config`` parameter's type annotation on their
    ``transcribe()`` override. Discovering it via introspection -- rather
    than importing one specific class by name -- lets this adapter drive any
    such subclass without hard-coding to
    ``EncDecHybridRNNTCTCBPEModelWithPrompt``.

    Returns:
        The dataclass type, or None if ``transcribe`` has no typed
        ``override_config`` parameter (e.g. a plain non-prompt ASR model, or
        a test double without one).

    """
    # get_type_hints (not inspect.signature(...).annotation) is required:
    # both this project's and NeMo's modules use
    # `from __future__ import annotations` (PEP 563), which stores
    # annotations as unevaluated strings -- inspect.signature would hand
    # back the literal string "Optional[HybridRNNTCTCPromptTranscribeConfig]"
    # instead of the type. get_type_hints resolves it against the function's
    # own module globals regardless of which annotation style is in effect.
    try:
        hints = typing.get_type_hints(model.transcribe)
    except (NameError, TypeError, AttributeError):
        return None
    annotation = hints.get("override_config")
    if annotation is None:
        return None
    for candidate in typing.get_args(annotation) or (annotation,):
        if candidate is not type(None) and isinstance(candidate, type):
            return candidate
    return None


def build_forced_language_config(model: NemoTranscriber, target_lang: str) -> object:
    """Build the ``override_config`` needed to reliably force ``target_lang``.

    See the ``nemo`` module docstring's two NeMo-quirk caveats:
    ``use_lhotse=False`` is required for ``target_lang`` to reach the model at
    all; ``timestamps=False`` works around a downstream crash on this code
    path. ``num_workers=0`` avoids a ``DataLoader`` multiprocessing crash
    (``OSError: AF_UNIX path too long``) that surfaces whenever the resolved
    working/temp directory path is long enough to overflow the platform's
    Unix-socket path limit -- deterministic given a fixed checkout path, not
    caller-input-dependent, so it is always disabled rather than detected.

    Raises:
        ValueError: If the model exposes no typed ``override_config``
            parameter, so a caller-supplied ``language`` cannot be forced
            reliably. This should not happen for a checkpoint with a prompt
            dictionary; surfacing loudly matches this adapter's no-silent-
            fallback policy for language handling.

    """
    config_cls = resolve_override_config_type(model)
    if config_cls is None:
        msg = (
            "This checkpoint's transcribe() has no typed override_config "
            "parameter, so language forcing cannot be driven reliably. This "
            "should not happen for a checkpoint with a prompt dictionary -- "
            "please check the checkpoint's model class."
        )
        raise ValueError(msg)
    return config_cls(
        batch_size=1,
        return_hypotheses=True,
        timestamps=False,
        verbose=False,
        target_lang=target_lang,
        use_lhotse=False,
        num_workers=0,
    )


def resolve_frame_conversion_factors(model: object) -> tuple[float, int] | None:
    """Discover a model's frame-index -> seconds conversion factors.

    Needed to convert the forced-language path's raw per-token frame-index
    ``hyp.timestamp`` into seconds:
    ``frame_idx * window_stride * subsampling_factor`` (the same formula
    NeMo's own ``timestamp_utils.process_timestamp`` applies internally).

    ``window_stride`` comes from the preprocessor config
    (``model.cfg.preprocessor.window_stride``). ``subsampling_factor`` comes
    from the encoder module's own instance attribute
    (``model.encoder.subsampling_factor``) -- *not*
    ``model.cfg.subsampling_factor`` or
    ``model.cfg.model_defaults.subsampling_factor``, both of which were
    tried against the real checkpoint and don't exist: the
    ``subsampling_factor: 8`` line visible in NeMo's own config-dump log at
    load time is nested inside the train/validation/test dataset configs
    (a dataloader-side duplicate used for prompt one-hot bucketing), not
    the architectural source of truth. The encoder attribute is also
    ``struct``-mode-safe, unlike probing arbitrary nested ``cfg`` paths.

    Returns:
        ``(window_stride, subsampling_factor)``, or None if either is
        missing -- callers must degrade to text-fallback timing rather
        than guess at a conversion.

    """
    preprocessor = getattr(getattr(model, "cfg", None), "preprocessor", None)
    try:
        window_stride = float(getattr(preprocessor, "window_stride"))  # noqa: B009 -- AttributeError is the "missing" signal
    except (AttributeError, TypeError, ValueError):
        return None
    subsampling_factor = getattr(getattr(model, "encoder", None), "subsampling_factor", None)
    if subsampling_factor is None:
        return None
    try:
        return window_stride, int(subsampling_factor)
    except (TypeError, ValueError):
        return None
