"""Sortformer streaming latency tiers and how they reach a shared model.

A latency tier is the set of Sortformer streaming parameters
(``chunk_len``, ``chunk_right_context``, ``fifo_len``,
``spkcache_update_period``, ``spkcache_len``) a Streaming Diarization Flow
runs with. NeMo reads them off ``model.sortformer_modules`` at call time;
this module applies them without retuning the model shared with the batch
Diarization Adapter.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Generator

    from coro.backends.diarization.nemo.sortformer import HasSortformerModules


@dataclass(frozen=True)
class LatencyTierParams:
    """Immutable Sortformer streaming parameters for one latency tier."""

    chunk_len: int
    chunk_right_context: int
    fifo_len: int
    spkcache_update_period: int
    spkcache_len: int


LATENCY_TIER_PARAMS: dict[str, LatencyTierParams] = {
    "very-high": LatencyTierParams(
        chunk_len=340,
        chunk_right_context=40,
        fifo_len=40,
        spkcache_update_period=300,
        spkcache_len=188,
    ),
    "high": LatencyTierParams(
        chunk_len=124,
        chunk_right_context=1,
        fifo_len=124,
        spkcache_update_period=124,
        spkcache_len=188,
    ),
    "low": LatencyTierParams(
        chunk_len=6,
        chunk_right_context=7,
        fifo_len=188,
        spkcache_update_period=144,
        spkcache_len=188,
    ),
    "ultra-low": LatencyTierParams(
        chunk_len=3,
        chunk_right_context=1,
        fifo_len=188,
        spkcache_update_period=144,
        spkcache_len=188,
    ),
}


def get_latency_tier_params(tier: str) -> LatencyTierParams:
    """Return the immutable streaming parameters for a latency tier."""
    return LATENCY_TIER_PARAMS[tier]


@contextmanager
def applied_streaming_params(
    sortformer_modules: object, params: LatencyTierParams
) -> Generator[None]:
    """Apply latency-tier streaming parameters for the duration of the block only.

    ``sortformer_modules`` is owned by the Sortformer model, and that same
    model object is shared with the batch Diarization Adapter — the batch
    Diarization Flow reads ``chunk_len``, ``chunk_right_context``,
    ``fifo_len``, ``spkcache_len`` and ``spkcache_update_period`` off it during
    ``forward_streaming``/``streaming_update``. Assigning the streaming tier's
    values permanently therefore silently retunes batch diarization, which
    invalidates any batch-vs-streaming comparison in one process.

    NeMo reads these attributes at call time, so they cannot simply be left
    unset; they are applied around each model call and restored afterwards,
    leaving the shared model exactly as it was found.

    NOTE: single-threaded use only (the diarization bench). Overlapping scopes
    on one model interleave their save/restore: one call runs with the other's
    restored values and the model is left retuned. Concurrent request paths use
    :func:`bind_latency_tier`, which never writes to the shared model.
    """
    previous = {f.name: getattr(sortformer_modules, f.name) for f in fields(params)}
    try:
        for name, value in ((f.name, getattr(params, f.name)) for f in fields(params)):
            setattr(sortformer_modules, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(sortformer_modules, name, value)


def bind_latency_tier[M: HasSortformerModules](model: M, params: LatencyTierParams) -> M:
    """Return a view of ``model`` whose ``sortformer_modules`` carries ``params``.

    NeMo's ``forward_streaming_step`` reads the tier parameters off
    ``self.sortformer_modules`` at call time. Rather than writing them onto the
    shared model around each call — which races between concurrent requests
    (see ADR 0022) — the view gets its own shallow copy of that one submodule
    with the tier applied once. Shallow copies share every parameter tensor and
    every other submodule, so the view costs no model memory and the shared
    model, batch Diarization Adapter included, is never written.
    """
    modules = copy.copy(model.sortformer_modules)
    for field in fields(params):
        setattr(modules, field.name, getattr(params, field.name))

    view = copy.copy(model)
    if isinstance(view, torch.nn.Module):
        # nn.Module resolves submodules through ``_modules``; the shallow copy
        # still shares that dict with the original, so give the view its own.
        object.__setattr__(view, "_modules", {**view._modules, "sortformer_modules": modules})
    else:
        # A plain attribute on the copy; read-only only in StreamingSortformerModel.
        setattr(view, "sortformer_modules", modules)  # noqa: B010
    return view
