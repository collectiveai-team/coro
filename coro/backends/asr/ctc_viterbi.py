"""Monotonic alignment of CTC emissions to a known label sequence."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


def ctc_state_path(
    emissions: npt.NDArray[np.float32], labels: list[int], *, blank_id: int
) -> npt.NDArray[np.int64]:
    """Return the visited state at each frame, with blanks between target labels."""
    if not labels or not len(emissions):
        raise ValueError("CTC alignment requires audio frames and target labels")
    states = np.full(2 * len(labels) + 1, blank_id, dtype=np.int64)
    states[1::2] = labels
    skip = (states[2:] != blank_id) & (states[2:] != states[:-2])
    scores = np.full(len(states), -np.inf, dtype=np.float32)
    scores[0] = 0
    back = np.empty((len(emissions), len(states)), dtype=np.uint8)
    moves = np.full((3, len(states)), -np.inf, dtype=np.float32)
    for i, frame in enumerate(emissions):
        moves[0] = scores
        moves[1, 1:] = scores[:-1]
        moves[2, 2:] = np.where(skip, scores[:-2], -np.inf)
        back[i] = moves.argmax(axis=0)
        scores = moves.max(axis=0) + frame[states]
    state = len(states) - 2 + int(np.argmax(scores[-2:]))
    if not np.isfinite(scores[state]):
        raise ValueError("CTC alignment found no complete path")
    path = np.empty(len(emissions), dtype=np.int64)
    for i in range(len(path) - 1, -1, -1):
        path[i] = state
        state -= int(back[i, state])
    return path
