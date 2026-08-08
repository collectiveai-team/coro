"""Adapter Concurrency Policy primitives shared by the ASR Adapters.

Every ASR Adapter owns its Adapter Concurrency Policy: whether concurrent model
calls are safe, how many may run at once, and how many may wait. This module
supplies the one mechanism all three adapters express that policy with, so the
policy stays a per-adapter *decision* rather than a per-adapter *implementation*.

Two knobs, both surfaced through Server Startup Selection:

``max_concurrency``
    How many inference calls may execute at once. This bounds total thread
    demand: each in-flight call drives a backend thread pool, so the product of
    concurrency and per-call threads is what must stay near the core count.

``max_queue_depth``
    How many further calls may wait for a permit. Beyond that the call is
    rejected with :class:`AsrCapacityError`, which the API boundary renders as an
    OpenAI-Style Error carrying a ``Retry-After`` hint. Rejecting is deliberate:
    accepting unbounded work under overload degrades every in-flight request and
    makes Performance Benchmark process-tree sampling uninterpretable.

An adapter whose backend is genuinely not thread-safe expresses that as
``max_concurrency=1`` rather than as a ``threading.Lock``. The effect is the same
serialisation, but waiting happens on the event loop instead of pinning a worker
thread, and the queue-depth cap still applies.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Nominal backend threads consumed by one in-flight inference call. Matches
# CTranslate2's default ``intra_threads`` (4); ONNX Runtime shares one intra-op
# pool across concurrent ``run`` calls on the same session, so this over- rather
# than under-estimates its demand.
_THREADS_PER_INFERENCE = 4

# Auto-sizing never resolves below this. A single permit would re-introduce the
# serialisation this module exists to remove, so small hosts accept mild
# oversubscription in exchange for request-level overlap (feature extraction,
# decoding and response assembly overlap even when the compute pool is busy).
_MIN_AUTO_CONCURRENCY = 2

# Retry hint handed to a rejected caller, in seconds. Sized to roughly one short
# transcription so a client backs off past the current admission burst.
DEFAULT_RETRY_AFTER_SECONDS = 5.0


# MARK: Capacity Failure
class AsrCapacityError(RuntimeError):
    """Raised when the ASR admission queue is full.

    Carries the retry hint the API boundary converts into a ``Retry-After``
    header on the OpenAI-Style Error response.
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float = DEFAULT_RETRY_AFTER_SECONDS,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.retry_after_seconds = retry_after_seconds


# MARK: Concurrency Sizing
def resolve_max_concurrency(configured: int) -> int:
    """Resolve the configured ASR concurrency limit to a concrete permit count.

    Args:
        configured: The ``asr_max_concurrency`` Server Startup Selection value.
            Zero (the default) means auto-size from the host core count.

    Returns:
        Number of inference calls allowed to run at once (always >= 1).

    """
    if configured > 0:
        return configured
    cores = os.cpu_count() or 1
    return max(_MIN_AUTO_CONCURRENCY, cores // _THREADS_PER_INFERENCE)


# MARK: Admission Control
class AdmissionController:
    """Bounds in-flight inference calls and rejects past a queue-depth cap.

    Not thread-safe by design: it is driven from the event loop, and adapters
    hand the actual model call to ``asyncio.to_thread`` only *after* a permit has
    been acquired.
    """

    def __init__(
        self,
        *,
        max_concurrency: int,
        max_queue_depth: int,
        retry_after_seconds: float = DEFAULT_RETRY_AFTER_SECONDS,
    ) -> None:
        """Create an admission controller.

        Args:
            max_concurrency: Inference calls allowed to run at once (clamped to >= 1).
            max_queue_depth: Calls allowed to wait for a permit (clamped to >= 0).
            retry_after_seconds: Retry hint attached to rejections.

        """
        self._max_concurrency = max(1, max_concurrency)
        self._max_queue_depth = max(0, max_queue_depth)
        self._retry_after_seconds = retry_after_seconds
        self._semaphore = asyncio.Semaphore(self._max_concurrency)
        self._in_flight = 0
        self._queued = 0

    @property
    def max_concurrency(self) -> int:
        """Permit count: inference calls allowed to run at once."""
        return self._max_concurrency

    @property
    def max_queue_depth(self) -> int:
        """Waiter cap: calls allowed to queue before rejection kicks in."""
        return self._max_queue_depth

    @property
    def in_flight(self) -> int:
        """Inference calls currently holding a permit."""
        return self._in_flight

    @property
    def queued(self) -> int:
        """Calls currently waiting for a permit."""
        return self._queued

    @asynccontextmanager
    async def admit(self) -> AsyncIterator[None]:
        """Hold one admission permit for the duration of the context.

        Yields:
            None, once a permit has been acquired.

        Raises:
            AsrCapacityError: If every permit is taken and the wait queue is full.

        """
        if self._in_flight >= self._max_concurrency and self._queued >= self._max_queue_depth:
            logger.warning(
                "ASR admission rejected: in_flight=%d/%d queued=%d/%d",
                self._in_flight,
                self._max_concurrency,
                self._queued,
                self._max_queue_depth,
            )
            msg = (
                f"Server is at ASR capacity ({self._max_concurrency} concurrent "
                f"transcriptions, {self._max_queue_depth} queued). Retry in "
                f"{self._retry_after_seconds:.0f}s."
            )
            raise AsrCapacityError(msg, retry_after_seconds=self._retry_after_seconds)

        self._queued += 1
        try:
            await self._semaphore.acquire()
        finally:
            self._queued -= 1

        self._in_flight += 1
        try:
            yield
        finally:
            self._in_flight -= 1
            self._semaphore.release()


def build_admission_controller(
    *,
    max_concurrency: int,
    max_queue_depth: int,
    serialized: bool = False,
) -> AdmissionController:
    """Build the admission controller expressing one adapter's concurrency policy.

    Args:
        max_concurrency: Configured ``asr_max_concurrency`` (0 = auto-size).
        max_queue_depth: Configured ``asr_max_queue_depth``.
        serialized: True when the adapter's backend requires serialised model
            calls, which forces a single permit regardless of configuration.

    Returns:
        An admission controller sized for the adapter.

    """
    resolved = 1 if serialized else resolve_max_concurrency(max_concurrency)
    return AdmissionController(max_concurrency=resolved, max_queue_depth=max_queue_depth)
