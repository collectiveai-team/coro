"""Per-client rate limits: requests per minute and audio minutes per hour.

A client over its budget is rejected with HTTP 429 and a ``Retry-After`` hint,
never queued: delaying work is what sharing the CPU already does, and a queued
excess would still cost disk and CPU later (ADR 0024). Both limits are token
buckets keyed by client IP, held in memory per server process.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass

from coro.audio_probe import probe_duration_seconds

logger = logging.getLogger(__name__)

# Idle buckets are dropped once they have refilled, checked every this many calls.
_SWEEP_EVERY = 1024

UNKNOWN_CLIENT = "unknown"


@dataclass
class _Bucket:
    tokens: float
    updated: float


class TokenBucket:
    """Token buckets of one shape, one per key.

    ``capacity`` is the burst a fresh client may spend at once; tokens refill
    at ``refill_per_second`` up to that capacity.
    """

    def __init__(
        self,
        *,
        capacity: float,
        refill_per_second: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._capacity = capacity
        self._rate = refill_per_second
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}
        self._calls = 0

    def try_consume(self, key: str, cost: float) -> float | None:
        """Spend ``cost`` tokens for ``key``; return None, or seconds to wait.

        A cost larger than the whole capacity can never be covered, so it is
        admitted once the bucket is full and leaves it in debt: the client
        then waits for the debt to refill before its next admission.
        """
        bucket = self._refilled(key)
        needed = min(cost, self._capacity)
        if bucket.tokens < needed:
            return (needed - bucket.tokens) / self._rate
        bucket.tokens -= cost
        return None

    def wait_until_positive(self, key: str) -> float | None:
        """Return None if ``key`` has any tokens left, else seconds until it has one."""
        bucket = self._refilled(key)
        return None if bucket.tokens > 0 else (1 - bucket.tokens) / self._rate

    def _refilled(self, key: str) -> _Bucket:
        now = self._clock()
        self._calls += 1
        if self._calls % _SWEEP_EVERY == 0:
            self._sweep(now)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = self._buckets[key] = _Bucket(tokens=self._capacity, updated=now)
            return bucket
        bucket.tokens = min(self._capacity, bucket.tokens + (now - bucket.updated) * self._rate)
        bucket.updated = now
        return bucket

    def _sweep(self, now: float) -> None:
        full = [
            key
            for key, bucket in self._buckets.items()
            if bucket.tokens + (now - bucket.updated) * self._rate >= self._capacity
        ]
        for key in full:
            del self._buckets[key]

    def __len__(self) -> int:
        return len(self._buckets)


class ClientRateLimits:
    """The server's two per-client limits; either may be disabled (``None``)."""

    def __init__(
        self,
        *,
        requests_per_minute: int,
        audio_minutes_per_hour: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.requests = (
            TokenBucket(
                capacity=requests_per_minute,
                refill_per_second=requests_per_minute / 60,
                clock=clock,
            )
            if requests_per_minute > 0
            else None
        )
        audio_seconds_per_hour = audio_minutes_per_hour * 60
        self.audio = (
            TokenBucket(
                capacity=audio_seconds_per_hour,
                refill_per_second=audio_seconds_per_hour / 3600,
                clock=clock,
            )
            if audio_seconds_per_hour > 0
            else None
        )

    @classmethod
    def from_settings(cls, settings) -> ClientRateLimits:
        """Build the limits, failing startup if the audio limit cannot be enforced.

        Raises:
            RuntimeError: The audio limit is enabled and ``ffprobe`` is not on
                PATH, so uploads could not be measured before processing.

        """
        if settings.rate_limit_audio_minutes_per_hour > 0 and shutil.which("ffprobe") is None:
            raise RuntimeError(
                "rate_limit_audio_minutes_per_hour is set but ffprobe is not on PATH; "
                "install ffmpeg (which ships ffprobe) or disable the limit."
            )
        return cls(
            requests_per_minute=settings.rate_limit_requests_per_minute,
            audio_minutes_per_hour=settings.rate_limit_audio_minutes_per_hour,
        )

    def admit_request(self, client: str) -> float | None:
        """Count one request; return None, or seconds until it would be admitted."""
        if self.requests is None:
            return None
        return self.requests.try_consume(client, 1)

    def charge_audio(self, client: str, seconds: float) -> float | None:
        """Charge ``seconds`` of audio; return None, or seconds until it fits."""
        if self.audio is None:
            return None
        return self.audio.try_consume(client, seconds)


REQUESTS_EXCEEDED_MESSAGE = "Rate limit exceeded: too many transcription requests from this client."
AUDIO_EXCEEDED_MESSAGE = "Rate limit exceeded: this client's audio quota is used up."


@dataclass(frozen=True)
class RateLimited:
    """A rejection, rendered by each route in its vendor's error shape."""

    message: str
    retry_after_seconds: float


def admit_request(connection) -> RateLimited | None:
    """Count one transcription request from this connection's client."""
    limits = rate_limits_of(connection)
    if limits is None:
        return None
    wait = limits.admit_request(client_key(connection))
    if wait is None:
        return None
    logger.info(
        "rate_limit requests exceeded client=%s retry_after=%.1fs", client_key(connection), wait
    )
    return RateLimited(REQUESTS_EXCEEDED_MESSAGE, wait)


def audio_quota_empty(connection) -> RateLimited | None:
    """Reject a stream at connect time when the client has no audio left."""
    limits = rate_limits_of(connection)
    if limits is None or limits.audio is None:
        return None
    wait = limits.audio.wait_until_positive(client_key(connection))
    return None if wait is None else RateLimited(AUDIO_EXCEEDED_MESSAGE, wait)


async def admit_upload(connection, audio) -> RateLimited | None:
    """Charge an upload's duration, measured before any decoding.

    ``audio`` is an ``AudioInput``; it is spooled to disk only when the audio
    limit is enabled. An upload ffprobe cannot measure is admitted uncharged:
    the pipeline's decoder then rejects it if it is not audio at all.
    """
    limits = rate_limits_of(connection)
    if limits is None or limits.audio is None:
        return None
    seconds = await probe_duration_seconds(await audio.temp_path())
    if seconds is None:
        logger.warning("rate_limit could not measure upload; admitted uncharged")
        return None
    wait = limits.charge_audio(client_key(connection), seconds)
    if wait is None:
        return None
    logger.info(
        "rate_limit audio exceeded client=%s upload_s=%.1f retry_after=%.1fs",
        client_key(connection),
        seconds,
        wait,
    )
    return RateLimited(AUDIO_EXCEEDED_MESSAGE, wait)


def client_key(connection) -> str:
    """Return the rate-limit key of an HTTP request or WebSocket: the client IP.

    Behind a reverse proxy this is the proxy's address unless uvicorn trusts
    its forwarding headers (``--forwarded-allow-ips``).
    """
    client = getattr(connection, "client", None)
    return client.host if client is not None and client.host else UNKNOWN_CLIENT


def rate_limits_of(connection) -> ClientRateLimits | None:
    """Return the app's limits, or None for an app built without them."""
    return getattr(connection.app.state, "rate_limits", None)
