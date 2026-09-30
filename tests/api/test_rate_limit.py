"""Per-client token buckets: burst, refill, isolation, debt and cleanup (ADR 0024)."""

from __future__ import annotations

import pytest

from coro.api import rate_limit
from coro.api.rate_limit import ClientRateLimits, TokenBucket
from coro.settings import ServerSettings


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _bucket(clock: _Clock, *, capacity: float = 3, per_second: float = 1) -> TokenBucket:
    return TokenBucket(capacity=capacity, refill_per_second=per_second, clock=clock)


class TestTokenBucket:
    def test_a_fresh_client_may_spend_the_whole_burst(self):
        bucket = _bucket(_Clock())
        assert [bucket.try_consume("a", 1) for _ in range(3)] == [None, None, None]

    def test_beyond_the_burst_the_wait_is_the_time_to_refill_the_shortfall(self):
        bucket = _bucket(_Clock(), per_second=0.5)
        for _ in range(3):
            bucket.try_consume("a", 1)
        assert bucket.try_consume("a", 1) == pytest.approx(2.0, abs=1e-9)

    def test_tokens_refill_with_time(self):
        clock = _Clock()
        bucket = _bucket(clock)
        for _ in range(3):
            bucket.try_consume("a", 1)
        clock.now += 1.0
        assert bucket.try_consume("a", 1) is None
        assert bucket.try_consume("a", 1) == pytest.approx(1.0, abs=1e-9)

    def test_refill_never_exceeds_the_capacity(self):
        clock = _Clock()
        bucket = _bucket(clock)
        bucket.try_consume("a", 1)
        clock.now += 3600
        assert [bucket.try_consume("a", 1) for _ in range(4)][-1] == pytest.approx(1.0, abs=1e-9)

    def test_clients_do_not_share_a_bucket(self):
        bucket = _bucket(_Clock())
        for _ in range(3):
            bucket.try_consume("a", 1)
        assert bucket.try_consume("b", 1) is None

    def test_a_cost_above_capacity_is_admitted_once_and_leaves_debt(self):
        clock = _Clock()
        bucket = _bucket(clock, capacity=10)
        assert bucket.try_consume("a", 25) is None
        # 15 tokens in debt: one more token needs 16 s of refill.
        assert bucket.try_consume("a", 1) == pytest.approx(16.0, abs=1e-9)

    def test_wait_until_positive_reports_the_time_to_one_token(self):
        bucket = _bucket(_Clock(), capacity=10)
        assert bucket.wait_until_positive("a") is None
        bucket.try_consume("a", 12)
        assert bucket.wait_until_positive("a") == pytest.approx(3.0, abs=1e-9)

    def test_idle_refilled_buckets_are_swept(self):
        clock = _Clock()
        bucket = _bucket(clock)
        for i in range(rate_limit._SWEEP_EVERY - 1):
            bucket.try_consume(f"client-{i}", 1)
        clock.now += 10
        bucket.try_consume("last", 1)
        assert len(bucket) == 1


class TestClientRateLimits:
    def test_zero_disables_each_limit(self):
        limits = ClientRateLimits(requests_per_minute=0, audio_minutes_per_hour=0)
        assert limits.requests is None
        assert limits.audio is None
        assert limits.admit_request("a") is None
        assert limits.charge_audio("a", 10_000) is None

    def test_requests_per_minute_is_a_burst_refilling_per_minute(self):
        clock = _Clock()
        limits = ClientRateLimits(requests_per_minute=2, audio_minutes_per_hour=0, clock=clock)
        assert [limits.admit_request("a") for _ in range(2)] == [None, None]
        assert limits.admit_request("a") == pytest.approx(30.0, abs=1e-9)

    def test_audio_minutes_per_hour_is_charged_in_seconds(self):
        clock = _Clock()
        limits = ClientRateLimits(requests_per_minute=0, audio_minutes_per_hour=1, clock=clock)
        assert limits.charge_audio("a", 60) is None
        # 1 min/h refills 1 s of audio per minute.
        assert limits.charge_audio("a", 1) == pytest.approx(60.0, abs=1e-9)

    def test_defaults_limit_requests_but_not_audio(self):
        limits = ClientRateLimits.from_settings(ServerSettings(_env_file=None))
        assert limits.requests is not None
        assert limits.audio is None

    def test_audio_limit_without_ffprobe_fails_at_startup(self, monkeypatch):
        monkeypatch.setattr(rate_limit.shutil, "which", lambda name: None)
        settings = ServerSettings(_env_file=None, rate_limit_audio_minutes_per_hour=10)
        with pytest.raises(RuntimeError, match="ffprobe"):
            ClientRateLimits.from_settings(settings)
