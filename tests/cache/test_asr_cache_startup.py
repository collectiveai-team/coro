"""Enabling the ASR window cache must not change what starting up means.

Two invariants are easy to break and expensive to notice. A misconfigured cache
directory that only fails on the first request turns an operator error into a
production incident. And a Server Warmup served from cache would report
readiness on every start after the first without having loaded a model at all,
which makes the health contract a lie.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from coro.cache.directory import CacheDirectoryError, default_cache_dir, resolve_cache_dir
from coro.settings import ServerSettings


# MARK: Directory Resolution
def test_a_real_disk_directory_is_accepted(fake_mounts):
    mounts = fake_mounts(disk="ext4")
    assert resolve_cache_dir(str(mounts.path("disk"))) == str(mounts.path("disk"))


def test_a_ram_backed_directory_is_rejected(fake_mounts):
    mounts = fake_mounts(ram="tmpfs")
    with pytest.raises(CacheDirectoryError, match="RAM-backed"):
        resolve_cache_dir(str(mounts.path("ram")))


def test_the_default_lives_under_the_user_cache_root(monkeypatch, tmp_path):
    """A persistent cache must not default to a directory the OS reclaims."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert default_cache_dir().startswith(str(tmp_path))


def test_an_undeterminable_filesystem_is_accepted(tmp_path, undetectable_filesystem):
    """An unreadable mount table is not evidence of a RAM-backed directory."""
    assert resolve_cache_dir(str(tmp_path)) == str(tmp_path)


# MARK: Strict Startup Validation
def test_startup_resolves_the_cache_directory_when_enabled(fake_mounts):
    mounts = fake_mounts(disk="ext4")
    disk = str(mounts.path("disk"))
    settings = ServerSettings(asr_cache="enabled", asr_cache_dir=disk, _env_file=None)
    assert settings.asr_cache_dir == disk


def test_startup_fills_in_a_default_cache_directory_when_enabled(
    monkeypatch, tmp_path, undetectable_filesystem
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    settings = ServerSettings(asr_cache="enabled", _env_file=None)
    assert settings.asr_cache_dir is not None


def test_startup_rejects_a_ram_backed_cache_directory(fake_mounts):
    mounts = fake_mounts(ram="tmpfs")
    with pytest.raises(ValidationError, match="RAM-backed"):
        ServerSettings(asr_cache="enabled", asr_cache_dir=str(mounts.path("ram")), _env_file=None)


def test_a_disabled_cache_leaves_the_directory_untouched(fake_mounts):
    """A stale directory setting must not fail a server that is not caching."""
    mounts = fake_mounts(ram="tmpfs")
    ram = str(mounts.path("ram"))
    settings = ServerSettings(asr_cache="disabled", asr_cache_dir=ram, _env_file=None)
    assert settings.asr_cache_dir == ram


def test_the_cache_is_disabled_by_default():
    """Upgrading must not silently start consuming an operator's disk."""
    assert ServerSettings(_env_file=None).asr_cache == "disabled"


# MARK: Server Warmup
def test_warmup_still_reaches_the_model_with_a_populated_cache(
    tmp_path, monkeypatch, undetectable_filesystem
):
    """Readiness must keep meaning "the model loaded and ran", start after start."""
    from unittest.mock import patch

    from starlette.testclient import TestClient

    from coro.app import create_app
    from coro.core.models import TranscriptToken

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls: list[int] = []

    class _CountingASR:
        honours_prompt = False

        async def transcribe_pcm(self, pcm, *, language=None, prompt=None):
            calls.append(len(pcm))
            return [TranscriptToken(start=0.0, end=1.0, text=" warm", probability=0.5)]

    settings = ServerSettings(asr_cache="enabled", warmup="enabled", _env_file=None)
    readiness: list[bool] = []
    with patch(
        "coro.backends.asr.factory.build_asr_adapter",
        autospec=True,
        side_effect=lambda _s: _CountingASR(),
    ):
        for _ in range(2):
            with TestClient(create_app(settings)) as client:
                readiness.append(client.get("/health").json()["warmup_ready"])

    assert readiness == [True, True]
    # Two starts, and the second still ran the model: a warmup served from the
    # cache would have left this at one window's worth of calls.
    assert len(calls) == 2
