"""The variables that give the coro-bench flags their defaults, read without a prefix."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class BenchSettings(BaseSettings):
    """``OUT_DIR``, ``SAMPLE_INTERVAL``, ``SERVER_PID``, ``SERVER_MATCH``; a flag wins over each."""

    model_config = SettingsConfigDict(case_sensitive=False, env_ignore_empty=True, extra="ignore")

    out_dir: Path | None = None
    sample_interval: float = 0.25
    server_pid: int | None = None
    server_match: str | None = None


@lru_cache
def get_settings() -> BenchSettings:
    """Return the bench settings of the process, built once.

    A test that changes the environment calls ``get_settings.cache_clear()``.
    """
    return BenchSettings()
