"""Variables of the host that coro reads but does not own (no ``CORO_`` prefix)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class HostEnvironment(BaseSettings):
    """The XDG and CUDA variables that place caches and describe the visible GPUs."""

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    xdg_cache_home: str | None = None
    cuda_visible_devices: str = ""

    @property
    def cache_root(self) -> Path:
        """Return the user cache root: ``$XDG_CACHE_HOME``, else ``~/.cache``."""
        return Path(self.xdg_cache_home) if self.xdg_cache_home else Path.home() / ".cache"


def host_environment() -> HostEnvironment:
    """Read the host variables now.

    Not cached: they are cheap to read, and tests set them per test.
    """
    return HostEnvironment()
