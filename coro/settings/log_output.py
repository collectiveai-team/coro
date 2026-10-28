"""The log settings: the level and the output format of the house logger."""

from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogSettings(BaseSettings):
    """The level and the environment that pick how ``coro.core.logger`` writes events.

    ``CORO_LOG_LEVEL`` is the variable that ``ServerSettings.log_level`` reads, so the
    server and the house logger agree; the bare ``LOG_LEVEL`` and ``ENV`` of the
    structured-logging standard are accepted when the ``CORO_`` ones are unset.
    """

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: str = Field(
        default="info",
        validation_alias=AliasChoices("CORO_LOG_LEVEL", "LOG_LEVEL"),
        description="Level of the root logger when the house logger configures it.",
    )
    env: str = Field(
        default="dev",
        validation_alias=AliasChoices("CORO_ENV", "ENV"),
        description="``prod`` or ``production`` writes JSON lines; anything else a console line.",
    )

    @property
    def is_prod(self) -> bool:
        """Return whether events are written as JSON lines."""
        return self.env.lower() in {"prod", "production"}


@lru_cache(maxsize=1)
def get_settings() -> LogSettings:
    """Return the log settings, read once per process.

    The package exports it as ``coro.settings.get_log_settings``.
    """
    return LogSettings()
