"""Package-owned server settings using pydantic-settings.

Heavy model initialization lives in application lifespan, not here.
Logging is configured only from CLI/startup paths; importing this
module must not mutate global logging policy.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class ServerSettings(BaseSettings):
    """Runtime-injectable settings for the asr_diar_server package."""

    host: str = Field(default="0.0.0.0", description="Bind host.")
    port: int = Field(default=8000, description="Bind port.")
    cors_origins: list[str] = Field(default=["*"], description="Allowed CORS origins.")
    backend: str = Field(default="whisper", description="ASR backend identifier.")
    log_level: str = Field(default="info", description="Log level (for CLI use only).")

    model_config = {"env_prefix": "ASR_DIAR_"}
