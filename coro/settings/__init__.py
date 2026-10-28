"""Package-owned settings using pydantic-settings: the only place that reads the environment.

Heavy model initialization lives in application lifespan, not here.
Logging is configured only from CLI/startup paths; importing this
package must not mutate global logging policy.
"""

from coro.settings.host import HostEnvironment, host_environment
from coro.settings.selection import (
    DEFAULT_MODEL_SLUG,
    FP32_QUANTIZATION_SENTINEL,
    MODEL_SLUGS,
    ASRBackendProvider,
    ASRDevice,
    DiarizationBackendProvider,
    DiarizationDevice,
    DiarizationLatencyTier,
    OnnxVadSelector,
    PipelineSelector,
)
from coro.settings.server import ServerSettings, get_settings

__all__ = [
    "DEFAULT_MODEL_SLUG",
    "FP32_QUANTIZATION_SENTINEL",
    "MODEL_SLUGS",
    "ASRBackendProvider",
    "ASRDevice",
    "DiarizationBackendProvider",
    "DiarizationDevice",
    "DiarizationLatencyTier",
    "HostEnvironment",
    "OnnxVadSelector",
    "PipelineSelector",
    "ServerSettings",
    "get_settings",
    "host_environment",
]
