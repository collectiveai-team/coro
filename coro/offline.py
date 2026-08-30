"""Transcribe a local file without a server.

Getting a transcript used to mean starting the server and uploading the file to
it, which for a multi-gigabyte recording means pushing the whole thing through a
socket before any work starts. Running the same pipeline in-process against the
path skips all of that.

Two execution modes, and the choice is always explicit:

- **In-process** (default). The Configured Transcription Pipeline runs directly
  against the file. No upload, no server, no port.
- **Attached**, only when a server URL is given. The file is uploaded to a
  server that is already running, which is worth doing when that server has a
  warm model. Never auto-probed: the same command must not behave differently
  depending on whether something happens to be listening on a port.

An attached run is governed by the server's own configuration, not by the flags
passed locally, so both modes report which configuration actually produced the
result rather than assuming the local one did.

Both modes emit the same body: the in-process path renders through the very
function the transcription endpoint renders through, so `coro run` cannot become
a fourth, undocumented response shape.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from coro.audio import AudioInput

if TYPE_CHECKING:
    from coro.settings import ServerSettings

DEFAULT_RESPONSE_FORMAT = "diarized_json"
"""Richest project-native shape: sentence segments carrying speaker labels."""

logger = logging.getLogger(__name__)

_TRANSCRIPTION_PATH = "/v1/audio/transcriptions"
_HEALTH_PATH = "/health"
_ATTACHED_TIMEOUT_SECONDS = 3600.0


def build_http_client() -> Any:
    """Return the HTTP client used for attached runs.

    A named seam rather than an inline import: the in-process path never calls
    it, so httpx stays unimported there, and tests can substitute a loopback
    client without a live socket.
    """
    import httpx

    return httpx.AsyncClient(timeout=_ATTACHED_TIMEOUT_SECONDS)


@dataclass(frozen=True)
class OfflineRunReport:
    """What actually produced a transcript, and how much of it was cached."""

    mode: str
    """``"in-process"`` or ``"attached"``."""

    pipeline: str
    asr_provider: str
    asr_model: str
    diarization_provider: str
    cache: str
    cache_hits: int | None = None
    cache_misses: int | None = None
    server_url: str | None = None

    def summary(self) -> str:
        """Render a one-line human-readable description of this run."""
        parts = [
            f"mode={self.mode}",
            f"pipeline={self.pipeline}",
            f"asr={self.asr_provider}:{self.asr_model}",
            f"diarization={self.diarization_provider}",
            f"cache={self.cache}",
        ]
        if self.server_url is not None:
            parts.append(f"server={self.server_url}")
        if self.cache_hits is not None:
            total = (self.cache_hits or 0) + (self.cache_misses or 0)
            parts.append(f"windows={total} hits={self.cache_hits} misses={self.cache_misses}")
        return " ".join(parts)


def _report_from_settings(settings: ServerSettings, asr: Any) -> OfflineRunReport:
    """Describe the local configuration, including cache counters when present."""
    hits = getattr(asr, "hits", None)
    misses = getattr(asr, "misses", None)
    return OfflineRunReport(
        mode="in-process",
        pipeline=settings.pipeline,
        asr_provider=settings.backend_asr,
        asr_model=settings.model_asr,
        diarization_provider=settings.backend_diarization,
        cache=settings.asr_cache,
        cache_hits=hits,
        cache_misses=misses,
    )


def render_source(source: Any, *, response_format: str, language: str | None) -> str:
    """Render a Transcript Source exactly as the transcription endpoint would.

    Args:
        source: The transcription's Transcript Source.
        response_format: One of the endpoint's supported ``response_format`` values.
        language: Language to report, for the formats that carry one.

    Returns:
        The response body as JSON text.

    """
    from coro.api.openai.formats import ResponseFormat
    from coro.api.openai.render import render_for_format

    return "".join(render_for_format(ResponseFormat(response_format), source, language=language))


async def transcribe_in_process(
    path: str,
    *,
    settings: ServerSettings,
    language: str | None = None,
    prompt: str | None = None,
    response_format: str = DEFAULT_RESPONSE_FORMAT,
) -> tuple[str, OfflineRunReport]:
    """Transcribe a local file with no server and no upload.

    The ASR Adapter is built lazily, so a run whose every window is already
    cached never loads the model at all.

    Args:
        path: Filesystem path to the audio or video file. Left untouched.
        settings: Server Startup Selection, as the server would use.
        language: Optional language hint.
        prompt: Optional initial prompt.
        response_format: Response shape, matching the endpoint's own values.

    Returns:
        The rendered response body and a report of what produced it.

    """
    from coro.backends.asr.factory import build_asr_adapter_stack
    from coro.backends.diarization import factory as diarization_factory
    from coro.pipelines.factory import build_pipeline
    from coro.pipelines.source import transcript_source

    asr = build_asr_adapter_stack(settings, lazy=True)

    diarization = None
    streaming_diarizer_factory = None
    if settings.backend_diarization != "none":
        diarization = diarization_factory.build_diarization_adapter(
            settings.backend_diarization,
            settings.model_diarization or "",
            device=settings.diarization_device,
            hf_token=settings.hf_token.get_secret_value() if settings.hf_token else None,
            postprocessing=settings.diarization_postprocessing,
            postprocessing_max_speakers=settings.diarization_postprocessing_max_speakers,
        )
        if settings.pipeline == "streaming" and diarization_factory.supports_streaming(
            settings.backend_diarization
        ):
            streaming_diarizer_factory = diarization_factory.build_streaming_diarizer_factory(
                settings.backend_diarization,
                diarization,
                tier=settings.diarization_latency,
            )

    pipeline = build_pipeline(
        settings,
        asr=asr,
        diarization=diarization,
        streaming_diarizer_factory=streaming_diarizer_factory,
    )
    # from_path references the file without owning it: both pipelines call
    # cleanup() in a finally, and an owning AudioInput would delete the input.
    audio = AudioInput.from_path(path)
    source = await transcript_source(pipeline, audio, language=language, prompt=prompt)
    try:
        body = render_source(source, response_format=response_format, language=language)
    finally:
        source.close()
    return body, _report_from_settings(settings, asr)


async def transcribe_attached(
    path: str,
    *,
    server_url: str,
    language: str | None = None,
    prompt: str | None = None,
    response_format: str = DEFAULT_RESPONSE_FORMAT,
) -> tuple[str, OfflineRunReport]:
    """Upload a local file to an already-running server.

    The server's response body is returned verbatim rather than parsed and
    re-rendered: it is already the same shape the in-process path produces, and
    round-tripping it through a local model would only introduce a way for the
    two to disagree.

    Args:
        path: Filesystem path to the audio or video file. Left untouched.
        server_url: Base URL of the running server.
        language: Optional language hint.
        prompt: Optional initial prompt.
        response_format: Response shape to request.

    Returns:
        The response body and a report describing the *server's* configuration,
        since that — not the local flags — is what produced it.

    Raises:
        RuntimeError: If the server rejects the request.

    """
    base = server_url.rstrip("/")
    data: dict[str, str] = {"response_format": response_format}
    if language:
        data["language"] = language
    if prompt:
        data["prompt"] = prompt

    async with build_http_client() as client:
        with Path(path).open("rb") as handle:
            response = await client.post(
                f"{base}{_TRANSCRIPTION_PATH}", files={"file": (path, handle)}, data=data
            )
        if response.status_code != 200:
            msg = f"Server returned HTTP {response.status_code}: {response.text[:500]}"
            raise RuntimeError(msg)
        body = response.text
        health = await client.get(f"{base}{_HEALTH_PATH}")

    selection = health.json().get("startup_selection", {}) if health.status_code == 200 else {}
    report = OfflineRunReport(
        mode="attached",
        pipeline=str(selection.get("pipeline", "unknown")),
        asr_provider=str(selection.get("asr_provider", "unknown")),
        asr_model=str(selection.get("asr_model", "unknown")),
        diarization_provider=str(selection.get("diarization_provider", "unknown")),
        cache="server-configured",
        server_url=base,
    )
    return body, report
