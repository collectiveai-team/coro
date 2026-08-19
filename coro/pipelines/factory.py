"""Construction of the Configured Transcription Pipeline from settings.

Pipelines are thin orchestrators over adapters, so building one is cheap and
building a second one over a *different* adapter is a legitimate way to reach a
different model path — which is how Server Warmup bypasses the ASR window cache,
and how the offline command runs the same pipeline the server would.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from coro.core.protocols import ASRAdapter, DiarizationAdapter
    from coro.settings import ServerSettings


def build_pipeline(
    settings: ServerSettings,
    *,
    asr: ASRAdapter,
    diarization: DiarizationAdapter | None = None,
    streaming_diarizer_factory: Any | None = None,
) -> Any:
    """Build the pipeline named by the Configured Transcription Pipeline selector.

    Args:
        settings: Server Startup Selection.
        asr: The ASR Adapter the pipeline should call.
        diarization: Optional Diarization Adapter for the Full-Memory Pipeline.
        streaming_diarizer_factory: Optional per-request diarizer factory for the
            Streaming Pipeline.

    Returns:
        A ready transcription pipeline, typed as ``Any`` for the same reason
        ``RuntimeState.pipeline`` is: ``StreamingPipeline.stream`` is an async
        generator whose inferred yield type is wider than the protocol's, so
        naming the protocol here would be a type error rather than a stronger
        guarantee.

    """
    if settings.pipeline == "streaming":
        from coro.pipelines.streaming import StreamingPipeline

        return StreamingPipeline(
            asr=asr,
            streaming_diarizer_factory=streaming_diarizer_factory,
            spill_dir=settings.transcript_spill_dir,
        )

    from coro.pipelines.full_memory import FullMemoryPipeline

    return FullMemoryPipeline(asr=asr, diarization=diarization)
