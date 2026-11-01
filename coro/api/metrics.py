"""Prometheus metrics endpoint for coro.

``GET /metrics`` publishes the readiness and the ASR admission state of this
server process, plus the process and Python runtime metrics of
``prometheus-client``, so an autoscaler can scale replicas on queued inference
rather than on CPU alone. Every value is per process: a deployment with several
replicas is aggregated by the scraper.

Each app owns its :class:`~prometheus_client.CollectorRegistry` instead of the
library's global one, so building several apps in one process (the tests do)
never registers a collector twice. The coro values are read from the runtime at
scrape time by :class:`RuntimeCollector`; nothing on the inference path updates
a metric object.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from fastapi import APIRouter, Request, Response
from prometheus_client import CollectorRegistry, GCCollector, PlatformCollector, ProcessCollector
from prometheus_client.core import (
    CounterMetricFamily,
    GaugeMetricFamily,
    InfoMetricFamily,
    Metric,
)
from prometheus_client.exposition import choose_encoder

import coro
from coro.cache.adapter import unwrap_asr_adapter

if TYPE_CHECKING:
    from coro.runtime import RuntimeState

router = APIRouter()


# MARK: Snapshot
class AdmissionStats(Protocol):
    """The read-only view of an ASR Adapter's admission controller that metrics need."""

    @property
    def max_concurrency(self) -> int: ...
    @property
    def max_queue_depth(self) -> int: ...
    @property
    def in_flight(self) -> int: ...
    @property
    def queued(self) -> int: ...
    @property
    def admitted_total(self) -> int: ...
    @property
    def rejected_total(self) -> int: ...


@dataclass(frozen=True)
class MetricsSnapshot:
    """The coro values one scrape publishes, read from the runtime at one instant."""

    ready: bool
    build_labels: dict[str, str]
    admission: AdmissionStats | None


def snapshot_runtime(runtime: RuntimeState) -> MetricsSnapshot:
    """Read the coro metric values from the runtime.

    The admission controller is read beneath the ASR window cache, so the ASR
    metrics count model calls; a cache hit never takes a permit.
    """
    adapter = runtime.asr_adapter
    admission = None if adapter is None else getattr(unwrap_asr_adapter(adapter), "admission", None)
    return MetricsSnapshot(
        ready=runtime.ready and runtime.warmup_ready,
        build_labels={
            "version": coro.__version__,
            "pipeline": runtime.pipeline_selector,
            "asr_provider": runtime.asr_provider,
            "asr_model": runtime.asr_model,
            "diarization_provider": runtime.diarization_provider,
        },
        admission=admission,
    )


# MARK: Collector
def snapshot_metrics(snapshot: MetricsSnapshot) -> list[Metric]:
    """Return the metric families of one snapshot.

    ASR admission metrics are omitted while no ASR Adapter with an admission
    controller is loaded.
    """
    metrics: list[Metric] = [
        InfoMetricFamily(
            "coro_build",
            "Version and Server Startup Selection of this process.",
            value=snapshot.build_labels,
        ),
        GaugeMetricFamily(
            "coro_ready",
            "1 once Capability Readiness and Warmup Readiness both hold, else 0.",
            value=int(snapshot.ready),
        ),
    ]
    stats = snapshot.admission
    if stats is None:
        return metrics
    return [
        *metrics,
        GaugeMetricFamily(
            "coro_asr_inference_permits",
            "ASR inference calls allowed to run at once.",
            value=stats.max_concurrency,
        ),
        GaugeMetricFamily(
            "coro_asr_inference_queue_capacity",
            "ASR inference calls allowed to wait for a permit.",
            value=stats.max_queue_depth,
        ),
        GaugeMetricFamily(
            "coro_asr_inferences_in_flight",
            "ASR inference calls holding a permit.",
            value=stats.in_flight,
        ),
        GaugeMetricFamily(
            "coro_asr_inferences_queued",
            "ASR inference calls waiting for a permit.",
            value=stats.queued,
        ),
        CounterMetricFamily(
            "coro_asr_inferences_admitted",
            "ASR inference calls granted a permit.",
            value=stats.admitted_total,
        ),
        CounterMetricFamily(
            "coro_asr_inferences_rejected",
            "ASR inference calls rejected because the queue was full.",
            value=stats.rejected_total,
        ),
    ]


class RuntimeCollector:
    """A Prometheus collector that reads the coro metrics from the runtime at each scrape.

    It takes a callable rather than the runtime itself because the app's runtime
    can be replaced after the registry is built, as the tests do.
    """

    def __init__(self, runtime: Callable[[], RuntimeState]) -> None:
        """Create a collector over the runtime that ``runtime`` returns."""
        self._runtime = runtime

    def collect(self) -> Iterator[Metric]:
        """Yield the coro metric families for one scrape."""
        yield from snapshot_metrics(snapshot_runtime(self._runtime()))


def build_metrics_registry(runtime: Callable[[], RuntimeState]) -> CollectorRegistry:
    """Build the registry one app publishes: the coro metrics plus process and Python metrics.

    Args:
        runtime: Returns the app's current runtime state.

    Returns:
        A registry owned by one app, never the library's global one.

    """
    registry = CollectorRegistry()
    registry.register(RuntimeCollector(runtime))
    ProcessCollector(registry=registry)
    PlatformCollector(registry=registry)
    GCCollector(registry=registry)
    return registry


# MARK: Route
@router.get("/metrics", include_in_schema=False)
async def metrics(request: Request) -> Response:
    """Publish this process's metrics, in the format the scraper's ``Accept`` header asks for."""
    encoder, content_type = choose_encoder(request.headers.get("accept", ""))
    return Response(encoder(request.app.state.metrics_registry), media_type=content_type)
