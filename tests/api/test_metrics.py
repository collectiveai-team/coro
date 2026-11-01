"""GET /metrics: readiness and ASR admission state in the Prometheus formats."""

from __future__ import annotations

import asyncio
import sys

import pytest
from httpx import ASGITransport, AsyncClient
from prometheus_client.openmetrics.parser import (
    text_string_to_metric_families as parse_openmetrics,
)
from prometheus_client.parser import text_string_to_metric_families

from coro.app import create_app
from coro.backends.asr.concurrency import AdmissionController, AsrCapacityError
from coro.cache.adapter import CachingASRAdapter
from coro.cache.store import ASRCacheStore
from coro.runtime import RuntimeState
from coro.settings import ServerSettings
from support.adapters import RecordingASR

PCM = b"\x00\x00" * 1600
OPENMETRICS = "application/openmetrics-text; version=1.0.0"


class AdmittedASR(RecordingASR):
    """An in-memory ASR Adapter that takes a permit from a real admission controller."""

    def __init__(self, admission: AdmissionController) -> None:
        super().__init__()
        self.admission = admission
        self.admitted = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    async def transcribe_pcm(self, pcm, *, language=None, prompt=None):
        """Hold a permit until ``release`` is set, then answer as RecordingASR."""
        async with self.admission.admit():
            self.admitted.set()
            await self.release.wait()
            return await super().transcribe_pcm(pcm, language=language, prompt=prompt)


async def _scrape(runtime: RuntimeState, *, accept: str | None = None):
    app = create_app(ServerSettings(_env_file=None))
    app.state.runtime = runtime
    headers = {"accept": accept} if accept else {}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.get("/metrics", headers=headers)


def _samples(body: str) -> dict[str, float]:
    """Parse a text exposition into sample name -> value, for unlabelled samples."""
    return {
        sample.name: sample.value
        for family in text_string_to_metric_families(body)
        for sample in family.samples
        if not sample.labels
    }


@pytest.mark.asyncio
async def test_a_server_without_a_model_reports_not_ready_and_no_asr_metrics():
    response = await _scrape(RuntimeState(asr_adapter=None))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain; version=0.0.4")
    samples = _samples(response.text)
    assert samples["coro_ready"] == 0
    assert not [name for name in samples if name.startswith("coro_asr_")]


@pytest.mark.asyncio
async def test_a_warmed_up_server_reports_ready_and_its_admission_limits():
    adapter = AdmittedASR(AdmissionController(max_concurrency=3, max_queue_depth=7))
    await adapter.transcribe_pcm(PCM)
    await adapter.transcribe_pcm(PCM)

    response = await _scrape(RuntimeState(asr_adapter=adapter, warmup_ready=True))

    samples = _samples(response.text)
    assert samples["coro_ready"] == 1
    assert samples["coro_asr_inference_permits"] == 3
    assert samples["coro_asr_inference_queue_capacity"] == 7
    assert samples["coro_asr_inferences_in_flight"] == 0
    assert samples["coro_asr_inferences_queued"] == 0
    assert samples["coro_asr_inferences_admitted_total"] == 2
    assert samples["coro_asr_inferences_rejected_total"] == 0


@pytest.mark.asyncio
async def test_a_full_queue_shows_as_in_flight_and_rejected():
    adapter = AdmittedASR(AdmissionController(max_concurrency=1, max_queue_depth=0))
    adapter.release.clear()
    holder = asyncio.create_task(adapter.transcribe_pcm(PCM))
    await adapter.admitted.wait()
    with pytest.raises(AsrCapacityError):
        await adapter.transcribe_pcm(PCM)

    response = await _scrape(RuntimeState(asr_adapter=adapter, warmup_ready=True))
    adapter.release.set()
    await holder

    samples = _samples(response.text)
    assert samples["coro_asr_inferences_in_flight"] == 1
    assert samples["coro_asr_inferences_rejected_total"] == 1


@pytest.mark.asyncio
async def test_admission_is_read_beneath_the_asr_window_cache(tmp_path):
    inner = AdmittedASR(AdmissionController(max_concurrency=2, max_queue_depth=4))
    with ASRCacheStore(str(tmp_path / "cache"), max_bytes=0, ttl_seconds=0.0) as store:
        cached = CachingASRAdapter(inner, store=store, fingerprint="fp", honours_prompt=False)
        await cached.transcribe_pcm(PCM)
        await cached.transcribe_pcm(PCM)
        response = await _scrape(RuntimeState(asr_adapter=cached, warmup_ready=True))

    samples = _samples(response.text)
    assert samples["coro_asr_inference_permits"] == 2
    assert samples["coro_asr_inferences_admitted_total"] == 1


@pytest.mark.asyncio
async def test_build_info_carries_the_startup_selection_verbatim():
    runtime = RuntimeState(asr_adapter=None, asr_model='odd"model\\name')

    response = await _scrape(runtime)

    (family,) = [
        f for f in text_string_to_metric_families(response.text) if f.name == "coro_build_info"
    ]
    assert family.samples[0].labels["asr_model"] == 'odd"model\\name'
    assert family.samples[0].value == 1


@pytest.mark.asyncio
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="process metrics read /proc")
async def test_process_metrics_are_published_alongside():
    response = await _scrape(RuntimeState(asr_adapter=None))

    samples = _samples(response.text)
    assert samples["process_resident_memory_bytes"] > 0
    assert "process_cpu_seconds_total" in samples


@pytest.mark.asyncio
async def test_a_scraper_asking_for_openmetrics_gets_it():
    adapter = AdmittedASR(AdmissionController(max_concurrency=2, max_queue_depth=4))

    response = await _scrape(
        RuntimeState(asr_adapter=adapter, warmup_ready=True), accept=OPENMETRICS
    )

    assert response.headers["content-type"].startswith("application/openmetrics-text")
    names = {family.name for family in parse_openmetrics(response.text)}
    assert {"coro_ready", "coro_asr_inferences_admitted"} <= names


@pytest.mark.asyncio
async def test_two_apps_in_one_process_each_publish_their_own_runtime():
    first = AdmittedASR(AdmissionController(max_concurrency=1, max_queue_depth=0))
    second = AdmittedASR(AdmissionController(max_concurrency=5, max_queue_depth=0))

    first_samples = _samples((await _scrape(RuntimeState(asr_adapter=first))).text)
    second_samples = _samples((await _scrape(RuntimeState(asr_adapter=second))).text)

    assert first_samples["coro_asr_inference_permits"] == 1
    assert second_samples["coro_asr_inference_permits"] == 5


@pytest.mark.asyncio
async def test_metrics_stay_out_of_the_published_openapi_contract():
    app = create_app(ServerSettings(_env_file=None))
    assert "/metrics" not in app.openapi()["paths"]
