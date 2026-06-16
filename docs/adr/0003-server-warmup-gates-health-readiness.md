# Server Warmup Gates `/health` Warmup Readiness

**Server Warmup** is a first-class lifecycle stage that runs the **Configured Transcription Pipeline** against the vendored **Warmup Audio Asset** at startup, gates **Warmup Readiness** on `/health`, and fails server startup loudly on warmup failure.

## Context

Cold-model costs — JIT compilation, CUDA kernel compilation, lazy weight loading, and OS page-cache population — make the first transcription request after server startup pathologically slow compared to subsequent requests. Production clients do not run warmup before their first request. If the server advertises readiness as soon as adapters load, the first real client pays these costs silently.

There was no signal to distinguish "ready to load models" from "ready to serve real traffic." **Capability Readiness** (ASR adapter loaded) was the only health signal, but it fires before the pipeline has been exercised once.

## Decision

**Server Warmup** runs the **Configured Transcription Pipeline** against the vendored **Warmup Audio Asset** (`coro/bench/data/jfk.wav`, the whisper.cpp JFK sample) during FastAPI lifespan startup, after the **Backend Adapter Factory** finishes building adapters. The result is discarded.

Server Warmup is **enabled by default** (`CORO_WARMUP=enabled`). `CORO_WARMUP=disabled` skips warmup, logs a warning, and still reports `warmup_ready=true` (no gating requested).

`/health` gains a `warmup_ready: bool` flag. Overall readiness requires both **Capability Readiness** AND **Warmup Readiness**:

```json
{
  "ready": true,
  "warmup_ready": true,
  "capability_readiness": { "asr": true, "diarization": "disabled", "transcription": true }
}
```

Warmup failures **fail server startup loudly** — the exception propagates out of the lifespan context manager so FastAPI never completes startup and the server never accepts traffic.

The **Warmup Audio Asset** is vendored inside the package at `coro/bench/data/jfk.wav` so warmup never requires network access and works in air-gapped environments.

## Consequences

- Server startup is slower by a few seconds (one short transcription) in exchange for predictable, warm first-request latency.
- `/health` becomes a true readiness probe — load balancers and orchestrators can wait for `ready=true` before routing real traffic, confident that the pipeline has been exercised.
- A warmup failure indicates a real pipeline bug. Failing startup loudly is preferable to serving broken responses to real clients from a partially-loaded server.
- The **Warmup Audio Asset** is shared by **Server Warmup** and the opt-in **Benchmark Warmup Item** (`--warmup` CLI flag), so both use the same short clip without network access.
- `CORO_WARMUP=disabled` is an escape hatch for test environments or situations where startup speed is more important than first-request latency.

## Alternatives Considered

- **Lazy first-request warmup**: rejected — the first real client gets penalized with cold-model latency, which is exactly the problem we are solving.
- **Server Warmup as opt-in**: rejected — production deployments would routinely forget to enable it; the cost (a few seconds at startup) is small relative to the benefit.
- **Download warmup audio on first startup**: rejected — creates a silent network dependency, fails in air-gapped environments, and produces a failure mode ("server hung trying to download from GitHub") that is harder to diagnose than a ~330 KB file in the repo.
- **Synthetic silence warmup**: rejected — VAD-gated pipelines may early-exit on silence and skip diarization, leaving GPU kernels and model weights cold for the first real request.
- **Tolerate warmup failures and serve traffic anyway**: rejected — warmup failure indicates a real pipeline bug; a partially-loaded server will produce broken responses rather than slow ones, which is a worse failure mode.
