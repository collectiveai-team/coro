# ASR Benchmarking Context

This context defines the language used to compare custom ASR server implementations and decide whether disk-backed chunking is worthwhile.

## Language

**Server Process Tree**:
The benchmark target consisting of the configured server process and all descendant processes created while handling a request.
_Avoid_: Root PID only, single server PID

**Dynamic Process Tree**:
A server process tree whose descendant membership is rediscovered on every resource sample.
_Avoid_: Startup process tree

**Benchmark Run**:
A controlled comparison of one server implementation against another under the same audio input, server configuration, and hardware profile.
_Avoid_: Ad hoc timing, one-off curl

**Performance Benchmark**:
A benchmark run that measures resource usage and timing of the server process tree across a workload set, without scoring transcription quality.
_Avoid_: Resource Benchmark (legacy), accuracy benchmark

**Quality Benchmark**:
A benchmark run that scores transcription and diarization output of the server against reference STM files using MeetEval metrics, without making resource claims.
_Avoid_: Accuracy benchmark, WER-only benchmark

**Workload Item**:
One audio input plus its reference STM, transcribed once per repetition and scored once per benchmark run; the unit aggregated by both Performance Benchmark and Quality Benchmark.
_Avoid_: Single audio file, test case

**Workload Set**:
The ordered collection of workload items processed sequentially in one benchmark run.
_Avoid_: Test suite, audio batch

**Reference STM**:
The canonical reference format for both transcript and diarization quality scoring, holding speaker-attributed segments per workload item.
_Avoid_: Reference RTTM, reference transcript text file

**Hypothesis STM**:
The server's transcription response converted to STM, written per workload item and consumed by MeetEval.
_Avoid_: Hypothesis Diarization, response JSON

**MeetEval Metric Set**:
The fixed set of MeetEval scores reported for each workload item: siWER, cpWER, ORC-WER (greedy), DI-cpWER (greedy), and DER.
_Avoid_: WER alone, custom metric mix

**Server Warmup**:
A pipeline execution against a fixed warmup audio at server startup, completed before the server reports ready, so the first transcription endpoint request does not pay cold-model costs.
_Avoid_: Lazy first-request warmup, client-driven warmup

**Warmup Readiness**:
The /health flag indicating the configured transcription pipeline has completed Server Warmup and is ready to serve real requests.
_Avoid_: Capability Readiness conflation

**Benchmark Warmup Item**:
An audio request issued by the benchmark client before the measured workload set, whose response and resource samples are discarded, used to neutralize transient cold-cache effects between client and server.
_Avoid_: Real workload item with discarded results

**Warmup Audio Asset**:
The vendored short audio (whisper.cpp JFK sample) shared by Server Warmup and Benchmark Warmup, distributed inside the package so warmup never requires network access.
_Avoid_: Downloaded-on-demand warmup, synthetic silence

**Bench-Managed Server**:
A server subprocess started by the benchmark client, configured via bench CLI flags translated to ASR_DIAR_ environment variables, with /health polled until Warmup Readiness before the workload set runs and torn down on bench completion or failure.
_Avoid_: In-process TestClient bench, manually-started server assumption

**Bench-Attached Server**:
A pre-existing server process the benchmark client connects to via --server-url and samples via --server-pid or --server-match, without managing its lifecycle.
_Avoid_: Bench owning a server it did not start

**Time To First Delta**:
Wall time from request start to receipt of the first `transcript.text.delta` SSE event, measured per workload item × rep when streaming is enabled in a Performance Benchmark.
_Avoid_: Time to first byte, time to consolidated response

**Backfilled Performance Column**:
A Resource CSV column whose per-(item × rep) scalar value (wall_seconds, transcription_throughput, time_to_first_delta_s) is repeated on every sampled row after the request completes.
_Avoid_: Final-row-only performance field, Backfilled Quality Column (deprecated)

**Resource CSV**:
The per-repetition sampled metrics file containing memory, IO, CPU, and GPU observations for the server process tree.
_Avoid_: Memory CSV

**Stable Resource Schema**:
A resource CSV schema that writes the same columns for every benchmark run, even when some metrics are unsupported or empty.
_Avoid_: Machine-specific CSV schema

**Sample Rate Field**:
A resource CSV field derived from the difference between consecutive cumulative counter samples divided by the observed sample duration.
_Avoid_: Average rate when referring to per-interval samples

**Observed Hardware Profile**:
The hardware mode inferred from measured runtime behavior during a benchmark run.
_Avoid_: Declared hardware profile, launch profile

**CPU+GPU Run**:
An observed hardware profile where the server process tree shows GPU memory use or sustained GPU utilization during the request.
_Avoid_: GPU-visible run

**CPU-Only Run**:
An observed hardware profile where the server process tree shows no GPU memory use and no sustained GPU utilization during the request.
_Avoid_: No-GPU-machine run

**Process-Tree PSS**:
The primary memory metric for a benchmark run, summing proportional set size across the server process tree.
_Avoid_: Headline RSS, root-PID RSS

**Process-Tree USS**:
The private-memory metric for a benchmark run, summing unique set size across the server process tree to reveal private growth or leaks.
_Avoid_: Private RSS

**Logical IO Rate**:
The rate of bytes read or written through process IO interfaces across the server process tree.
_Avoid_: Disk IO rate when referring to `rchar` or `wchar`

**Physical IO Rate**:
The rate of bytes actually read from or written to storage across the server process tree.
_Avoid_: File IO rate when referring to `read_bytes` or `write_bytes`

**Process-Tree CPU Rate**:
The interval CPU consumption rate across the server process tree, normalized so 100 percent means one fully used CPU core.
_Avoid_: Request CPU time when referring to sampled CPU rate

**Transcription Throughput**:
The amount of input audio duration processed per wall-clock second during a benchmark run.
_Avoid_: Speed when not specifying audio-seconds per wall-second

**Workload Set**:
The collection of audio inputs used to compare server implementations across meaningful duration classes.
_Avoid_: Single test file when deciding architecture value

**Transcription Endpoint**:
The `/v1/audio/transcriptions` HTTP route that accepts audio transcription requests for the packaged ASR diarization service.
_Avoid_: Pipeline endpoint, behavior-specific endpoint

**Supported Endpoint Set**:
The intentionally exposed server endpoints for the packaged ASR diarization service.
_Avoid_: Every route from the prototype scripts

**Transcription Pipeline**:
The end-to-end processing path that turns an uploaded audio file into transcript, diarization, and raw-word response data.
_Avoid_: Endpoint handler, model call

**Full-Memory Pipeline**:
A transcription pipeline implementation that decodes the entire uploaded audio into PCM before chunked ASR processing.
_Avoid_: v1 pipeline, memory pipeline

**Chunked-File Pipeline**:
A legacy transcription pipeline implementation that spools uploaded audio to a file but still materializes decoded PCM before downstream processing.
_Avoid_: v2 pipeline, disk pipeline

**Streaming Pipeline**:
A transcription pipeline implementation that streams uploaded audio through decoding, ASR windowing, and diarization without materializing the full upload or decoded PCM.
_Avoid_: Chunked-File Pipeline, disk-backed pipeline, v2 pipeline

**Streaming Diarization Feed**:
A diarization flow where sequential PCM chunks are inserted into the online diarization model during decoding, while speaker assignment remains part of final response construction.
_Avoid_: Live speaker deltas, post-hoc full-PCM diarization

**No-Disk Audio Flow**:
A streaming audio flow where request audio is piped directly into ffmpeg rather than written to request-scoped temporary files.
_Avoid_: Temp-file fallback, disk staging, upload spooling

**ASR Windowing**:
The shared process of transcribing PCM in overlapping windows and emitting accepted transcript deltas per window.
_Avoid_: Pipeline versioning, full-audio ASR call

**Incremental ASR Windowing**:
ASR windowing fed by sequential PCM chunks while preserving the configured window and overlap semantics of full-buffer ASR windowing.
_Avoid_: Streaming ASR tuning, changed ASR windows

**Configured Transcription Pipeline**:
The transcription pipeline implementation selected at server startup for the public transcription endpoint.
_Avoid_: API version, route version

**Pipeline Dependency**:
A FastAPI dependency that provides the configured transcription pipeline to the endpoint.
_Avoid_: Direct route app-state lookup, per-request pipeline construction

**Singleton Runtime**:
The process-wide container for loaded adapters and the configured transcription pipeline.
_Avoid_: Per-request model load, route-owned runtime

**Server Startup Selection**:
The environment-backed configuration that chooses pipeline behavior, backend providers, and model selections before serving requests.
_Avoid_: Per-request model selection, route version selection

**Pipeline Selector Removal**:
The intentional removal of an obsolete configured transcription pipeline value rather than preserving it as a compatibility alias.
_Avoid_: Silent selector alias, deprecated pipeline fallback

**Settings Dependency**:
A FastAPI dependency that provides validated server settings to API code.
_Avoid_: Direct route app-state lookup, ad hoc environment reads

**Strict Startup Validation**:
Startup-time rejection of unknown pipeline, backend provider, or model-selection values.
_Avoid_: Silent fallback, request-time selector validation

**Capability Readiness**:
The health-report distinction between ASR availability, optional diarization availability, and overall transcription readiness.
_Avoid_: Single backend status, diarization-required readiness

**Audio Input**:
A package-owned representation of an uploaded audio file that can provide bytes or a temporary file path to a transcription pipeline.
_Avoid_: FastAPI UploadFile, raw bytes, temp path only

**Audio Input Cleanup**:
The audio-input-owned lifecycle that removes temporary files after transcription or streaming completes.
_Avoid_: Pipeline unlink, endpoint unlink

**Pipeline Module**:
The package area that orchestrates audio IO, ASR adapters, diarization adapters, and core response transformations for a transcription pipeline.
_Avoid_: API router, core model, ASR adapter

**Transcription API Contract**:
The versioned form-request and JSON/SSE-response shape used by transcription endpoints.
_Avoid_: Internal result dict, cleanup opportunity

**Pipeline-Internal Streaming**:
Streaming behavior contained behind the configured transcription pipeline while preserving the public transcription API contract.
_Avoid_: New transcription endpoint, behavior-specific API route

**Boundary Response Schema**:
A Pydantic model used to serialize successful transcription responses and OpenAI-style error responses at the API boundary.
_Avoid_: Internal pipeline model, request form model

**Strict Transcription Response Schema**:
The boundary response schema containing only the public transcription fields.
_Avoid_: Backend-native extras, permissive response object

**OpenAI-Compatible Request**:
A transcription form request that accepts OpenAI-style parameters for client compatibility without requiring every OpenAI response format.
_Avoid_: Full OpenAI API clone

**JSON Response Format Alias**:
A supported `response_format` value that maps to the same strict transcription JSON response.
_Avoid_: Separate output contract, OpenAI response-format clone

**Compatibility Model Field**:
The OpenAI-compatible request `model` field accepted by the transcription endpoint but not used for runtime model selection.
_Avoid_: Per-request model selection, configured model

**OpenAI-Style Error**:
A transcription endpoint error response shaped as an OpenAI-style `error` object rather than FastAPI's default `detail` object.
_Avoid_: FastAPI default error

**Transcription Exception Handler**:
An app-level handler that converts typed transcription exceptions into OpenAI-style error JSON responses.
_Avoid_: Inline error response branches, FastAPI default handler

**Transcription Response**:
The enriched transcription response containing transcript segments, word segments, diarization, transcript convenience data, and raw words.
_Avoid_: Minimal OpenAI text response

**OpenAI-Exact SSE**:
The streaming transcription event contract that matches OpenAI event framing without package-specific progress events.
_Avoid_: Progress extension, package-specific SSE

**Core Boundary**:
The package boundary containing API-agnostic datamodels, interfaces, and pure transcript or diarization transformations.
_Avoid_: Shared FastAPI utilities, route helpers

**Project-Owned Transcript Model**:
A core token, segment, speaker timeline, or response model defined by this package rather than by an ASR backend library.
_Avoid_: Whisperlivekit token, backend segment

**Audio Module**:
The package area for audio decoding, PCM streaming, upload spooling, and audio constants.
_Avoid_: API utility, core helper, ASR adapter

**ASR Adapter**:
A model integration that produces transcript tokens from audio while hiding backend-specific model APIs.
_Avoid_: Transcription engine wrapper, route model

**Adapter Concurrency Policy**:
The adapter-owned rule for serializing or allowing concurrent model calls based on backend safety.
_Avoid_: Global pipeline lock, assumed backend safety

**ML Model Integration**:
A package module under `backends/` that adapts an external ASR or diarization backend.
_Avoid_: Pydantic model, response schema

**Backend Provider**:
An external integration selected to provide ASR or diarization model access for the server.
_Avoid_: Model name, capability type

**Provider-First Backend Layout**:
A `backends/` package structure organized by backend provider rather than by ASR or diarization capability.
_Avoid_: Capability-first backend tree, duplicated provider setup

**ASR Model Selection**:
The Hugging Face-style model identifier passed to the configured ASR backend provider.
_Avoid_: ASR backend, provider name

**Diarization Model Selection**:
The Hugging Face-style model identifier passed to the configured diarization backend provider.
_Avoid_: Diarization backend, provider name

**Diarization Adapter**:
A model integration that produces speaker timeline segments from audio while hiding backend-specific diarization APIs.
_Avoid_: ASR backend, speaker helper

**Diarization Flow**:
The adapter-owned choice of batch or incremental speaker timeline generation for a transcription pipeline.
_Avoid_: Pipeline-owned diarization algorithm, forced batch diarization

**ASR-Only Server**:
A valid server configuration with an ASR adapter and no diarization adapter.
_Avoid_: Not-ready server, failed diarization server

**Backend Adapter Factory**:
Startup code that creates ASR and diarization adapters from backend provider and model selections.
_Avoid_: Pipeline-owned backend construction, direct provider calls

## Relationships

- A **Benchmark Run** processes a **Workload Set** of **Workload Item** values sequentially against one server process tree.
- A **Server Process Tree** is sampled as a **Dynamic Process Tree** during each workload item request.
- A **Performance Benchmark** writes one **Resource CSV** per (workload item × repetition) plus a run-level performance summary aggregating across the workload set.
- A **Quality Benchmark** writes one MeetEval result per workload item plus a run-level quality summary produced by `combine_error_rates` across the workload set.
- A **Performance Benchmark** and a **Quality Benchmark** may share one benchmark run; both consume the same hypothesis from the same request.
- A **Resource CSV** uses a **Stable Resource Schema** across hardware profiles and contains only resource and timing columns; quality columns are not embedded.
- A **Quality Benchmark** computes the **MeetEval Metric Set** for each workload item using its **Reference STM** and the converted **Hypothesis STM**.
- A **Resource CSV** contains both cumulative counters and **Sample Rate Field** values.
- An **Observed Hardware Profile** is inferred from measurements, not from how the server was launched.
- A **CPU+GPU Run** requires GPU activity attributable to the **Server Process Tree**.
- A **CPU-Only Run** can occur even when a GPU is visible but unused by the **Server Process Tree**.
- **Process-Tree PSS** is the headline memory comparison for a **Performance Benchmark**.
- **Process-Tree USS** is the private-growth companion to **Process-Tree PSS**.
- **Logical IO Rate** describes pipeline work during a **Performance Benchmark**.
- **Physical IO Rate** describes storage pressure during a **Performance Benchmark**.
- **Process-Tree CPU Rate** describes compute pressure during a **Performance Benchmark**.
- **Transcription Throughput** is the headline timing comparison for a **Performance Benchmark**.
- A **Workload Set** for deciding disk-backed chunking value includes short, medium, and long audio inputs.
- **Server Warmup** runs at startup against the **Warmup Audio Asset**, gates **Warmup Readiness** on the `/health` response, and is enabled by default.
- **Server Warmup** failures fail server startup loudly rather than allowing degraded readiness.
- A **Benchmark Warmup Item** is optional, opt-in via the benchmark CLI, runs once before the first measured workload item, and reuses the **Warmup Audio Asset**.
- The **Warmup Audio Asset** is vendored inside the package so neither **Server Warmup** nor **Benchmark Warmup Item** requires network access.
- A **Benchmark Run** uses a **Bench-Managed Server** by default and a **Bench-Attached Server** when `--server-url` is passed; the two modes are mutually exclusive.
- A **Bench-Managed Server** is configured by translating bench CLI flags into the same `ASR_DIAR_` environment variables used for **Server Startup Selection**.
- A **Bench-Managed Server** is considered ready only once `/health` reports both **Capability Readiness** and **Warmup Readiness**.
- Diarization is enabled by default for both quality and performance subcommands so the **Quality Benchmark** can report cpWER, ORC-WER, DI-cpWER, and DER, and the **Performance Benchmark** measures the production-shaped pipeline.
- The **Supported Endpoint Set** contains `/health` and the `/v1/audio/transcriptions` **Transcription Endpoint** only.
- The **Transcription Endpoint** receives the **Configured Transcription Pipeline** through a **Pipeline Dependency**.
- API code receives **Server Startup Selection** through a **Settings Dependency**.
- A **Pipeline Dependency** returns the **Configured Transcription Pipeline** from the **Singleton Runtime**.
- The default **Configured Transcription Pipeline** is the **Full-Memory Pipeline**.
- The **Chunked-File Pipeline** is selected with the startup value `chunked-file`; the **Full-Memory Pipeline** is selected with `full-memory`.
- **Server Startup Selection** uses the `ASR_DIAR_` environment prefix for pipeline, backend provider, and model selection settings.
- **Server Startup Selection** uses **Strict Startup Validation** for selector values.
- The default ASR **Backend Provider** is `whisperlivekit`.
- The default diarization **Backend Provider** is `none`.
- The default **ASR Model Selection** is `openai/whisper-medium`.
- When whisperlivekit diarization is enabled without an explicit **Diarization Model Selection**, the default is `nvidia/diar_sortformer_4spk-v1`.
- A **Configured Transcription Pipeline** preserves the public **Transcription API Contract** while changing internal processing behavior.
- `/health` reports **Server Startup Selection**, **Capability Readiness**, and **Warmup Readiness** rather than one ambiguous backend field.
- The **Full-Memory Pipeline** and **Chunked-File Pipeline** both use shared **ASR Windowing**; they differ in how PCM is sourced.
- A **Transcription Pipeline** receives **Audio Input** rather than FastAPI upload objects, raw bytes only, or temporary file paths only.
- **Audio Input** owns **Audio Input Cleanup** for any temporary file it creates.
- A **Pipeline Module** owns orchestration for one or more **Transcription Pipeline** implementations.
- A **Transcription API Contract** is preserved by the **Transcription Endpoint** unless a new public contract is intentionally introduced.
- A **Boundary Response Schema** defines response and error JSON shapes without replacing multipart form parsing with a request body model.
- A **Strict Transcription Response Schema** prevents backend-native fields from leaking into public transcription responses.
- An **OpenAI-Compatible Request** returns a **Transcription Response** in the current package contract.
- Supported **JSON Response Format Alias** values do not change the response schema.
- A **Compatibility Model Field** never overrides **Server Startup Selection**.
- Transcription endpoints return **OpenAI-Style Error** responses for request and processing failures.
- **Transcription Exception Handler** converts typed validation, readiness, and processing failures into **OpenAI-Style Error** responses.
- Streaming transcription uses **OpenAI-Exact SSE** rather than package-specific progress events.
- The **Core Boundary** excludes FastAPI request parsing, response classes, route dependencies, and HTTP errors.
- A **Project-Owned Transcript Model** crosses package boundaries; backend-native types are converted at adapter edges.
- The **Audio Module** owns ffmpeg and PCM IO concerns outside the **Core Boundary**.
- An **ASR Adapter** and a **Diarization Adapter** are separate capabilities that orchestration combines into an API response.
- An **ASR Adapter** owns its **Adapter Concurrency Policy**.
- An **ASR-Only Server** is ready when the ASR adapter is available, even if no **Diarization Adapter** is configured.
- A **Transcription Pipeline** depends on **ASR Adapter** and **Diarization Adapter** protocols, not directly on a **Backend Provider**.
- A **Diarization Adapter** owns its **Diarization Flow** while preserving the speaker timeline output expected by pipelines.
- A **Backend Adapter Factory** creates adapters during startup from configured backend providers and model selections.
- A **ML Model Integration** is distinct from core schemas and internal data types.
- A **Backend Provider** may provide ASR, diarization, or both capabilities.
- **ML Model Integration** modules use a **Provider-First Backend Layout**.
- **ASR Model Selection** and **Diarization Model Selection** are configured independently, even when they use the same **Backend Provider**.

## Example Dialogue

> **Dev:** "Should the benchmark track only the PID passed with `--server-pid`?"
> **Domain expert:** "No — compare the whole **Server Process Tree**, because upload handling, ffmpeg, workers, and model execution may live in child processes."

> **Dev:** "Can we discover child processes once before the request starts?"
> **Domain expert:** "No — sample a **Dynamic Process Tree** so transient children like ffmpeg are included."

> **Dev:** "Can we compare summed RSS across the process tree?"
> **Domain expert:** "Keep RSS as secondary data, but use **Process-Tree PSS** as the headline because shared pages would otherwise be double-counted."

> **Dev:** "If v1 streams through temp files, should we look only at disk writes?"
> **Domain expert:** "No — compare **Logical IO Rate** for pipeline work and **Physical IO Rate** for actual storage pressure, because page cache can hide physical IO."

> **Dev:** "Can request wall time tell us whether v1 costs more CPU?"
> **Domain expert:** "No — include **Process-Tree CPU Rate** because disk-backed chunking may trade memory for CPU work."

> **Dev:** "Should total wall time be the headline timing result?"
> **Domain expert:** "Keep it, but use **Transcription Throughput** as the headline so results compare across audio durations."

> **Dev:** "Can one short audio file tell us whether disk-backed chunking matters?"
> **Domain expert:** "No — use a **Workload Set** with short, medium, and long inputs because memory savings scale with input duration."

> **Dev:** "Should resource numbers be ignored when transcripts differ?"
> **Domain expert:** "Not for the **Performance Benchmark** — quality is scored separately by the **Quality Benchmark**, while performance is about resource usage and timing."

> **Dev:** "Can we compute WER or DER from whatever looks like ground truth in the repo?"
> **Domain expert:** "No — score a **Quality Benchmark** only against an explicit **Reference STM** for each **Workload Item**."

> **Dev:** "Does the server need to return RTTM for diarization scoring?"
> **Domain expert:** "No — the benchmark converts the transcription response to a **Hypothesis STM** and MeetEval scores DER against the **Reference STM**."

> **Dev:** "Should we report only WER for the **Quality Benchmark**?"
> **Domain expert:** "No — report the **MeetEval Metric Set** (siWER, cpWER, ORC-WER, DI-cpWER, DER) because each captures a different speaker-attribution assumption."

> **Dev:** "Should the sampled metrics file still be called `memory_N.csv`?"
> **Domain expert:** "No — use a **Resource CSV** because the file contains memory, IO, CPU, and GPU observations."

> **Dev:** "Should CPU-only and GPU runs have different CSV columns?"
> **Domain expert:** "No — use a **Stable Resource Schema** so notebooks can compare runs without schema branching."

> **Dev:** "Where should MeetEval scores be stored?"
> **Domain expert:** "In a **Quality Benchmark** artifact per **Workload Item**, plus a run-level summary — not embedded in the **Resource CSV**."

> **Dev:** "Should the **Resource CSV** still carry single WER and DER columns?"
> **Domain expert:** "No — quality lives in the **Quality Benchmark** artifact; the **Resource CSV** holds only resource and timing fields."

> **Dev:** "Should the server be considered ready as soon as adapters load?"
> **Domain expert:** "No — `/health` waits for **Warmup Readiness** because **Server Warmup** must run before the first real request."

> **Dev:** "Should the benchmark download a warmup clip on first run?"
> **Domain expert:** "No — the **Warmup Audio Asset** is vendored, so neither **Server Warmup** nor a **Benchmark Warmup Item** needs network access."

> **Dev:** "Should we write only IO rates?"
> **Domain expert:** "No — write cumulative counters and **Sample Rate Field** values so rates can be audited or recomputed."

> **Dev:** "Should we label a run CPU-only because we hid CUDA at launch?"
> **Domain expert:** "No — use the **Observed Hardware Profile** inferred from measured runtime behavior."

> **Dev:** "If `nvidia-smi` shows an idle GPU, is that a **CPU+GPU Run**?"
> **Domain expert:** "No — it is only a **CPU+GPU Run** when GPU activity is attributable to the **Server Process Tree**."

> **Dev:** "Should full-memory and chunked processing be exposed as `/v1` and `/v2` routes?"
> **Domain expert:** "No — they are **Transcription Pipeline** choices behind one **Transcription Endpoint**, selected as the **Configured Transcription Pipeline** at startup."

> **Dev:** "Should the packaged server keep `/`, `/asr`, `/v1/listen`, `/v1/models`, or behavior-specific transcription routes because the prototype has them?"
> **Domain expert:** "No — the **Supported Endpoint Set** is limited to `/health` and `/v1/audio/transcriptions`."

> **Dev:** "Should full-memory and disk-backed behavior remain separately benchmarkable?"
> **Domain expert:** "Yes — keep them as separately named **Transcription Pipeline** implementations, not separate public API versions."

> **Dev:** "What should replace `V1Pipeline` and `V2Pipeline`?"
> **Domain expert:** "Use **Full-Memory Pipeline** and **Chunked-File Pipeline**, with `full-memory` as the default configured value."

> **Dev:** "Does the **Full-Memory Pipeline** mean one ASR call over the whole file?"
> **Domain expert:** "No — both pipelines use **ASR Windowing**; full-memory means the PCM source is fully decoded in memory first."

> **Dev:** "Should the merged endpoint read uploads into bytes before calling the configured pipeline?"
> **Domain expert:** "No — wrap the upload as **Audio Input** so the **Full-Memory Pipeline** can read bytes and the **Chunked-File Pipeline** can spool to a path."

> **Dev:** "Should the chunked pipeline delete the temporary upload path when it finishes?"
> **Domain expert:** "No — **Audio Input Cleanup** owns temporary file removal, including after streaming completes."

> **Dev:** "Should route handlers read `request.app.state.runtime` directly?"
> **Domain expert:** "No — use a **Pipeline Dependency** and **Settings Dependency** so API dependencies are explicit and overrideable."

> **Dev:** "Should `get_pipeline()` construct a new pipeline for every request?"
> **Domain expert:** "No — it returns the configured pipeline from the **Singleton Runtime** so loaded models are shared."

> **Dev:** "Should clients choose the pipeline or model per request?"
> **Domain expert:** "No — **Server Startup Selection** chooses the pipeline, backend providers, and model selections before requests are served."

> **Dev:** "Should an unknown pipeline value fall back to `full-memory`?"
> **Domain expert:** "No — **Strict Startup Validation** rejects unknown selector values before the server starts."

> **Dev:** "What ASR model should be used when no ASR model environment variable is set?"
> **Domain expert:** "Use `openai/whisper-medium` as the default **ASR Model Selection**."

> **Dev:** "What backend providers should be used with no environment variables?"
> **Domain expert:** "Use `whisperlivekit` for ASR and `none` for diarization."

> **Dev:** "Should model selections use short names like `whisper-medium` or full names?"
> **Domain expert:** "Use Hugging Face-style full names such as `openai/whisper-medium` and `nvidia/diar_sortformer_4spk-v1`."

> **Dev:** "If whisperlivekit diarization is enabled without a model setting, what should load?"
> **Domain expert:** "Use `nvidia/diar_sortformer_4spk-v1` as the default **Diarization Model Selection**."

> **Dev:** "Should `/health` still return one `backend` field?"
> **Domain expert:** "No — it should expose **Server Startup Selection** and **Capability Readiness**, including optional diarization status."

> **Dev:** "Should route handlers own the 30-second chunk loop and response assembly?"
> **Domain expert:** "No — a **Pipeline Module** orchestrates that work so API routers stay thin."

> **Dev:** "Can we rename response fields while adding Pydantic models?"
> **Domain expert:** "No — Pydantic models should preserve the **Transcription API Contract** for the **Transcription Endpoint**."

> **Dev:** "Should Pydantic models define the multipart transcription request?"
> **Domain expert:** "No — use **Boundary Response Schema** models for successful and error JSON responses, while the route keeps multipart form parsing."

> **Dev:** "Can backend-specific fields pass through the response model as extras?"
> **Domain expert:** "No — use a **Strict Transcription Response Schema** so backend-native data is converted or dropped intentionally."

> **Dev:** "Can v2 remove ignored form fields while adding Pydantic request models?"
> **Domain expert:** "No — v1 and v2 preserve the same **Transcription API Contract** request fields for now."

> **Dev:** "If a client sends OpenAI-style form fields, should the server emulate every OpenAI response format?"
> **Domain expert:** "No — an **OpenAI-Compatible Request** is accepted for client compatibility, but the server returns the **Transcription Response** contract."

> **Dev:** "Should `verbose_json` and `diarized_json` produce different response schemas?"
> **Domain expert:** "No — they are **JSON Response Format Alias** values for the same strict transcription response."

> **Dev:** "If a request sends `model=small`, should it override `ASR_DIAR_MODEL_ASR`?"
> **Domain expert:** "No — `model` is a **Compatibility Model Field** and **Server Startup Selection** remains authoritative."

> **Dev:** "Can transcription endpoints use FastAPI's default `detail` error object?"
> **Domain expert:** "No — transcription failures use an **OpenAI-Style Error** so OpenAI-compatible clients can parse them consistently."

> **Dev:** "Should route handlers directly return JSON error responses for each failure branch?"
> **Domain expert:** "No — raise typed failures and let the **Transcription Exception Handler** produce the **OpenAI-Style Error** response."

> **Dev:** "Can v2 emit custom `transcript.progress` events because disk-backed processing has progress information?"
> **Domain expert:** "No — public streaming uses **OpenAI-Exact SSE**, so progress remains internal unless a future contract adds it."

> **Dev:** "Can a helper that accepts `UploadFile` and raises `HTTPException` live in core because both APIs use it?"
> **Domain expert:** "No — the **Core Boundary** contains only API-agnostic datamodels, interfaces, and pure transformations."

> **Dev:** "Can pipelines pass whisperlivekit `ASRToken` objects around because that is what the prototype uses?"
> **Domain expert:** "No — package boundaries use **Project-Owned Transcript Model** types, and adapters translate backend-native objects."

> **Dev:** "Should ffmpeg conversion live in core because both API versions use it?"
> **Domain expert:** "No — audio decoding and PCM IO belong in the **Audio Module**, while core stays pure."

> **Dev:** "Should `models/asr` own diarization because whisperlivekit exposes it from the same engine today?"
> **Domain expert:** "No — an **ASR Adapter** produces transcript tokens and a **Diarization Adapter** produces speaker timelines, even if one library currently supplies both."

> **Dev:** "Can `models` refer to both Pydantic models and ASR model integrations?"
> **Domain expert:** "No — `backends/` contains **ML Model Integration** modules, while core data shapes live in schemas or types."

> **Dev:** "If whisperlivekit can expose Faster Whisper and Sortformer, is `whisperlivekit` the model or the backend?"
> **Domain expert:** "It is the **Backend Provider**; Faster Whisper is an **ASR Model Selection** and Sortformer is a **Diarization Model Selection**."

> **Dev:** "Should backend modules be split first by ASR versus diarization?"
> **Domain expert:** "No — use a **Provider-First Backend Layout** so one provider can expose both adapter capabilities."

> **Dev:** "Should pipelines call whisperlivekit directly because it owns the selected models?"
> **Domain expert:** "No — a **Backend Adapter Factory** creates **ASR Adapter** and **Diarization Adapter** instances, and pipelines call only those protocols."

> **Dev:** "Should the pipeline lock every ASR call because Faster Whisper may mutate state?"
> **Domain expert:** "No — the **ASR Adapter** owns the **Adapter Concurrency Policy** and locks only when its backend requires it."

> **Dev:** "Should the server fail startup if diarization is not configured?"
> **Domain expert:** "No — an **ASR-Only Server** is valid and returns a transcription response without speaker attribution."

> **Dev:** "Should pipelines force diarization to be batch or streaming?"
> **Domain expert:** "No — the **Diarization Adapter** owns the **Diarization Flow** and returns the same speaker timeline shape either way."

## Flagged Ambiguities

- "server PID" was used to mean both the root process and the full resource footprint — resolved: benchmark metrics target the **Server Process Tree**.
- "memory" was used to mean RSS, VSZ, and actual footprint — resolved: benchmark memory headline is **Process-Tree PSS**, with **Process-Tree USS** as the private-growth signal.
- "IO" was used to mean both file-interface activity and storage-device pressure — resolved: benchmark output separates **Logical IO Rate** from **Physical IO Rate**.
- "benchmark" was used to imply both resource comparison and output-quality validation — resolved: split into **Performance Benchmark** and **Quality Benchmark**, optionally combined in one benchmark run.
- "ground truth" was used for existing output artifacts — resolved: quality scoring uses an explicit **Reference STM** per **Workload Item**.
- "WER" was used as a single headline number — resolved: a **Quality Benchmark** reports the **MeetEval Metric Set** (siWER, cpWER, ORC-WER, DI-cpWER, DER).
- "warmup" was used ambiguously between server lifecycle and benchmark client behavior — resolved: **Server Warmup** runs at startup and gates **Warmup Readiness**, while a **Benchmark Warmup Item** is opt-in client-side and shares the same **Warmup Audio Asset**.
- "memory CSV" was used for the sampled metrics file — resolved: the file is a **Resource CSV**.
- "rate" was used without specifying the denominator — resolved: per-sample rates are **Sample Rate Field** values using observed sample duration.
- "CPU only" and "CPU+GPU" were used as launch intentions — resolved: hardware mode is an **Observed Hardware Profile**.
- "GPU available" was used as a proxy for "GPU used" — resolved: **CPU+GPU Run** requires server-attributed GPU activity.
- "v1" and "v2" were used to mean both public API versions and internal pipeline behavior — resolved: the public API has one **Transcription Endpoint**, while behavior lives in the **Configured Transcription Pipeline**.
- "pipeline dependency" was used ambiguously — resolved: it means a FastAPI **Pipeline Dependency**, not direct route app-state lookup or per-request construction.
- "dependency" was used to imply object construction — resolved: settings can be cached, but loaded adapters and pipelines live in the **Singleton Runtime**.
- "current server" was used to imply all prototype routes — resolved: the **Supported Endpoint Set** excludes `/`, `/asr`, Deepgram-compatible `/v1/listen`, `/v1/models`, `/v2/audio/transcriptions`, and behavior-specific transcription routes.
- "refactor" was used to imply one pipeline implementation — resolved: full-memory and disk-backed behavior remain distinct **Transcription Pipeline** implementations selected by startup configuration.
- "v1 pipeline" and "v2 pipeline" were used for processing strategies — resolved: use **Full-Memory Pipeline** and **Chunked-File Pipeline**.
- "audio file" was used to imply both in-memory bytes and filesystem paths — resolved: pipelines receive **Audio Input** and choose the access mode they require.
- "temp file cleanup" was used to imply endpoint or pipeline ownership — resolved: **Audio Input Cleanup** owns temporary file lifecycle.
- "full-memory" was used to imply whole-file ASR — resolved: both pipeline implementations use **ASR Windowing**.
- "environment variable" was used without namespacing — resolved: **Server Startup Selection** uses `ASR_DIAR_PIPELINE`, `ASR_DIAR_BACKEND_ASR`, `ASR_DIAR_MODEL_ASR`, `ASR_DIAR_BACKEND_DIARIZATION`, and `ASR_DIAR_MODEL_DIARIZATION`.
- "default pipeline" was used to imply fallback behavior — resolved: defaults apply only when unset; invalid values fail **Strict Startup Validation**.
- "backend provider default" was unspecified — resolved: ASR defaults to `whisperlivekit` and diarization defaults to `none`.
- "ASR model default" was unspecified — resolved: the default **ASR Model Selection** is `openai/whisper-medium`.
- "model name" was used to imply short aliases — resolved: **ASR Model Selection** and **Diarization Model Selection** use Hugging Face-style full model identifiers.
- "diarization model default" was unspecified — resolved: enabled whisperlivekit diarization defaults to `nvidia/diar_sortformer_4spk-v1`.
- "health backend" was used to imply one backend status — resolved: `/health` reports **Server Startup Selection** and **Capability Readiness** separately.
- "route code" was used to include transcription orchestration — resolved: orchestration belongs in a **Pipeline Module**.
- "Pydantic models" was used to imply request parsing, response serialization, and response cleanup — resolved: use **Boundary Response Schema** models for successful and error JSON while preserving multipart form parsing and the existing **Transcription API Contract**.
- "response schema" was used to imply extensibility for backend-native fields — resolved: use a **Strict Transcription Response Schema**.
- "OpenAI compatible" was used to imply full response-format emulation — resolved: the package accepts **OpenAI-Compatible Request** parameters and returns a **Transcription Response**.
- "response_format" was used to imply multiple output contracts — resolved: supported JSON-like values are **JSON Response Format Alias** values.
- "model" in the request was used to imply runtime model switching — resolved: it is a **Compatibility Model Field** and does not override startup-selected models.
- "HTTP error" was used to imply FastAPI defaults — resolved: transcription endpoints return **OpenAI-Style Error** responses.
- "JSONResponse errors" was used to imply inline route branches — resolved: typed failures are converted by a **Transcription Exception Handler**.
- "OpenAI streaming" was used while considering custom progress events — resolved: public streaming uses **OpenAI-Exact SSE** with no package-specific progress events.
- "core" was used as a general shared-code bucket — resolved: the **Core Boundary** excludes FastAPI-specific helpers and HTTP concerns.
- "token" and "segment" were used to imply whisperlivekit classes — resolved: package boundaries use **Project-Owned Transcript Model** types.
- "API utils" was used to imply audio conversion helpers — resolved: ffmpeg and PCM IO belong in the **Audio Module**.
- "ASR model integration" was used to include diarization by implication — resolved: **ASR Adapter** and **Diarization Adapter** are separate capabilities.
- "model" was used to mean both Pydantic data shapes and ML backends — resolved: **ML Model Integration** lives under `backends/`, while core uses schemas and types.
- "backend" was used to mean provider, capability, and model family — resolved: **Backend Provider** selects the integration, while **ASR Model Selection** and **Diarization Model Selection** select models within providers.
- "backend layout" was used to imply capability-first directories — resolved: use a **Provider-First Backend Layout**.
- "backend interface" was used to imply pipelines might call providers directly — resolved: pipelines call **ASR Adapter** and **Diarization Adapter** protocols created by a **Backend Adapter Factory**.
- "ASR lock" was used to imply a pipeline-level concern — resolved: concurrency is an **Adapter Concurrency Policy**.
- "diarization backend" was used to imply a required server dependency — resolved: diarization is optional, and an **ASR-Only Server** is valid.
- "diarization chunks" was used to imply pipeline-owned diarization behavior — resolved: **Diarization Flow** belongs to the **Diarization Adapter**.
