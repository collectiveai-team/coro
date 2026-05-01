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

**Resource Benchmark**:
A benchmark run that compares resource usage and timing without gating results on transcription equivalence.
_Avoid_: Accuracy benchmark, quality benchmark

**Quality Reference**:
An explicit transcript or diarization artifact used to score benchmark output quality.
_Avoid_: Implicit ground truth

**Reference RTTM**:
The canonical diarization quality reference format for DER scoring.
_Avoid_: Implicit WhisperX ground truth

**Hypothesis Diarization**:
The diarization timeline parsed from the server's WhisperX JSON or SSE response for DER scoring.
_Avoid_: Hypothesis RTTM requirement

**DER Policy**:
The diarization scoring settings used for a benchmark run: 0.25 second collar and overlapped speech included by default.
_Avoid_: Unspecified DER settings

**WER Normalization**:
The Spanish-friendly text normalization used before WER scoring: lowercase, Unicode normalization, punctuation removal, and whitespace collapse.
_Avoid_: Raw WER when normalized WER is intended

**Quality Metric**:
A benchmark result such as WER or DER computed only when the corresponding quality reference is provided.
_Avoid_: Resource metric

**Backfilled Quality Column**:
A resource CSV column whose aggregate quality value is repeated on every sampled row after the repetition is scored.
_Avoid_: Final-row-only quality field

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

**API Version**:
The paired package namespace and HTTP route prefix for a server behavior contract.
_Avoid_: Internal-only version, route-only version

**Supported Endpoint Set**:
The intentionally exposed server endpoints for the packaged ASR diarization service.
_Avoid_: Every route from the prototype scripts

**Transcription Pipeline**:
The end-to-end processing path that turns an uploaded audio file into transcript, diarization, and raw-word response data.
_Avoid_: Endpoint handler, model call

**Pipeline Module**:
The package area that orchestrates audio IO, ASR adapters, diarization adapters, and core response transformations for a transcription pipeline.
_Avoid_: API router, core model, ASR adapter

**Transcription API Contract**:
The versioned form-request and JSON/SSE-response shape used by transcription endpoints.
_Avoid_: Internal result dict, cleanup opportunity

**OpenAI-Compatible Request**:
A transcription form request that accepts OpenAI-style parameters for client compatibility without requiring every OpenAI response format.
_Avoid_: Full OpenAI API clone

**OpenAI-Style Error**:
A transcription endpoint error response shaped as an OpenAI-style `error` object rather than FastAPI's default `detail` object.
_Avoid_: FastAPI default error

**WhisperX-Style Response**:
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

**ML Model Integration**:
A package module under `backends/` that adapts an external ASR or diarization backend.
_Avoid_: Pydantic model, response schema

**Diarization Adapter**:
A model integration that produces speaker timeline segments from audio while hiding backend-specific diarization APIs.
_Avoid_: ASR backend, speaker helper

## Relationships

- A **Benchmark Run** measures one **Server Process Tree** per implementation under test.
- A **Server Process Tree** is sampled as a **Dynamic Process Tree** during a request.
- A **Resource Benchmark** may compare implementations even when their transcription output differs.
- A **Resource Benchmark** writes one **Resource CSV** per repetition.
- A **Resource CSV** uses a **Stable Resource Schema** across hardware profiles.
- A **Resource CSV** may also carry aggregate **Quality Metric** values for the repetition.
- A **Quality Metric** stored in a **Resource CSV** is a **Backfilled Quality Column**.
- A **Quality Metric** is computed only from an explicit **Quality Reference**.
- DER compares a **Reference RTTM** against **Hypothesis Diarization** parsed by the benchmark.
- DER results are interpreted only together with the **DER Policy**.
- WER results are interpreted only together with the **WER Normalization**.
- A **Resource CSV** contains both cumulative counters and **Sample Rate Field** values.
- An **Observed Hardware Profile** is inferred from measurements, not from how the server was launched.
- A **CPU+GPU Run** requires GPU activity attributable to the **Server Process Tree**.
- A **CPU-Only Run** can occur even when a GPU is visible but unused by the **Server Process Tree**.
- **Process-Tree PSS** is the headline memory comparison for a **Benchmark Run**.
- **Process-Tree USS** is the private-growth companion to **Process-Tree PSS**.
- **Logical IO Rate** describes pipeline work during a **Benchmark Run**.
- **Physical IO Rate** describes storage pressure during a **Benchmark Run**.
- **Process-Tree CPU Rate** describes compute pressure during a **Benchmark Run**.
- **Transcription Throughput** is the headline timing comparison for a **Resource Benchmark**.
- A **Workload Set** for deciding disk-backed chunking value includes short, medium, and long audio inputs.
- An **API Version** is reflected both in the Python package namespace and in the public HTTP route prefix.
- The **Supported Endpoint Set** contains `/health`, `/v1/audio/transcriptions`, and `/v2/audio/transcriptions` only.
- Each **API Version** owns a distinct **Transcription Pipeline** while sharing core contracts and response transformations.
- A **Pipeline Module** owns orchestration for one or more **Transcription Pipeline** implementations.
- A **Transcription API Contract** is preserved within an **API Version** unless a new version intentionally changes it.
- An **OpenAI-Compatible Request** returns a **WhisperX-Style Response** in the current package contract.
- Transcription endpoints return **OpenAI-Style Error** responses for request and processing failures.
- Streaming transcription uses **OpenAI-Exact SSE** rather than package-specific progress events.
- The **Core Boundary** excludes FastAPI request parsing, response classes, route dependencies, and HTTP errors.
- A **Project-Owned Transcript Model** crosses package boundaries; backend-native types are converted at adapter edges.
- The **Audio Module** owns ffmpeg and PCM IO concerns outside the **Core Boundary**.
- An **ASR Adapter** and a **Diarization Adapter** are separate capabilities that orchestration combines into an API response.
- A **ML Model Integration** is distinct from core schemas and internal data types.

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
> **Domain expert:** "Not for this **Resource Benchmark** — correctness can be checked separately, but this run is about resource usage and timing."

> **Dev:** "Can we compute WER or DER from whatever looks like ground truth in the repo?"
> **Domain expert:** "No — compute a **Quality Metric** only when an explicit **Quality Reference** is provided."

> **Dev:** "Does the server need to return RTTM for DER?"
> **Domain expert:** "No — the benchmark parses **Hypothesis Diarization** from the WhisperX response and compares it to a **Reference RTTM**."

> **Dev:** "Can we report DER without the collar and overlap settings?"
> **Domain expert:** "No — DER must be reported with the **DER Policy** because those settings change the score."

> **Dev:** "Should punctuation and casing differences affect WER?"
> **Domain expert:** "No — use **WER Normalization** so WER reflects word differences rather than formatting differences."

> **Dev:** "Should the sampled metrics file still be called `memory_N.csv`?"
> **Domain expert:** "No — use a **Resource CSV** because the file contains memory, IO, CPU, and GPU observations."

> **Dev:** "Should CPU-only and GPU runs have different CSV columns?"
> **Domain expert:** "No — use a **Stable Resource Schema** so notebooks can compare runs without schema branching."

> **Dev:** "Where should WER and DER be stored?"
> **Domain expert:** "Store them in the **Resource CSV** for the repetition, even though they are aggregate **Quality Metric** values."

> **Dev:** "Should WER and DER appear only on the last CSV row?"
> **Domain expert:** "No — use a **Backfilled Quality Column** so every row for the repetition carries the aggregate value."

> **Dev:** "Should we write only IO rates?"
> **Domain expert:** "No — write cumulative counters and **Sample Rate Field** values so rates can be audited or recomputed."

> **Dev:** "Should we label a run CPU-only because we hid CUDA at launch?"
> **Domain expert:** "No — use the **Observed Hardware Profile** inferred from measured runtime behavior."

> **Dev:** "If `nvidia-smi` shows an idle GPU, is that a **CPU+GPU Run**?"
> **Domain expert:** "No — it is only a **CPU+GPU Run** when GPU activity is attributable to the **Server Process Tree**."

> **Dev:** "Can the experimental chunked server live under `api.v2` while still exposing `/v1/audio/transcriptions`?"
> **Domain expert:** "No — an **API Version** must align its package namespace and HTTP route prefix, so `api.v2` serves `/v2/...`."

> **Dev:** "Should the packaged server keep `/`, `/asr`, `/v1/listen`, or `/v1/models` because the prototype has them?"
> **Domain expert:** "No — the **Supported Endpoint Set** is limited to `/health`, `/v1/audio/transcriptions`, and `/v2/audio/transcriptions`."

> **Dev:** "Should v1 and v2 immediately call one unified transcription service?"
> **Domain expert:** "No — each **API Version** keeps its own **Transcription Pipeline** so full-memory and disk-backed behavior remain separately benchmarkable."

> **Dev:** "Should route handlers own the 30-second chunk loop and response assembly?"
> **Domain expert:** "No — a **Pipeline Module** orchestrates that work so API routers stay thin."

> **Dev:** "Can we rename response fields while adding Pydantic models?"
> **Domain expert:** "No — Pydantic models should preserve the **Transcription API Contract** for each **API Version**."

> **Dev:** "Can v2 remove ignored form fields while adding Pydantic request models?"
> **Domain expert:** "No — v1 and v2 preserve the same **Transcription API Contract** request fields for now."

> **Dev:** "If a client sends OpenAI-style form fields, should the server emulate every OpenAI response format?"
> **Domain expert:** "No — an **OpenAI-Compatible Request** is accepted for client compatibility, but the server returns the **WhisperX-Style Response** contract."

> **Dev:** "Can transcription endpoints use FastAPI's default `detail` error object?"
> **Domain expert:** "No — transcription failures use an **OpenAI-Style Error** so OpenAI-compatible clients can parse them consistently."

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

## Flagged Ambiguities

- "server PID" was used to mean both the root process and the full resource footprint — resolved: benchmark metrics target the **Server Process Tree**.
- "memory" was used to mean RSS, VSZ, and actual footprint — resolved: benchmark memory headline is **Process-Tree PSS**, with **Process-Tree USS** as the private-growth signal.
- "IO" was used to mean both file-interface activity and storage-device pressure — resolved: benchmark output separates **Logical IO Rate** from **Physical IO Rate**.
- "benchmark" was used to imply both resource comparison and output-quality validation — resolved: the current scope is a **Resource Benchmark**.
- "ground truth" was used for existing output artifacts — resolved: quality scoring uses an explicit **Quality Reference**.
- "memory CSV" was used for the sampled metrics file — resolved: the file is a **Resource CSV**.
- "rate" was used without specifying the denominator — resolved: per-sample rates are **Sample Rate Field** values using observed sample duration.
- "CPU only" and "CPU+GPU" were used as launch intentions — resolved: hardware mode is an **Observed Hardware Profile**.
- "GPU available" was used as a proxy for "GPU used" — resolved: **CPU+GPU Run** requires server-attributed GPU activity.
- "v2" was used as an internal module label while the code still exposed `/v1` routes — resolved: an **API Version** names both the module namespace and HTTP route prefix.
- "current server" was used to imply all prototype routes — resolved: the **Supported Endpoint Set** excludes `/`, `/asr`, Deepgram-compatible `/v1/listen`, and `/v1/models`.
- "refactor" was used to imply pipeline unification — resolved: each **API Version** keeps a distinct **Transcription Pipeline** for now.
- "route code" was used to include transcription orchestration — resolved: orchestration belongs in a **Pipeline Module**.
- "Pydantic models" was used to imply request or response cleanup — resolved: models preserve the existing **Transcription API Contract**.
- "OpenAI compatible" was used to imply full response-format emulation — resolved: the package accepts **OpenAI-Compatible Request** parameters and returns a **WhisperX-Style Response**.
- "HTTP error" was used to imply FastAPI defaults — resolved: transcription endpoints return **OpenAI-Style Error** responses.
- "OpenAI streaming" was used while considering custom progress events — resolved: public streaming uses **OpenAI-Exact SSE** with no package-specific progress events.
- "core" was used as a general shared-code bucket — resolved: the **Core Boundary** excludes FastAPI-specific helpers and HTTP concerns.
- "token" and "segment" were used to imply whisperlivekit classes — resolved: package boundaries use **Project-Owned Transcript Model** types.
- "API utils" was used to imply audio conversion helpers — resolved: ffmpeg and PCM IO belong in the **Audio Module**.
- "ASR model integration" was used to include diarization by implication — resolved: **ASR Adapter** and **Diarization Adapter** are separate capabilities.
- "model" was used to mean both Pydantic data shapes and ML backends — resolved: **ML Model Integration** lives under `backends/`, while core uses schemas and types.
