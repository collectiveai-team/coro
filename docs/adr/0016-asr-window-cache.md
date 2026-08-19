# ASR Window Cache

Coro caches ASR results per **ASR Windowing** window on local disk, so re-running audio that has already been transcribed skips the model entirely. A cache hit costs a decode and a hash; a miss costs exactly what it cost before. The cache is **disabled by default** (`CORO_ASR_CACHE=enabled` turns it on), because it introduces disk growth to a service that currently has none.

The motivating cost is measured, not assumed. Thirty minutes of audio through the default `onnx-asr` / `nemo-parakeet-tdt-0.6b-v3` configuration on CPU: **392.8 s cold, 0.31 s fully cached**. The floor on a cached re-run is ffmpeg decode, which is paid every time and is therefore what limits the win — it came to **0.38 s, or 0.1% of the cold run**. That number was the one open question about whether this was worth building, since on a GPU inference is fast enough that decode could plausibly have dominated; even at a 40× faster inference path decode would still be only a few percent, so the answer holds on both devices.

## One seam

The cache is a decorator satisfying the **ASR Adapter** protocol, constructed by the **ASR Backend Adapter Factory**. That is its only integration point. Everything a key needs — the window PCM, the language and the prompt — is already in `transcribe_pcm`'s signature, so the decorator requires no change to ASR Windowing, to either pipeline, or to any route, and it therefore covers the **Full-Memory Pipeline**, the **Streaming Pipeline** and the live socket automatically, because all three reach the model through that one call. A cache wired into the pipelines instead would have needed three implementations and could drift out of sync with any of them.

**Server Warmup** deliberately bypasses it, running against the unwrapped adapter. Warmup exists to prove the model loads and runs; served from cache, every start after the first would report **Warmup Readiness** without having loaded anything, which would make the health contract a lie.

## What is in the key, and what is deliberately not

A window's key is the digest of its **canonical PCM** — post-decode, post-resample, 16 kHz mono s16le — plus a normalised language, plus the prompt for backends that honour one, plus an **ASR fingerprint**.

Keying on canonical PCM rather than on the encoded upload is what makes container, encoding, declared sample rate, channel count and filename automatically irrelevant: the same audio re-encoded for convenience decodes to the same bytes and hits the same entry, and the live socket's client-declared `sample_rate` stops being a key input because its effect is already baked into the resampled bytes.

The fingerprint covers everything *outside* the request that can change a prediction: ASR Backend Provider, **ASR Model Selection**, the provider-specific knobs that provider actually honours (quantization, compute type, VAD and its threshold), the *resolved* device, the inference runtime version, the accelerator identity, the ASR Windowing geometry, the prompt capability, and an explicit cache format version.

`auto` is resolved rather than recorded, because `auto` is not a device: it means CPU on one host and CUDA on another, and those two do not agree. Measured, they produce identical text and identical timestamps but token probabilities differing by up to 4e-4 — and those probabilities are part of the published response, which is required to be byte-identical between the materialised response and the streamed done frame. Rounding them to hide the divergence was evaluated and rejected: it still diverges at two decimal places, and it would change a published contract to work around a caching concern.

Only four request-level inputs can change what the ASR Adapter returns: the audio, the language, the prompt (on the OpenAI-compatible surface only), and the live socket's sample rate. Everything else across both vendor surfaces is response formatting or is ignored outright, and is excluded on purpose. `response_format`, `stream`, `temperature`, `timestamp_granularities`, the Deepgram-compatible `diarize`, `utterances`, `punctuate` and `smart_format` — including them would destroy hit rate for no correctness gain. Notably the Deepgram `diarize` flag only decides whether speaker fields are rendered, and `stream` provably takes the identical ASR path. The concurrency knobs are excluded for the same reason: ASR was measured deterministic under concurrent, interleaved and cross-process execution, so they cannot change output either.

There is **no tenant dimension**, because the service has no authentication, API key or caller identity of any kind. A content-addressed cache is therefore shared across all callers of one deployment. That consequence was reviewed and accepted for the deployment model in use; it would have to be revisited the day an auth scheme lands.

There is deliberately **no per-request cache control**. Neither vendor API has such a parameter, and adding one would break the fidelity policy those surfaces are held to (ADR 0015).

## Prompt handling

The prompt carried between windows is derived from previous windows' output, so whether it belongs in the key depends on the backend, and each adapter declares the capability explicitly.

`faster-whisper` honours it — `initial_prompt` is a native Whisper capability — so its keys chain: a window's key depends on the previous windows' tokens. That is still self-healing, because recomputing a missing window yields identical tokens and therefore leaves the following window's key unchanged.

`onnx-asr` ignores it *architecturally*: a transducer has no text input port, so there is nowhere for a prompt to go. `onnx-genai` ignores it *implementationally*, as a gap in the GenAI streaming API. Both get independent per-window keys, which is the best hit rate available and makes a missing window purely local. The distinction between the two reasons matters because the second could change, which is why the capability is part of the fingerprint: were `onnx-genai` to start honouring prompts, existing entries would invalidate rather than be served under assumptions that no longer hold.

## Storage and retention

One shared SQLite database in WAL mode, separate from the per-request **Transcript Spill Store** and its delete-on-close lifecycle. Rows hold the key, the resulting tokens, and creation and access timestamps. **No audio and no decoded PCM is stored** — explicitly rejected: inputs are large, the caller re-supplies them anyway, and storing them would reopen decisions this project has already made about not staging audio on disk. Storage is on the order of a few megabytes per hour of audio.

Each window is committed as it completes rather than in batches, so a process killed part-way loses at most the window in flight and the next attempt resumes where it stopped. Resume is a *consequence* of that choice, not a feature: no lease, heartbeat or abandoned-entry reclamation is in scope, and the out-of-memory failure that originally motivated this line of work was second-hand and never reproduced. Writes happen on worker threads, because a shared store with a busy timeout can block, unlike the uncontended per-request spill store, and blocking the event loop would make enabling the cache degrade concurrency.

Retention has two bounds and needs both. A **maximum size** with least-recently-used eviction, swept on write, is the real protection against unbounded growth; sweeping on write rather than from a background task means a frequently-restarted process still enforces it. A **time-to-live** applied lazily on lookup stops stale entries outliving the runs that produced them. Access time is refreshed on read, so an entry in active use is never evicted from underneath the run reading it. Eviction can never corrupt a result, because every entry is recomputable and none is load-bearing: a full disk degrades performance, not correctness.

The cache directory rejects RAM-backed filesystems and is validated during **Strict Startup Validation**, so a misconfiguration fails when the server starts rather than on the first request. It shares the probe with the transcript spill directory, which refuses one for the same reason. Unlike the spill directory it defaults under the user cache root rather than the system temp dir, because a persistent cache that the OS reclaims on reboot would miss on exactly the re-runs it exists to serve. Entries are never migrated: a fingerprint mismatch is simply a miss, which costs one recomputation rather than a wrong answer.

## What this rests on

Correctness rests on ASR being a pure function of its inputs, which was measured rather than assumed: both the default ONNX backend and `faster-whisper` produced byte-identical tokens across repeated calls, interleaved neighbours, concurrent execution, three separate processes, and GPU contention, on both CPU and CUDA. That property is guarded by an opt-in real-model test rather than left as an assumption. The acceptance test asserts the consequence that actually matters — a cold run, a fully warm run, and a run over a cache with deliberate holes all produce byte-identical responses — because a cached transcript is only worth having if it can be trusted as much as a fresh one.

## Out of scope

**Diarization is not cached.** The default configuration has diarization disabled, and the streaming diarizer's per-chunk state is not reconstructible from its outputs, so per-chunk caching would save nothing. Caching a completed speaker timeline is a reasonable later addition, but it needs a second seam and only helps identical end-to-end re-runs.

Also excluded: caching audio or PCM, a shared or distributed cache, multi-tenant isolation, per-request cache control, and any change to the HTTP contract.
