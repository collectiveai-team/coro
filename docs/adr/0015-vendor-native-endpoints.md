# Vendor-Native Endpoints, and the Fidelity Policy They Are Held To

`coro` assigns a speaker to every word and carries a real per-word confidence
(ADR 0008), and discarded both at the API boundary. Per-word speakers are now
reachable — through **the vendor's own endpoint, implementing the vendor's own
contract**, not through a new value on the OpenAI endpoint.

`POST /v1/listen` is added, serving Deepgram's pre-recorded contract.

## This departs from the recorded decision, deliberately

Issue `46` recorded exposure option **X1**: per-word speakers reached through
additional opt-in `response_format` values on `/v1/audio/transcriptions`,
**AssemblyAI first, Deepgram second**. Issue `12` then cited X1 as the discharged
gate for its own segmentation decision. This ADR does **not** implement X1, and
says so here rather than letting `main` and the issue record disagree.

Two departures, both argued below:

1. **Vendor-native endpoints instead of new `response_format` values.** X1's
   stated goal was to leave the OpenAI-compatible surface unspent. New endpoints
   achieve that goal *more* completely than X1 does, because they leave the
   OpenAI request parameter alone as well as the response bodies — see the next
   section. X1 was implemented first on this branch and then withdrawn.
2. **AssemblyAI is not implemented, so the ordering is inverted.** Its contract
   is asynchronous and needs a job-state subsystem coro does not have; a
   synchronous single-POST imitation would be the partial clone rule 1 of the
   fidelity policy forbids. Deferred to its own issue, with the reasoning under
   *Consequences*.

**What does not change is the premise issue `12` depends on.** Its decision rule
is "if per-word speaker truth leaves the process → sentence-first + majority". It
leaves the process: on `POST /v1/listen` and `WebSocket /v1/listen`, with
`word_segments` carrying the per-word truth beside the segment summary. The door
is different; the gate is still discharged, so issue `12`'s decision stands
unaltered and is not reopened by this ADR.

## Each provider owns its own contract, including the endpoint

The OpenAI-compatible surface is defined by OpenAI. `response_format` is
OpenAI's parameter and its value set is OpenAI's to define. Adding
`deepgram_json` to that enum extends a format that a third party owns — which
is the same error as adding a per-word `speaker` to `TranscriptionDiarized`,
just relocated from the response body to the request parameter.

The rejected alternative was exactly that: vendor-shaped bodies selected by new
`response_format` values on `/v1/audio/transcriptions`. It was implemented
first and then withdrawn. It protected `diarized_json`'s bytes while polluting
the parameter that selects it, so the OpenAI surface was not actually left
alone. It also has no answer for the parts of a vendor contract that are not
the response body — Deepgram takes a raw audio body rather than multipart,
gates speakers behind `diarize=true`, and reports errors as
`err_code`/`err_msg` rather than as an OpenAI `error` object. A format-only
adaptation gets all three wrong while claiming the vendor's name.

So: **one endpoint per provider, each implementing that provider's documented
contract; no provider's format is ever extended to carry another's data.**

| provider | endpoint | request | errors |
|---|---|---|---|
| OpenAI | `POST /v1/audio/transcriptions` | multipart form | `{"error": {...}}` |
| Deepgram | `POST /v1/listen` | raw audio body + query params | `{"err_code","err_msg","request_id"}` |

`/v1/audio/transcriptions` is byte-unchanged, asserted by
`tests/test_openai_formats_unchanged.py`. `diarized_json` therefore stays a
byte-exact clone of OpenAI's `TranscriptionDiarized` permanently.

## This supersedes the Supported Endpoint Set in ADR 0001

ADR 0001 states: *"The supported endpoint set is `/health` and
`/v1/audio/transcriptions`… Prototype-only UI, WebSocket, Deepgram-compatible,
model-list, behavior-specific transcription routes… do not shape the packaged
API."*

That exclusion was written against a **prototype** `/v1/listen` inherited from
the pre-package server, whose purpose was to avoid carrying unowned, untested
routes into the packaged API. It is superseded only for a deliberately
implemented, tested, SDK-conformant vendor endpoint. The rest of ADR 0001
stands: `/`, `/asr`, `/v1/models`, `/v2/audio/transcriptions`, WebSocket and
behavior-specific transcription routes remain excluded, and
`tests/test_supported_endpoint_set.py` still enforces that.

The Supported Endpoint Set becomes: `/health`, `/v1/audio/transcriptions`,
`/v1/listen`.

## Fidelity policy

A partial clone carrying a vendor's name is a liability, because clients will
point that vendor's SDK at it. The standard:

1. **A vendor endpoint is a documented subset, not a full clone.** Responses
   are structurally valid against the vendor's published schema and are
   validated against the vendor's own SDK types in
   `tests/test_deepgram_sdk_conformance.py` — not against a local copy.
2. **Every field emitted is measured or structurally required.** A response the
   SDK cannot parse defeats the purpose, so SDK-required fields are populated;
   each is listed below with its source.
3. **Optional and unmeasured means omitted, never stubbed.** No placeholder
   values. This is what decides `speaker_confidence`.
4. **Vendor features `coro` does not compute are absent, not empty.**
   Summarization, sentiment, entities, topics, intents, paragraphs and search
   are omitted rather than emitted as empty containers, which would imply the
   feature ran and found nothing.
5. **Unhonoured parameters are ignored and documented; two are refused.**

   Almost everything `coro` cannot honour is accepted and ignored, and its
   effect is stated in the OpenAPI schema for that parameter. This keeps the
   endpoint drop-in: a client's standard parameter bundle still works, and a
   flag Deepgram adds next year does not break the endpoint. It matches how
   the OpenAI endpoint already treats `model` and `temperature`. When a feature
   is not computed, its response key is simply absent — the client reads
   nothing there and can handle it.

   Two cannot be ignored safely, because ignoring them fails *silently and
   harmfully* rather than merely omitting data:

   | parameter | why refusal, not silence |
   |---|---|
   | `redact` | returning unredacted text under a redaction request is a compliance failure wearing a 200. The client has no way to detect it. |
   | `callback`, `callback_method` | the client is built to receive `{request_id}` and then wait for a webhook. Ignoring it hangs that workflow forever rather than handing back data it can inspect. |

   The dividing line is whether the client can *observe* the difference. A
   missing `summary` key is visible; unredacted text that was supposed to be
   redacted is not, and a webhook that never arrives is indistinguishable from
   one that is merely slow.

   An earlier draft refused sixteen parameters, including every analysis
   feature. That was over-strict: it broke otherwise-serviceable requests for
   features the client may never have read, and traded real compatibility for
   a purity that only the two rows above actually need.

   Refusal triggers on any value that is not an explicit disable, because these
   carry a payload rather than a boolean (`callback` a URL, `redact` a policy
   name). `redact=false` asks for nothing and is not refused.

### Coverage: what of Deepgram's contract is implemented

This is a documented subset, and "documented" means the boundary is written
down rather than discovered by a client at runtime.

Deepgram's `listen` API has **two transports**, not two API versions of one
transport:

| surface | transport | status |
|---|---|---|
| `POST /v1/listen`, raw audio body | REST | **implemented** |
| `POST /v1/listen`, `application/json` `{"url": ...}` ingest | REST | **refused** with an explicit 400 |
| `wss://…/v1/listen` | WebSocket | **implemented** (`linear16`; other encodings refused) |
| `listen/v2` | WebSocket **only** | **not implemented** |
| 37 pre-recorded parameters | REST | 3 honoured, 3 refused, 31 accepted and ignored |

`listen/v2` is not a newer pre-recorded API. It exposes only `connect` —
*"real-time conversational speech recognition with contextual turn detection"*
— with 13 parameters that are all streaming concerns (`eot_threshold`,
`eot_timeout_ms`, `eager_eot_threshold`, `encoding`, `sample_rate`,
`language_hint`). There is no `v2` REST surface at all: the SDK has
`listen/v1/media/` and no `listen/v2/media/`. So v2 is not a separate gap; it
is part of the streaming gap below, and nothing about it applies to a
pre-recorded endpoint.

**Honoured (3):** `diarize`, `utterances`, and `language` as a hint.

**Refused (3)** — rule 5: `redact`, `callback`, `callback_method`.

**Accepted and ignored (31):** everything else, each documented in the OpenAPI
schema with what its absence means — the analysis features (`summarize`,
`sentiment`, `topics`, `intents`, `detect_entities`, `paragraphs`, `search`,
`measurements`, `replace`), the formatting knobs (`punctuate`, `smart_format`,
`numerals`, `profanity_filter`, `filler_words`, `dictation`), `multichannel`,
`detect_language`, `model`, `version`, and any parameter Deepgram adds later.

On the REST endpoint `encoding` is among the ignored: it describes headerless
PCM, and `coro` decodes with ffmpeg, which sniffs the container, so a
headerless upload fails with the ordinary undecodable-audio 400. On the
WebSocket there *is* no container to sniff, so `encoding` and `sample_rate`
are honoured there — see below.
That is a real gap for raw-PCM clients, left open rather than papered over.

### Streaming: a WebSocket, and it is genuinely live

Deepgram's streaming contract is a different transport, not a different
encoding of the same one: a WebSocket carrying `Results` frames as audio is
transcribed and a closing `Metadata` frame, with control in-band as JSON text
frames (`KeepAlive`, `Finalize`, `CloseStream`). `coro`'s existing SSE stream
is OpenAI's framing and a Deepgram client cannot consume it, so
`WebSocket /v1/listen` is implemented separately.

**It is live, not buffer-then-transcribe.** That distinction is the whole
value, so it is asserted directly: a test streams audio and reads a `Results`
frame *before* sending `CloseStream`, which an implementation that waits for
end-of-stream cannot pass.

This was almost free architecturally. `ASRWindowing.stream_chunks` already
accepts any async iterator of PCM, and a StreamingDiarizer already ingests
chunk by chunk; the Streaming Pipeline reads from a spooled file only because
an HTTP upload arrives whole. `coro/pipelines/live.py` supplies the other
source — a bounded queue fed by socket frames — and everything below it is the
same code, so a socket stream and an upload cannot drift apart in behaviour.
The queue is bounded so a client sending audio faster than the ASR consumes it
gets backpressure rather than unbounded memory growth.

**Audio format is declared, not sniffed.** A live socket has no container, so
the client's `encoding` and `sample_rate` are authoritative and are validated
at connection time — a misconfigured client fails immediately with an `Error`
frame instead of streaming a minute of audio that decodes to noise. Only
`linear16` is accepted; other rates are resampled to the canonical 16 kHz with
a stateful resampler, since restarting one per chunk would put a discontinuity
at every frame boundary.

**Interim frames carry no speaker.** The diarization timeline is still being
built while audio arrives, so a per-word label mid-stream would be a guess that
a later frame silently contradicts. With `diarize=true` a final attributed
frame is emitted once the timeline is complete. This is a deliberate deviation
from Deepgram, which labels interim words; `coro` prefers absence to a label it
would have to retract.

`listen/v2` remains unimplemented. It is WebSocket-only and its distinguishing
feature is contextual turn detection, which `coro` has no equivalent of: the
Streaming Pipeline emits transcript deltas, not turn-boundary events. Deepgram
v1's `vad_events`, `interim_results` and `utterance_end_ms` are likewise not
implemented — `coro` surfaces only tokens it has already accepted, so every
frame it emits is final and there are no interim frames to suppress.

### This amends ADR 0001 a second time

ADR 0001 excludes WebSocket routes from the packaged API, on the same grounds
as the prototype `/v1/listen`: unowned, untested routes inherited from the
pre-package server. A tested vendor-conformant streaming endpoint is not that,
and streaming is a product requirement. The exclusion continues to hold for
every other WebSocket route.

ADR 0001 excludes WebSocket routes, so implementing it would require amending
that exclusion a second time, on its own evidence. Recorded as a known gap.

### Required fields and their sources

| field | required by SDK | source |
|---|---|---|
| `metadata.request_id` | yes | server request id, also in the logs |
| `metadata.sha256` | yes | SHA-256 of the submitted bytes |
| `metadata.created` | yes | ISO 8601 completion time |
| `metadata.duration` | yes | measured audio duration |
| `metadata.channels` | yes | `1`; audio is converted to mono |
| `metadata.models` / `model_info` | yes | configured ASR Model Selection and Backend Provider |
| `metadata.transaction_key` | no | omitted — a billing concept with no `coro` equivalent |

## Vendor defaults are honoured, not overridden

`diarize` and `utterances` both default to **false**, as they do at Deepgram.
Per-word speakers require `?diarize=true`, and the speaker-turn view requires
`?utterances=true`. Defaulting them to true would be more useful and less
faithful; a client written against Deepgram's documentation would get a
different response shape from the same request.

This also makes the payload opt-in at the vendor's own control point rather
than at one `coro` invents.

## `speaker_confidence` is omitted

Deepgram's `words[].speaker_confidence` is the diarizer's posterior for the
assigned speaker. `coro`'s diarization adapters binarize their per-frame
probabilities into a speaker timeline before it reaches the Core Boundary, so
that quantity does not exist downstream.

Deriving a substitute was considered and rejected. The available quantity is
the winning speaker's share of timeline overlap inside a word's span, which
`attribute_span` already computes. That measures *timeline dominance*, not
diarizer certainty: a word falling entirely inside one confidently-wrong
timeline entry scores 1.0. Publishing it under Deepgram's name would invite
clients to read it as a posterior, which is what rule 3 exists to prevent.

Revisit when per-frame posteriors are surfaced past binarization. The `overlap`
flag from ADR 0008 already covers the narrower case of concurrent speakers.

## Speaker labels: absent, not invented; numbered, not renumbered

**The `-1` sentinel becomes an absent key.** ADR 0008 uses `-1` for a word the
diarization timeline does not support. Deepgram has no sentinel and never emits
a null speaker, so the key is omitted for those words (`exclude_none` at
serialization). Emitting `-1` would present it as a speaker *named* `-1`;
emitting `null` would invent a convention Deepgram does not use. Absence is the
honest and native encoding, and it is what an undiarized response looks like
anyway.

**Numbering is passed through.** `coro` labels speakers from 1; Deepgram's own
output is 0-based. Renumbering would break correspondence with `diarized_json`
for the same audio, and speaker labels are passed verbatim into the Hypothesis
STM used for cpWER, DI-cpWER and DER scoring (ADR 0008), so translating here
would put the wire format and the scored format out of step. Deepgram types the
field as `Optional[int]`, which accepts the pass-through.

## Utterances are derived from words, not from segments

An utterance is a maximal run of consecutive `word_segments` entries sharing a
speaker — not a one-to-one mapping of `segments[]`.

Under ADR 0008 the two coincide, because segments are already split at every
word-level speaker change. That coincidence is not durable: issue 12 is
deciding whether segments become sentence-shaped with a duration-weighted
majority speaker, which would make a segment a *summary* over several speakers
and therefore an invalid utterance. Deriving turns from the per-word truth
keeps this correct under either outcome, and means the per-word view and the
turn view cannot disagree, because only one of them is a source.

## Payload size, measured

Measured on a synthetic 600 s workload — 1405 words, 157 segments, 3 speakers:

| request | bytes | vs `diarized_json` |
|---|---|---|
| `/v1/audio/transcriptions` `json` | 9,220 | 0.27× |
| `/v1/audio/transcriptions` `diarized_json` | 34,203 | 1.00× |
| `/v1/audio/transcriptions` `verbose_json` | 105,545 | 3.09× |
| `/v1/listen` (defaults) | 100,698 | 2.94× |
| `/v1/listen?diarize=true` | 117,558 | 3.44× |
| `/v1/listen?diarize=true&utterances=true` | 246,951 | 7.22× |

The jump at `utterances=true` is because the shape carries every word twice —
once flat under `channels[]`, once nested under its utterance. That duplication
is what the real Deepgram API returns, and rule 1 forbids dropping half of it.
It is also why honouring Deepgram's `utterances=false` default matters: the
expensive view is opt-in at the vendor's own switch.

Attribution cost is not a factor. The endpoint selects a projection over a
result the pipeline has already computed; per-word attribution runs regardless
and is ~0.2% of pipeline time (ADR 0008).

## The API tree is grouped by provider, not by version

If each provider owns its own contract, the module tree should say so:

```
coro/api/
  schemas.py          Strict Transcription Response Schema (provider-agnostic)
  utterances.py       speaker-turn grouping, shared by provider projections
  exceptions.py       typed failures, provider-agnostic
  health.py           not a provider surface
  openai/             POST /v1/audio/transcriptions, its schemas, SSE, errors
  deepgram/           POST + WebSocket /v1/listen, its schemas, live frames
```

The previous `coro/api/v1/` grouped by *path version*, which fails on its own
terms: it held OpenAI's `v1` and Deepgram's `v1` side by side, two unrelated
numbers owned by different companies. It does not scale either — AssemblyAI's
`/v2/transcript` and Deepgram's WebSocket `v2` would land in one `v2/`
directory with nothing in common. A version is a fact *about a provider*, not
an axis the codebase shares.

This is the opposite choice from ADR 0007, which moved `coro/backends` from
provider-first to capability-first, and deliberately so. Backends are
interchangeable implementations of one internal protocol, so the capability is
the stable axis and the provider is a detail. API surfaces are the reverse:
each provider is a *different external contract*, no two are substitutable, and
the provider is the only stable axis.

**Paths are unchanged and must stay unchanged.** The tree is an internal
concern; `/v1/audio/transcriptions` and `/v1/listen` are what each vendor's SDK
expects, and moving a module must never move a route. Asserted by
`tests/test_supported_endpoint_set.py`.

## Consequences

- `coro/api/deepgram/listen.py` is a second router. It does not raise
  `TranscriptionError`, because the app-wide handler renders OpenAI-style
  errors; it returns Deepgram-shaped bodies directly. Any future provider
  endpoint must do the same.
- The app-wide error body is OpenAI's, which is why the handler now lives in
  `coro/api/openai/errors.py`. That is a legacy of `coro` being OpenAI-first,
  named rather than hidden.
- `deepgram-sdk` becomes a dev dependency, used only to validate responses
  against the vendor's published types in tests. `numpy`, `soxr` and
  `websockets` become runtime dependencies for the live transport.
- `ResponseFormat` is unchanged and carries only values OpenAI defines,
  asserted by an exact-membership test.
- Authorization is accepted and never validated. Both vendors' SDKs always send
  a token, and rejecting requests without one would add authentication `coro`
  does not otherwise have.
- No package `__init__` re-exports anything; consumers import the module they
  need, so a re-export shim cannot accumulate unread.
- **AssemblyAI is not implemented here.** Its contract is asynchronous —
  `POST /v2/upload`, `POST /v2/transcript` returning `queued`, then polling
  `GET /v2/transcript/{id}` — which needs a job-state subsystem `coro` does not
  have. Implementing it as a synchronous single POST would be a partial clone
  of exactly the kind rule 1 forbids. Deferred to its own issue; when it lands
  it gets `coro/api/assemblyai/`.
