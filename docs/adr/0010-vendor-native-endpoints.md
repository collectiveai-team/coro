# Vendor-Native Endpoints, and the Fidelity Policy They Are Held To

`coro` assigns a speaker to every word and carries a real per-word confidence
(ADR 0008), and discarded both at the API boundary. Per-word speakers are now
reachable — through **the vendor's own endpoint, implementing the vendor's own
contract**, not through a new value on the OpenAI endpoint.

`POST /v1/listen` is added, serving Deepgram's pre-recorded contract.

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
5. **Quality parameters are ignored; contract parameters are refused.** This
   distinction is load-bearing and an earlier draft of this ADR got it wrong by
   ignoring everything.

   A *quality* parameter changes how good the transcript is. `punctuate`,
   `smart_format`, `numerals`, `filler_words`, `dictation`, `model` and unknown
   future flags are accepted and ignored, matching how the OpenAI endpoint
   already treats `model` and `temperature`. The client still gets a
   transcript; it is merely not tuned the way it asked.

   A *contract* parameter changes what the response **is**. Ignoring it returns
   a body that does not answer the request that was made:

   | parameter | what silence would mean |
   |---|---|
   | `callback`, `callback_method` | client is told to expect `{request_id}` and a later POST; gets a full synchronous body and no callback ever fires |
   | `summarize`, `sentiment`, `topics`, `intents`, `detect_entities`, `paragraphs`, `search`, `measurements` | client reads a response key that is simply absent |
   | `redact` | client believes PII was removed; it was not |
   | `replace` | client believes terms were substituted; they were not |
   | `multichannel` | client expects N channels; gets one |
   | `detect_language` | client reads `detected_language`; it is absent |

   These are refused with a Deepgram-shaped 400 naming the parameter. A refusal
   is a contract; silence is a lie with the vendor's name on it, which rule 1
   exists to prevent. `redact` is the clearest case: silently not redacting is
   a compliance failure disguised as a success.

   Refusal triggers on any value that is not an explicit disable, because
   several of these carry a payload rather than a boolean (`callback` a URL,
   `search` a term, `summarize` accepts `v2`). `summarize=false` asks for
   nothing and is not refused.

### Coverage: what of Deepgram's contract is implemented

This is a documented subset, and "documented" means the boundary is written
down rather than discovered by a client at runtime.

Deepgram's `listen` API has **two transports**, not two API versions of one
transport:

| surface | transport | status |
|---|---|---|
| `POST /v1/listen`, raw audio body | REST | **implemented** |
| `POST /v1/listen`, `application/json` `{"url": ...}` ingest | REST | **refused** with an explicit 400 |
| `wss://…/v1/listen` | WebSocket | **not implemented** |
| `listen/v2` | WebSocket **only** | **not implemented** |
| 37 pre-recorded parameters | REST | 3 honoured, 16 refused, 18 accepted and ignored |

`listen/v2` is not a newer pre-recorded API. It exposes only `connect` —
*"real-time conversational speech recognition with contextual turn detection"*
— with 13 parameters that are all streaming concerns (`eot_threshold`,
`eot_timeout_ms`, `eager_eot_threshold`, `encoding`, `sample_rate`,
`language_hint`). There is no `v2` REST surface at all: the SDK has
`listen/v1/media/` and no `listen/v2/media/`. So v2 is not a separate gap; it
is part of the streaming gap below, and nothing about it applies to a
pre-recorded endpoint.

**Honoured (3):** `diarize`, `utterances`, and `language` as a hint.

**Refused (16)** — rule 5: `callback`, `callback_method`, `summarize`,
`sentiment`, `topics`, `custom_topic`, `intents`, `custom_intent`,
`detect_entities`, `paragraphs`, `search`, `measurements`, `redact`,
`replace`, `detect_language`, `multichannel`.

**Accepted and ignored (18):** `model`, `version`, `punctuate`,
`smart_format`, `numerals`, `profanity_filter`, `filler_words`, `dictation`,
`keywords`, `keyterm`, `diarize_model`, `utt_split`, `tag`, `extra`,
`mip_opt_out`, `encoding`, `custom_topic_mode`, `custom_intent_mode`, and any
parameter Deepgram adds later.

`custom_topic_mode` and `custom_intent_mode` are modifiers, meaningless without
`custom_topic` / `custom_intent`, which are refused — so the pair is always
caught by its head parameter.

`encoding` deserves a note, as would `sample_rate` on the streaming transport:
these describe headerless PCM. `coro` decodes with ffmpeg, which sniffs the
container, so a headerless upload fails with the ordinary undecodable-audio 400
rather than being interpreted.
That is a real gap for raw-PCM clients, left open rather than papered over.

### Streaming is not implemented, and `coro` streams

`coro` has a Streaming Pipeline and serves SSE on
`/v1/audio/transcriptions?stream=true`, but in OpenAI's event framing
(`transcript.text.delta` / `.done`). Deepgram's streaming contract is a
different transport entirely — a WebSocket carrying `Results`, `Metadata`,
`SpeechStarted` and `UtteranceEnd` messages, with `interim_results`,
`vad_events` and `utterance_end_ms`. A Deepgram client cannot consume the SSE
stream, and none of the streaming parameters appear on the pre-recorded
endpoint, so there is no silent-ignore hazard here — the capability is simply
absent.

Both `listen/v1` `connect` (30 parameters) and the whole of `listen/v2` (13)
live on this transport. `listen/v2` adds contextual turn detection, which is a
capability `coro` has no equivalent of at all: its Streaming Pipeline emits
transcript deltas, not turn-boundary events.

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

## Consequences

- `coro/api/v1/listen.py` is a second router. It does not raise
  `TranscriptionError`, because the app-wide handler renders OpenAI-style
  errors; it returns Deepgram-shaped bodies directly.
- `deepgram-sdk` becomes a dev dependency, used only to validate responses
  against the vendor's published types in tests.
- `ResponseFormat` is unchanged. `deepgram_json` is **not** a value;
  `/v1/audio/transcriptions` rejects it, asserted by test.
- Authorization is accepted and never validated. Both vendors' SDKs always send
  a token, and rejecting requests without one would add authentication `coro`
  does not otherwise have.
- Streaming is unaffected, and Deepgram streaming is absent rather than
  approximated — see the coverage section above.
- `coro/api/vendor/__init__.py` re-exports nothing. Consumers import the
  submodule they need, so the package cannot accumulate a re-export shim that
  nothing reads.
- **AssemblyAI is not implemented here.** Its contract is asynchronous —
  `POST /v2/upload`, `POST /v2/transcript` returning `queued`, then polling
  `GET /v2/transcript/{id}` — which needs a job-state subsystem `coro` does not
  have. Implementing it as a synchronous single POST would be a partial clone
  of exactly the kind rule 1 forbids. Deferred to its own issue.
