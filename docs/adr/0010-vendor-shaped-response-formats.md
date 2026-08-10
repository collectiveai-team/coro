# Vendor-Shaped Response Formats and Their Fidelity Policy

`coro` assigns a speaker to every word and carries a real per-word confidence
(ADR 0008), and until now discarded both at the API boundary. Two opt-in
`response_format` values — `assemblyai_json` and `deepgram_json` — expose them.
The OpenAI-compatible formats are byte-unchanged.

## Why a third-party shape rather than an extended OpenAI one

There is no OpenAI-compatible slot for a per-word speaker. `verbose_json` word
objects are `{word, start, end}`, and `TranscriptionDiarized` has no word-level
data at all — its speaker lives on `segments[]`. Extending either means adding
fields the OpenAI SDK types do not declare.

`diarized_json` is a byte-exact clone of OpenAI's `TranscriptionDiarized`, and
that exactness is a compatibility asset: an OpenAI SDK client can point at this
server and parse the typed object. Adding fields to it would spend that asset
for every client in order to serve the few that want word-level data.

Both dedicated speech vendors already solved this, and their per-word speaker
slot is *native* rather than bolted on:

| vendor | turn-level speaker | per-word speaker |
|---|---|---|
| AssemblyAI | `utterances[].speaker` | `utterances[].words[].speaker` |
| Deepgram | `results.utterances[].speaker` | `words[].speaker` |
| OpenAI `diarized_json` | `segments[].speaker` | none |

Adopting a documented external shape means no schema to design, defend or
version, and existing client tooling can consume it. No OpenAI client ever
requests the new values, so `diarized_json` stays byte-identical to OpenAI
permanently — enforced by golden-byte tests in
`tests/test_openai_formats_unchanged.py`.

## Fidelity policy

A partial clone carrying a vendor's name is a liability, because clients will
hand it to that vendor's SDK. The standard `coro` already applies to
`diarized_json` is stated here for any vendor-shaped format:

1. **A vendor format is a documented subset, not a full clone.** It is
   structurally valid against the vendor's published schema, and validated
   against the vendor's own SDK types in `tests/test_vendor_sdk_conformance.py`
   — not against a local copy of the schema.
2. **Every field emitted is either measured or structurally required.** Fields
   the vendor's SDK marks required are populated, because a response the SDK
   cannot parse defeats the purpose; each is enumerated below with its source.
3. **Optional and unmeasured means omitted, never stubbed.** No field is
   emitted with a placeholder. This is what decides `speaker_confidence`.
4. **Vendor features `coro` does not compute are absent, not empty.**
   Summarization, sentiment, entities, chapters, paragraphs, topics, intents
   and search are omitted entirely rather than emitted as empty containers,
   which would imply the feature ran and found nothing.

Only the response *shape* is adopted. Vendor endpoints (`/v1/listen`,
`/v2/transcript`), auth schemes, request parameters, and AssemblyAI's async
submit/poll model are all out of scope; ADR 0001's Supported Endpoint Set is
unchanged.

### Required fields and their sources

| field | vendor requires | source |
|---|---|---|
| `status` (AssemblyAI) | yes | literal `completed`; a synchronous response has no other state |
| `id` (AssemblyAI) | no | the server request id, also in the logs |
| `audio_url` (AssemblyAI) | yes | the upload filename — see below |
| `metadata.request_id` (Deepgram) | yes | the server request id |
| `metadata.sha256` (Deepgram) | yes | SHA-256 of the uploaded bytes |
| `metadata.created` (Deepgram) | yes | ISO 8601 completion time |
| `metadata.duration` (Deepgram) | yes | measured audio duration |
| `metadata.channels` (Deepgram) | yes | `1`; uploads are converted to mono |
| `metadata.models` / `model_info` (Deepgram) | yes | configured ASR Model Selection and Backend Provider |
| `metadata.transaction_key` (Deepgram) | no | omitted — billing concept with no `coro` equivalent |

`audio_url` is the one field with no honest value: AssemblyAI's model is that
the client supplies a URL, whereas this endpoint receives bytes. It is
populated with the upload filename and is a **provenance label, not a
dereferenceable URL**. Omitting it is not an option — the SDK marks it required,
so the response would not parse, and rule 1 would fail.

## `speaker_confidence` is omitted

Deepgram's `words[].speaker_confidence` is the diarizer's posterior for the
assigned speaker. `coro`'s diarization adapters binarize their per-frame
probabilities into a speaker timeline before it reaches the Core Boundary, so
that quantity does not exist downstream.

Deriving a substitute was considered and rejected. The available quantity is
the winning speaker's share of the timeline overlap inside a word's span, which
`attribute_span` already computes. That is a measure of *timeline dominance*,
not of diarizer certainty: a word falling entirely inside one confidently-wrong
timeline entry scores 1.0. Publishing it under Deepgram's name would invite
clients to read it as a posterior, which is exactly the mislabelling rule 3
exists to prevent.

The field is `Optional` in Deepgram's schema, so omission parses cleanly and is
distinguishable from a low confidence. Revisit if and when per-frame posteriors
are surfaced past binarization; the `overlap` flag from ADR 0008 already covers
the specific case of concurrent speakers.

## Speaker label mapping

**The `-1` sentinel becomes `null`.** ADR 0008 uses `-1` for a word the
diarization timeline does not support. Both vendors spell that absence as a
null speaker. Emitting the literal `"-1"` would present it as a speaker *named*
`-1`, which is worse than absence — an AssemblyAI client would render it as a
participant. `null` is the vendors' native "no speaker" and preserves the
abstention.

**Numbering is passed through, not renumbered.** `coro` labels speakers from 1;
Deepgram's own output is 0-based and AssemblyAI's is `A`, `B`, `C`. Neither is
adopted. Renumbering would break correspondence with `diarized_json` for the
same audio, and speaker labels are passed verbatim into the Hypothesis STM used
for cpWER, DI-cpWER and DER scoring (ADR 0008), so a translation layer would
put the wire format and the scored format out of step. Both vendors type the
field loosely enough to accept the pass-through: AssemblyAI as `Optional[str]`,
Deepgram as `Optional[int]`.

## Utterances are derived from words, not from segments

An utterance is built as a maximal run of consecutive `word_segments` entries
sharing a speaker — not by mapping `segments[]` one-to-one.

Under ADR 0008 the two happen to coincide, because segments are already split
at every word-level speaker change. That coincidence is not durable: issue 12
is deciding whether segments become sentence-shaped with a duration-weighted
majority speaker, which would make a segment a *summary* over several speakers
and therefore an invalid utterance. Deriving turns from the per-word truth
keeps this projection correct under either outcome, and means the per-word view
and the turn view cannot disagree, because only one of them is a source.

## Payload size, measured

The richer shapes are opt-in because they are large. Measured on a synthetic
600 s workload — 1405 words, 157 segments, 3 speakers:

| format | bytes | vs `diarized_json` |
|---|---|---|
| `json` | 9,220 | 0.27× |
| `diarized_json` | 34,203 | 1.00× |
| `verbose_json` | 105,545 | 3.09× |
| `deepgram_json` | 246,951 | 7.22× |
| `assemblyai_json` | 251,090 | 7.34× |

**This is far above the ~2× originally estimated.** The cause is that both
vendor shapes carry every word *twice* — once in the flat word list and again
nested inside its utterance — and both word objects carry a speaker and a
confidence. The duplication is required for fidelity: it is what the real
vendor APIs return, and rule 1 forbids dropping half of it. Recorded as a
measured correction rather than treated as a defect, and it strengthens rather
than weakens the case for these formats being opt-in.

Attribution cost is not a factor. `response_format` selects a projection over a
result the pipeline has already computed; per-word attribution runs regardless
and is ~0.2% of pipeline time (ADR 0008). The coarse-grained compute switch is
the diarization adapter, which is already optional.

## Consequences

- `ResponseFormat` gains `assemblyai_json` and `deepgram_json`. Unknown values
  still fail as an OpenAI-style 400 on `response_format`.
- `assemblyai` and `deepgram-sdk` become dev dependencies, used only to
  validate responses against the vendors' published types in tests.
- The vendor projections need request-scoped provenance (request id, audio
  digest, model identity) that the OpenAI projections do not, so a
  `VendorContext` is threaded into the response builder.
- Streaming is unaffected: `stream=true` still emits the internal schema, which
  has always carried per-word speakers in its done frame.
