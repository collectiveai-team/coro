# Flat-Memory Response Rendering

The **Streaming Pipeline** exists to hold bounded memory regardless of audio length, and it does — on the SSE path only. On the non-SSE JSON path, which is the default on both **Vendor-Native Endpoints** (`stream=false`) and the only path `coro run` has, peak heap grows **linearly with audio length and has no ceiling**. This ADR makes both paths bounded, and removes a duplicated type tree that turned out to be most of the cost.

## What was measured

Peak traced Python heap, synthetic transcript, one segment and one word per one-second window, ASR window cache disabled:

| windows | `stream()` (SSE) | `transcribe()` (JSON) |
|---|---|---|
| 50 | 346 KiB | 432 KiB |
| 500 | 386 KiB | 4,122 KiB |
| 2,000 | 493 KiB | 15,882 KiB |
| 8,000 | 506 KiB | 62,108 KiB |

SSE is bounded: 160× the audio costs 1.5× the heap. JSON is linear at **~7.9 KiB per transcript item** with no bound, so hours-long audio costs hundreds of megabytes of heap for the response alone.

Decomposing that 7.9 KiB shows the cost is **not** mostly where it was assumed to be:

| stage | per item | share |
|---|---|---|
| the pipeline's own `TranscriptionResult` | 0.97 KiB | 12% |
| `asdict()` + `TranscriptionResponse.model_validate()` at the route | +6.36 KiB | **80%** |
| projecting to `verbose_json` | +0.19 KiB | 2% |
| serialising to bytes | +0.41 KiB | 5% |

Four fifths of it is a round-trip through a Pydantic mirror of a type the pipeline already produced. `asdict()` is doubly wasteful here: `segments[].words` and `word_segments` hold the *same* `TranscriptWord` objects by reference, and `asdict()` expands both, so every word becomes two dicts before Pydantic copies it a third time.

Deleting that round-trip **measured a 60% reduction** — 7.9 KiB per item down to 3.1 KiB, and 62,108 KiB down to 24,837 KiB at 8,000 windows. Less than the 80% the decomposition projected, because removing the largest contributor simply moves the peak to the next-largest moment rather than subtracting cleanly from it. The remaining cost was still **linear in audio length**, which is why it was not the whole change.

Rendering incrementally removed the rest. The JSON path now matches the SSE path exactly:

| windows | `stream()` | `transcribe()` before | `transcribe()` after |
|---|---|---|---|
| 50 | 346 KiB | 432 KiB | 344 KiB |
| 2,000 | 493 KiB | 15,882 KiB | 493 KiB |
| 8,000 | 506 KiB | 62,108 KiB | 506 KiB |

A 123× reduction at 8,000 windows, and — the point — **bounded** rather than merely smaller.

## Correcting the record on the motivating figures

The head-to-head that prompted this work reported 283 MB of audio-proportional RSS for the **Full-Memory Pipeline** against 219 MB for the **Streaming Pipeline** over 30 minutes, and attributed the 219 MB residual to the materialised response. That attribution does not survive arithmetic and is recorded here so it is not repeated.

Decoded PCM at the canonical 16 kHz mono s16le is 32,000 B/s, so 30 minutes is **57.6 MB**, not the ~345 MB the comparison assumed — a factor-of-six error. The Streaming Pipeline's measured 66 MB saving therefore already exceeds the entire decoded PCM it eliminates, leaving no shortfall for the response to explain. The response accounts for tens of megabytes at that duration, not 219 MB.

**Where the remaining audio-proportional RSS goes is still unknown.** It is not resident Python data in either pipeline; the plausible candidates are inference-runtime arena growth, allocator high-water behaviour, and peak-RSS-over-process-tree sampling. That is a separate investigation, and it is not addressed here.

The case for this change is therefore not a fixed number of megabytes. It is that one of the two response paths is **unbounded in audio length while the other is bounded**, on a pipeline whose entire reason to exist is the bound.

## The guarantee was never tested

`tests/test_streaming_memory.py` asserted the SSE bound by comparing peak heap for short and long audio. Its fake ASR emitted every token at window-relative `start=0.0`, and `_seconds_to_bytes` floors an overlap of `0.0` s at one sample, so `accept_from` sits half a sample *after* each window's start and ASR Windowing reconciled away every token but the first. Both arms of the comparison transcribed a **one-segment transcript**, at every audio length, so the test would have passed against a fully materialising implementation.

The bound turned out to hold anyway, but that was luck. The fixture now emits mid-window tokens, and a guard asserts the transcript actually grows with audio length before the memory assertions are trusted — the same guard the response-parity suite already carries for the same reason.

## One type tree

`coro/api/schemas.py`'s `TranscriptionResponse` and its five item models were a field-for-field Pydantic duplicate of the **Project-Owned Transcript Model** dataclasses in `coro/core/models/` — identical field names in identical order, verified mechanically, in two type systems. They are **deleted**. The dataclasses are the single internal representation, and the vendor projections read them directly.

The boundary itself does not move. `JsonResponse`, `VerboseJsonResponse`, `DiarizedJsonResponse` and `DeepgramResponse` remain strict Pydantic models and remain the published contract, because those are *vendor* shapes. `TranscriptionResponse` was never a vendor shape; it was coro's own shape written a second time, and its only runtime job was to be the destination of the `asdict()` copy.

That copy did buy one thing: `extra="forbid"` validation that the pipeline really produced the expected structure, on every request. That check moves to **import and test time**, asserting the dataclass tree conforms field-for-field to each vendor projection's expectations, in the same spirit as the existing import-time renderer guard in the streamed done frame. The protection is retained; paying for it per request, forever, in proportion to audio length, is not.

## One renderer, both pipelines

The vendor projections render from a **Transcript Source**: an iterator interface over finalized segments and raw words, plus the scalars the vendor shapes need before their arrays. The **Streaming Pipeline** supplies a store-backed source that reads through the **Transcript Spill Store**; the **Full-Memory Pipeline** supplies a view over the lists it already holds.

Both pipelines therefore reach the wire through **one implementation per response format**. Byte parity between them stops depending on two implementations being kept in step by a fixture and becomes true by construction — the property the streamed done frame already argues for, applied one level up. A capability branch in the routes, rendering flat for one pipeline and materialised for the other, was rejected for exactly that reason: it would have created a second implementation of every format's bytes, held together only by tests.

The Full-Memory Pipeline gains nothing in bound from this — it holds the whole decoded PCM regardless — but it does gain the 80%, which no amount of work on the Streaming Pipeline alone could have given it.

`duration` and Deepgram's `metadata.duration` are computed by a dedicated pass over the source, because `verbose_json`, `diarized_json` and Deepgram all serialise a duration *before* the arrays it summarises. A SQL aggregate over the spill store would have been cheaper and was rejected: the stored segment ends are pre-clamp and the word ends live inside a JSON column, so an aggregate would be an upper bound rather than the exact value the materialised path reports, and `duration` is a published field the two pipelines must agree on byte-for-byte. A source is iterated several times regardless — once per array it feeds — so one more pass for a scalar is consistent and obviously correct.

## The response keeps its Content-Length

The rendered body is written fragment by fragment into a spool file beside the transcript store, then streamed back to the client with an explicit `Content-Length` taken from the file's size. Nothing is ever fully resident, and the HTTP framing is byte-identical to what the routes emit today.

The obvious alternative — a `StreamingResponse` with no length, which makes the server fall back to chunked transfer-encoding — was rejected even though it breaks nothing measurable. No test in the repository reads the response `Content-Length`, every real client handles chunked transparently, and ADR 0015 scopes vendor fidelity to route, request shape, parameter defaults, response schema and error body, so framing is out of its scope. But both vendors' own servers do send a length on non-streaming JSON, the difference would be invisible to the `oasdiff` gate that guards the REST contract (ADR 0013 documents that these routes publish no success schema at all), and the spool costs one extra write and read of a body already small relative to the transcript spill. Preserving framing exactly was cheaper than reasoning about who might notice.

Rendering the body twice — once to count bytes, once to send — would also have preserved the length at flat memory, at double the CPU. The spool trades disk for that, on a disk the request is already using.

## Errors still precede the first byte

The pipeline runs to completion and fills the source before any body byte is written, so capacity rejections, undecodable audio and processing failures still map to an **OpenAI-Style Error** or a Deepgram `err_code` body exactly as before. Rendering concurrently with transcription was rejected: no ordinary JSON client consumes a body incrementally, so it would buy no latency while making mid-render failures unreportable.

## One serialiser, chosen by experiment

The elements are rendered with `json.dumps` under Starlette's exact keywords — compact separators, `ensure_ascii=False`, `allow_nan=False` — because that is what `JSONResponse` uses and therefore what both routes already emit. Pydantic's own serialiser is **not** interchangeable with it: `model_dump_json` writes `1e-7` where `json.dumps` writes `1e-07`. Rendering elements per item through `model_dump_json` was the first design, and it would have produced identical bytes for almost every transcript and different bytes for some — the worst way for a byte-identity guarantee to fail. Which serialiser the routes actually used was settled by sending a `1e-7` timestamp through the live endpoint rather than by reading the framework's source.

## The `dirized_json` alias is removed

`dirized_json` was accepted as a typo-tolerant alias of `diarized_json`. It is removed: a misspelling that silently succeeds trains clients to depend on the misspelling, and the server cannot later tell a typo from an intent. `json_verbose` is retained, because unlike a dropped letter it is a plausible reordering of a real OpenAI name.

This is the one deliberate contract *narrowing* here. `response_format` is published as an enum in `/openapi.json`, so removing a member is a breaking change that the `oasdiff` gate will and should flag — unlike everything else in this ADR, which leaves the generated document byte-identical.

## What is unchanged

`transcribe()` still returns a `TranscriptionResult`, so the response-parity suite that pins the two pipelines byte-for-byte continues to pass untouched. The flat path is a sibling of it, mirroring how `stream()` already returns a store-owning frame rather than a materialised result. No request parameter is added to either vendor surface, and the generated OpenAPI document is unchanged.
