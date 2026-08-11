# Response Segmentation: Sentence-First Boundaries with a Duration-Weighted Majority Speaker

Response segments are shaped by the transcript, not by the diarizer. Boundaries come from Spanish-aware sentence punctuation and are **never** cut at a word-level speaker change. Every word is attributed independently and keeps its own label; a segment's `speaker` is the **duration-weighted majority** of its own words, and is `-1` only when every one of its words is. Words also carry each ASR Adapter's real `start`, `end` and `score`, and both words and segments carry an additive `overlap` flag.

This records issue `12`'s operative decision (*"2026-08-09: the gate is resolved"*). It **supersedes a withdrawn draft** of this ADR that argued the opposite — speaker-homogeneous segments cut at every word-level speaker change, plus a flicker-correction pass. That draft was written before the exposure question was settled and claimed ADR number `0008`, which is retired and must not be reused.

## The decision this rests on, and why the order matters

Segmentation was never independently decidable. It follows from one prior question: *does per-word speaker truth leave the process?*

- If per-word speakers **do not** ship, then `segments[].speaker` is the only speaker surface a client has, and a sentence-shaped segment carrying its sentence's majority is externally **indistinguishable from the punctuation-majority word-relabelling rule** — the arm measured and rejected below. Speaker-first segmentation would then be the only way to carry a per-word win through a one-slot contract.
- If per-word speakers **do** ship, the majority label is an auditable *summary* sitting beside the truth, and sentence-first is correct.

That question was decided **yes** on issue `46`: per-word speaker labels ship via opt-in vendor-shaped `response_format` values (exposure option X1 — an additional format routing to the internal `TranscriptionResponse`), leaving `json`, `verbose_json` and `diarized_json` byte-unchanged so the OpenAI compatibility asset is preserved rather than spent. This ADR is downstream of that, and is only sound while it holds. **If per-word exposure is ever dropped, this decision must be reopened, not patched.**

## Segmentation policy

Boundaries were previously triggered by any of `.`, `!`, `?` or `,`. Treating the comma as a terminator fragments comma-heavy Spanish subordinate clauses into very short segments, which adds no information and makes any span-based decision noisier by shrinking the span. `coro/core/segmentation.py` applies three rules instead:

- only sentence-final punctuation (`.!?…`) closes a run;
- a maximum run duration (`MAX_SEGMENT_SECONDS`, 15 s) bounds long unpunctuated stretches that the comma previously bounded only by accident;
- the Spanish opening marks `¿` and `¡` pull a boundary *before* the token carrying them, so an interrogative or exclamative sentence is not orphaned at the tail of the preceding segment.

These constants are module-level policy, not settings: neither has a demonstrated need for per-deployment tuning yet, and promoting them is a later step to take on evidence.

Because boundaries depend only on the transcript, they are final the moment their closing token arrives — which is what lets the Streaming Pipeline spill a finalized run before the diarizer has produced any timeline at all.

## Speakers are decided per word

Attribution runs per word against the merged speaker timeline. Previously it ran once per segment and the winning label was stamped onto every word, so any segment spanning a speaker turn mislabelled every word after the turn — invisibly, because the words agreed with the segment by construction.

Three attribution rules go with it:

- **Unknown rather than arbitrary.** A span with no timeline overlap previously defaulted to speaker `1`, silently misattributing anything in a diarization gap. It now receives the `-1` sentinel. The `1` label survives only for an ASR-Only Server, where there is no timeline and no speaker claim is being made.
- **Gap-bounded timeline merging.** Coalescing consecutive same-speaker entries regardless of the gap turned a speaker either side of a long silence into one entry that outweighed everything inside the gap. Entries are grouped per speaker and merged only across a gap of at most `MAX_SPEAKER_MERGE_GAP_SECONDS` (0.5 s). Grouping per speaker also fixes a second-order defect: merging previously depended on entries being adjacent in the globally sorted timeline, so an interleaved speaker blocked a legitimate merge.
- **Overlapped speech is flagged, not silently collapsed.** The diarizer emits an independent timeline per speaker, so overlapped speech appears as concurrently active entries. The winner is still a single label — the contract is one speaker per word — but a word whose span contains two concurrently active speakers is marked.

## The segment label is a duration-weighted majority

Each word votes with its own duration, so a few long words outweigh many short ones. `-1` words **abstain**: they contribute no duration and are never counted, which is precisely what makes a segment `-1` only when all of its words are. Letting abstentions vote would let a diarization gap outweigh a speaker the diarizer did cover.

Ties break on total word count, then on the lowest speaker label — the latter mirroring `attribute_span`, so the pipeline is independent of timeline ordering and the degenerate all-zero-duration case is well defined rather than dictionary-order dependent. The rule has no tuned parameter: it compares measured quantities.

`word_segments` remains the concatenation of the segments' words, so the per-word truth and its summary cannot drift apart.

## What was measured

**Protocol.** Six AMI clips, 600 s each (`EN2001a`, `ES2002a`, `IB4001`, `IN1001`, `IS1000a`, `TS3003a`, all at offset 300 s), `Mix-Headset.wav` (AMI *IHM*), public AMI corpus only. ASR `nemo-parakeet-tdt-0.6b-v3` via onnx-asr on CPU; diarization `nvidia/diar_streaming_sortformer_4spk-v2` on CPU, default post-processing. Metric **WDER** (ADR 0009): speaker errors over the words present in both transcripts, under MeetEval's cpWER speaker assignment. `wder` charges an abstention as an error; `wder_claimed` excludes abstentions and so measures precision on the labels the system actually asserts. Reference: the per-clip `.ref.stm` derived from AMI's manual annotation. No DER collar is involved — WDER is a word-level metric with no collar.

**Every arm is built from one shared inference pass.** Each arm is a pure function of the same ASR tokens and the same diarization timeline, so re-running inference per arm injects run-to-run variation into a comparison that should be exactly deterministic. Tokens and timeline were dumped once and every arm rebuilt offline from them; the identical `scored` count (6697 words) across all arms is the check that this held.

**This matters, because the arms previously on disk are not comparable to each other.** Their hypothesis word counts are 8451, 8420, 8441 and 7964 across sessions — the ASR output itself moved as unrelated work landed on `main`. Any table splicing those runs together, including a straight re-score of them, measures ASR drift as much as response assembly.

| arm | surface | `wder` | `wder_claimed` | abstention | speaker errors |
|---|---|---|---|---|---|
| segment stamp (pre-per-word rule, same segmentation) | either | **0.1583** | 0.1573 | 0.12% | 1060 |
| word split (speaker-first) | either | 0.1786 | **0.1525** | 3.08% | 1196 |
| word split + sandwich rule | either | 0.1795 | 0.1534 | 3.08% | 1202 |
| **sentence-first + majority (this ADR)** | **per-word** | 0.1786 | **0.1525** | 3.08% | 1196 |
| **sentence-first + majority (this ADR)** | **segments** | 0.1607 | 0.1597 | 0.12% | 1076 |

The baseline arm here is *not* the historical `big_before` run. It is the pre-per-word speaker rule — one attribution per sentence, stamped onto every word — evaluated on **today's** segmentation, so that per-word attribution is the only variable. The historical baseline also changed the segmentation policy and cannot isolate it.

Four things follow, and the third is not favourable.

1. **Per-word labels are unchanged by this ADR.** The per-word row is bit-identical to the speaker-first arm — same `wder`, same `wder_claimed`, same 1196 speaker errors. Dropping the speaker-change split changes *where boundaries go*, not what any word is labelled. This was a requirement, and it is verified rather than asserted.

2. **Dropping the sandwich rule is not a cost.** Removing it improves both `wder` (0.1795 → 0.1786) and `wder_claimed` (0.1534 → 0.1525), six fewer speaker errors. The effect is small — consistent with issue `12`'s "word-label-neutral" reading — but it is not negative, so the rule loses its premise without anything to trade for it.

3. **On the surface clients see today, this is a wash, not a win.** At segment granularity the majority summary scores 0.1607 against the segment-stamp rule's 0.1583, with 1076 speaker errors against 1060: 2 clips better, 3 worse, sign test p = 1.000 — statistically indistinguishable. This is exactly what issue `12` predicted when it warned the summary would be "externally indistinguishable" from the rule it rejected. **The gain from per-word attribution is only visible at per-word granularity** (`wder_claimed` 0.1525 against 0.1573), and only reaches a consumer once exposure ships.

4. **Per-word attribution trades coverage for precision, and total WDER charges the trade.** It abstains on 3.08% of words where the segment-stamp rule abstains on 0.12%, so `wder` — which counts an abstention as an error — is worse on 6 of 6 clips (p = 0.031) while `wder_claimed` is better. That is a real, reportable cost of the `-1` sentinel, not an artefact: the sentinel exists because inferring a speaker for a word the diarizer never covered is a guess dressed as a measurement.

### Measured end-to-end through the API

Everything above was measured in-process. It has now been confirmed **on the wire**, which
was previously impossible: the Quality Benchmark requested `diarized_json`, which carries no
word field, so `hyp_response_to_stm` took its `segments` fallback on every response it ever
received. Every WDER figure in this repository's history before this run therefore scored the
**segment majority summary**, not per-word labels.

Both arms ran back to back against **one** server process — models loaded once — so the arms
differ only in which wire surface the transport requested. The check that this held is that
`scored` (6697), `correct` (4638) and `substitutions` (2059) are **identical** across both:
same audio, same ASR output, same words. 6/6 clips succeeded in both arms, no failures.

| wire surface | `wder` | `wder_claimed` | abstentions | speaker errors | errors on claimed words | cpWER | DER |
|---|---|---|---|---|---|---|---|
| `POST /v1/audio/transcriptions` → `diarized_json` (segments fallback) | **0.1607** | 0.1597 | 8 (0.12%) | 1076 | 1068 | 0.6294 | 0.7508 |
| `POST /v1/listen` → per-word speakers | 0.1786 | **0.1525** | 206 (3.08%) | 1196 | **990** | 0.6430 | 0.7409 |

**The naive reading is backwards.** Total `wder` looks *better* on the summary surface
(0.1607 vs 0.1786), and it is not a better result: the majority label collapses 206
abstentions into 8 by inheriting a segment's label for words the diarizer never covered. On
`wder_claimed` — the labels the system actually asserts — the per-word surface is **better**
(0.1525 vs 0.1597) with **78 fewer errors on claimed words** (990 vs 1068). On the normalized
lane the gap is wider: 0.0988 against 0.1159, a 14.8% relative reduction.

So per-word attribution is *more* accurate where it commits and honest where it does not,
and the summary surface hides both facts. Any WDER comparison that does not state its surface
is uninterpretable; `coro-bench quality --deepgram` is the one that measures per-word.

**These figures were predicted before they were run.** The offline rebuild
(`.tmp/score_arms_offline.py`, every arm from one shared token dump) predicted 0.1786 /
0.1525 / 3.08% for the per-word surface and 0.1607 / 0.1597 / 0.12% for the summary; the wire
returned 0.17859 / 0.15252 / 3.076% and 0.16067 / 0.15967 / 0.119%. Agreement to four decimals
across both surfaces is strong evidence that the offline instrument models the shipped path
faithfully, and that the fallback diagnosis was exact.

**The single-clip cpWER regression is settled.** It reproduces in *direction* — the per-word
surface is worse on 5 of 6 clips — but it is **not significant** (sign test p = 0.219), and it
is **not caused by this ADR**: the speaker-first and sentence-first-majority arms are identical
on every clip and every metric at per-word granularity. The cost is abstention, not
misattribution; on the claimed-labels lane the two are indistinguishable (p = 1.000). The
originally recorded figures (0.5673 → 0.5836) came from a contaminated pair of inference runs
and are not reproducible as numbers — only as a direction.

**ORC-WER is not computable on this workload.** MeetEval's MIMO matching is exponential in the
number of hypothesis streams and documented as intended for 1–2; these clips carry up to 5
speakers, where it self-reports requiring >4 TB. Recorded as unavailable rather than skipped.
The benchmark's own ORC-WER column comes from a different, non-exponential path and is
reported per clip above.

### The rejected alternative, kept because the distinction is easy to lose

The first implementation was the punctuation-bounded majority vote from MahmoudAshraf/whisper-diarization (`get_realigned_ws_mapping_with_punctuation`): **relabel every word** that disagrees with its sentence's duration-weighted majority. Measured over the same six meetings and rejected on the evidence. It behaved exactly as specified — unpunctuated speaker changes between two real speakers fell from 225 to 7 — and that was the problem: auditing what it flattened, **62% were clean two-way turns, not flicker**, and DER speaker-error rose to 204.32 s against a 185.08 s baseline, degrading on 6 of 6 clips (sign test p = 0.031). Its upstream design target is different: it compensates for "minor time shifts", runs *after* a punctuation-restoration model, and its guard flattens even a balanced 5/5 split. None of those assumptions hold for spontaneous multi-party meeting speech.

Those DER figures come from the Quality Benchmark's `der`, whose full protocol is in `docs/benchmark.md`: MeetEval `md_eval_22`, **collar 0 s**, overlapping speech scored (`regions="all"`), no UEM, reference from AMI's manual annotation rebased per clip, hypothesis the **ASR response segments**. Issue `16` established that this scores response assembly rather than the diarizer's timeline, which makes it applicable *here* but means it **must not be called diarization DER**; the diarizer's own DER comes from `coro-bench-diar`.

**Rejecting word relabelling is not an argument against the segment-level majority.** The two rules read the same evidence and write to different places: one overwrites the per-word truth, the other only summarises it beside it. Conflating them is the single easiest mistake to make here, and it is why `coro/core/realignment.py` is retained — as the record of a measured negative, out of the default path, with word-label smoothing refiled as issue `17`.

The sandwich rule in that module is also now premise-less. It existed because splitting on every raw speaker change tripled *flicker* (11.6% → 27.6% of segments) and doubled one-word segments. With sentence-first boundaries, segment count no longer depends on the timeline at all: 725 segments against the split arm's 1249, and 132 one-word segments against 462. Flicker is bounded by construction.

## Contract change: additive `overlap` flag

`WhisperWord` and `WhisperSegment` gain `overlap: bool = False`, mirrored on `TranscriptWord` and `ResponseSegment`. The schema is `extra="forbid"`, so a new field is a contract change and is recorded rather than slipped in.

The alternative was a composite `speaker` label (`"1+2"`). Rejected: `speaker` is passed through verbatim into the Hypothesis STM used for cpWER and WDER scoring, so composite labels would manufacture speakers appearing in no Reference STM and degrade the metric this work is judged by. A separate boolean keeps labels scorable and keeps the flag ignorable. The `False` default means existing payloads still validate.

**A segment is flagged when *any* of its words is** — not when it is *predominantly* overlapped. A duration-weighted majority was proposed for symmetry with the speaker label and is rejected on measurement: over the six clips it would flag 123 of 725 segments where "any" flags 321, disagreeing on 198 — silencing the signal on **62% of the segments that genuinely contain overlapped speech**. Two further reasons: the information is unrecoverable, since `overlap` reaches no client through `diarized_json` and the per-word flag is internal, so the segment flag is the only place it can live; and "predominantly overlapped" is *segmentation-dependent* — sentence-first roughly doubles segment length, mechanically shrinking any within-segment fraction — so the field's meaning would drift whenever an unrelated policy changed, while "contains overlapped speech" does not. Symmetry with the speaker label is not worth a contract field whose meaning moves.

## Batch and streaming stay identical by construction

Both pipelines must emit byte-identical responses. The Streaming Pipeline's finalizer previously spilled fully assembled segments with a provisional speaker, and assembly overwrote the label. Instead the `TranscriptSpillStore` `segments` table stores each finalized run's transcript tokens (`tokens_json`), and assembly calls the same `build_response_segment` the batch builder calls, one stored run at a time. Parity is a property of sharing the function rather than of two implementations agreeing, and flat memory is preserved because only one run is resident. Segmentation is shared the same way: both paths drive `SegmentAccumulator`.

## Consequences

- `TranscriptSegment` is removed from the Project-Owned Transcript Model tree. It existed only as the intermediate the old "group, then stamp a speaker" builder needed.
- **`segments[].speaker` is no longer a homogeneity guarantee.** A segment may span a speaker turn, and any consumer treating the label as exact for every word inside it is now wrong. `diarization`, built from segment labels, is a sentence-granularity summary and not a speaker timeline — for the diarizer's timeline use `coro-bench-diar`.
- Segment counts change for the same audio: fewer boundaries from commas, none from speaker turns, some from the duration fallback.
- Consumers that assumed `segments[].words[].score` was always `1.0`, or that word timings tiled the segment span exactly, will see real, gappy values.
- **A word may end after its own segment's `end`.** The one-segment-lookahead overlap clamp shortens a segment to the next segment's start; it deliberately does not trim the words inside it. The segment span is a display bound, the word timings are the ASR Adapter's measurement, and trimming them would corrupt the per-word timings the Vendor-Native Endpoints publish. Batch and streaming agree on both values. This replaces the inherited assertion that a segment's last word ended exactly at the segment end, which held only while words were interpolated over the span.
- `coro/core/segmentation.py` is owned by this concern — Spanish punctuation policy. A competing use of that path for Speaker Boundary Split was superseded by this decision.
