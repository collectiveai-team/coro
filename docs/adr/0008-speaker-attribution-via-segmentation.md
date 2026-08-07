---
status: accepted
---

# Speaker attribution via segmentation, not forced alignment

Segments were cut at punctuation only, so one segment could span a speaker
change while carrying a single speaker label. We will uphold the
**Single-Speaker Segment Invariant** with a **Speaker Boundary Split** — cutting
segments at speaker-timeline changes as well as at punctuation — rather than by
adding word-level speaker labels or a forced-alignment backend.

## Considered options

**Word-level speaker labels.** Rejected. The **Hypothesis STM** is written from
response segments only (`coro/bench/stm.py:36-67`), so a **Quality Benchmark**
cannot observe word-level labels and cpWER would not move. Exposing them to a
consumer would also require schema changes in three places across two other
repositories, whereas a **Speaker Boundary Split** ships through the existing
**Transcription API Contract** unchanged.

**A forced-alignment backend** (NeMo Forced Aligner, wav2vec2/MMS, or the
`canary-1b-v2` auxiliary CTC model). Rejected, and worth recording because it is
the canonical path elsewhere — WhisperX builds on it — so it will be proposed
again. It is unnecessary here for three reasons. Forced alignment exists to
rescue attention-based encoder-decoder models that emit no frame-level
information; the default ASR backend in practice is Parakeet TDT, which is
frame-synchronous and already emits a per-token emission time. A **Speaker
Boundary Split** needs only **Measured Word Start** values to decide which side
of a boundary a word falls on, never word ends or word durations. And no
consumer reads word timings at all today. An aligner would add a fourth
inference runtime to an image that already carries CTranslate2, onnxruntime and
torch, in exchange for boundaries the split does not use.

## Consequences

The split is gated on a **Speaker Attribution Gap** of at least 15% of cpWER,
measured on AMI ES 10-minute clips after **Overlap Token Acceptance** is in
place. Below that, this decision is not implemented and the aligner question
stays closed on the evidence.

**The gate passed.** On 61 ten-minute clips from the 30 AMI ES meetings the gap
is 19.56% of cpWER under the **Whisper English Text Schema** (95% CI 17.00–22.26%,
clustered by meeting; P(gap ≥ 15%) = 100.0% over 10,000 resamples). The verdict,
the issue 04 delta and the recorded deviations are in
`.scratch/speaker-attribution/gate-report.md`.

The threshold was fixed before any number was seen but did not name a text
schema, and the verdict depends on it — 5.87% on raw text, 15.09% under the
punctuation-stripping schema, 19.56% under the leaderboard one. Only raw (robust
fail) and leaderboard (robust pass) resolve it; the middle schema sits on the
threshold at P = 52.1% and cannot decide anything. That a percentage-of-cpWER
threshold is under-specified without its schema is a defect in how the gate was
written, and is why **a WER is meaningless without naming its text schema** is
now a CONTEXT.md invariant.

cpWER is insensitive to segment count, so it rewards fragmentation without
limit. The **Minimum Turn Threshold** exists to keep backchannels from shredding
transcripts, and **Segment Shape Counters** are reported so a cpWER win bought
with a segment explosion is visible in the same report.

The **Streaming Pipeline** persists segments before the speaker timeline exists,
so it applies the split at response assembly against the persisted segment's
words. That is why those words must carry **Measured Word Start** values rather
than **Interpolated Word Timing** — the honesty fix is an enabling condition for
pipeline parity, not a cosmetic one.

Word ends remain derived from the following word's start, because `onnx-asr`
discards the TDT duration prediction during decoding
(`onnx_asr/asr.py:222-223`). Nothing in this decision depends on word ends.
