---
status: accepted
---

# Diarization Post-Processing Configuration

Sortformer's raw frame-level speaker-activity predictions are turned into
speaker segments by a VAD-style hysteresis post-processing step — six
thresholds (`onset`, `offset`, `pad_onset`, `pad_offset`, `min_duration_on`,
`min_duration_off`) from Medennikov et al.'s TS-VAD scheme. NeMo's own model
card for `diar_streaming_sortformer_4spk-v2` ships two dataset-optimized
presets for this step (`dihard3-dev`, `callhome-part1`).

Coro originally called this step with no configuration at all — both the batch
Sortformer adapter (`coro/backends/diarization/nemo/diarization.py`) and the
streaming adapter (`coro/backends/diarization/nemo/streaming.py`) resolved to
`load_postprocessing_from_yaml(None)`, NeMo's raw, unsmoothed baseline
(`onset=offset=0.5`, all pads and durations `0`). No settings field, CLI flag,
or adapter parameter existed to change it.

This ADR was first accepted in the form "expose the mechanism, ship no
default", explicitly declining to pick a parameter set on the grounds that
choosing one on a single corpus' evidence would launder a single-domain
overfit. That reasoning was sound but untested: no measurement existed. It has
now been made, and it changes the decision. The revised decision is below; the
superseded position is recorded at the end.

## Decision

**1. The post-processing parameters are operator-configurable.** A setting
accepts a vendored preset name, a path to a custom YAML in the same schema, or
`none`/empty to return to NeMo's unconfigured baseline. An unrecognized preset
name or a path that does not exist fails Strict Startup Validation loudly,
never a silent fallback.

**2. The default is `dihard3-dev`, not the unconfigured baseline.** This is an
evidence-driven reversal of the original "no default" position; the measurement
is below.

**3. Each vendored preset records its Target Scoring Collar,** and benchmark
lanes select by collar rather than inheriting a default. The main Quality
Benchmark scores at collar 0.0 (ADR 0002); `coro-bench-diar` defaults to 0.25.

**4. Post-processing is gated on estimated speaker count,** bypassing the tuned
thresholds above a configurable ceiling (default 4).

**5. Latency-tier streaming parameters are scoped to each model call,** so
building the streaming diarizer factory no longer retunes the batch adapter.

## Measurement

34 AMI meetings (Mix-Headset, 15.3 h), NeMo Sortformer
`diar_streaming_sortformer_4spk-v2`, diarization only — no ASR, so the numbers
isolate speaker quality. One inference pass cached the raw activity matrices;
each arm was then applied offline to those cached predictions, so the arms
differ only in post-processing and cost no additional inference. DER scored
with MeetEval via `coro.bench.quality.score_item`, `regions=all`, corpus DER
duration-weighted across all meetings.

Combined DER:

| Arm | collar 0.0 | vs baseline | collar 0.25 | vs baseline |
|---|---|---|---|---|
| baseline (unconfigured) | 28.97% | — | 25.89% | — |
| `dihard3-dev` | 28.10% | **−3.02%** | 25.28% | −2.35% |
| `callhome-part1` | 26.40% | **−8.89%** | 20.75% | −19.86% |

Per-meeting win rate against the baseline:

| Arm | collar | better | worse | worst single-meeting regression |
|---|---|---|---|---|
| `dihard3-dev` | 0.0 | **33 / 34** | 1 | +1.54 pts |
| `dihard3-dev` | 0.25 | 32 / 34 | 2 | +1.62 pts |
| `callhome-part1` | 0.0 | 30 / 34 | 4 | +6.82 pts |
| `callhome-part1` | 0.25 | 31 / 34 | 3 | +0.55 pts |

Error decomposition at collar 0.0, against the baseline:

| Arm | missed detection | false alarm | speaker error |
|---|---|---|---|
| `dihard3-dev` | −438 s | +37 s | −3 s |
| `callhome-part1` | −2951 s | +1667 s | +96 s |

### Why `dihard3-dev` is the default despite `callhome-part1` scoring better

Sortformer's error on AMI is missed-detection dominated (9880 s of 13364 s
total baseline error). Both presets attack that, but they buy it at very
different prices:

- `dihard3-dev` trades **12 s of recovered miss for every 1 s of added false
  alarm**. That is close to free, and it does not depend on the domain already
  being miss-heavy — which is why it wins on 33 of 34 meetings with a worst
  case of +1.5 points.
- `callhome-part1` pads aggressively (`pad_onset=0.229`, `min_duration_off=0.296`)
  and trades **1.8 s of recovered miss per 1 s of added false alarm**. That
  ratio is only favourable while the model is substantially under-detecting.
  On audio where it is not, the same padding is close to pure false alarm. Its
  larger AMI gain is therefore a bet on a failure mode, and its worst single
  meeting regresses by +6.8 points.

Shipping the bigger number would be exactly the single-domain overfit the
original decision warned about. Shipping the smaller, structurally robust gain
is defensible on this evidence; `callhome-part1` stays one setting away for
operators whose audio resembles its domain, and remains the collar-matched
choice for `coro-bench-diar`'s 0.25 lane, where it measures best.

### What the measurement contradicts

The expectation going in was structural: zero-collar scoring should reward
near-zero padding, collar-tolerant scoring should reward generous padding, so
the 0 s-collar set should win at 0 s and the 0.25 s set at 0.25 s.

**That did not reproduce.** `callhome-part1` — the collar-tolerant,
heavily-padded, telephony-tuned set — scored better at *both* collars,
including at 0 s. The collar-matching principle is still the right way to
*select* a set (a set tuned at one collar is not evidence about another), but
on this corpus the miss-heavy failure mode dominates the collar effect. The
pairing is recorded as provenance, not asserted as a performance prediction.

### Scope limits

- One corpus, one language, one microphone condition (AMI, English business
  meetings, close-talking Mix-Headset). Transfer to other domains is not
  measured.
- Diarization-only. Effect on end-to-end cpWER/DI-cpWER is not measured here.
- Deterministic: fixed model, fixed predictions, no sampling. There is no run
  variance to report, and no repetitions were run because repetitions would be
  bit-identical.
- The default only takes effect when NeMo diarization is explicitly enabled;
  the shipped `backend_diarization` default is still `none`, so the
  out-of-the-box server is unaffected.

## Speaker-count gate

NVIDIA's v2 results report this post-processing improving DER for four or fewer
speakers and *degrading* it at five or more (+0.26 to +0.66 absolute):
short-segment deletion removes the brief, fragmentary evidence the model has
for the additional speakers, so applying it unconditionally makes the worst
case worse.

The gate estimates the speaker count from the raw activity matrix — a speaker
counts as present when it exceeds a fixed onset threshold for at least 0.5 s
total — and bypasses the tuned thresholds above the ceiling. The estimate uses
fixed thresholds rather than the configured ones, so the gate decision does not
depend on the parameters it is gating.

**The gate cannot fire on any currently shipped Sortformer revision.** They are
all 4-speaker models emitting a `T x 4` matrix, so the estimate can never
exceed 4. It is built now, with the ceiling as a setting rather than a
constant, so the behaviour is already correct when a >4-speaker Diarization
Model Selection is configured. This is a deliberate acceptance of presently
unreachable code, not an oversight.

Consequently the AMI A/B could not measure the ≥5-speaker case: the reference
speaker-count census over the 34 scored meetings is **33 meetings with exactly
4 speakers and 1 with 3 — none with 5 or more**. The ≥5 arm is unmeasured on
this corpus and is reported as such rather than estimated.

## Shared-state hazard

`NemoStreamingDiarizerFactory.__init__` used to write the latency tier onto
`model.sortformer_modules` permanently. That object is shared with the batch
Diarization Adapter, and the streaming Sortformer revisions set
`streaming_mode=True`, so batch `diarize()` runs through `forward_streaming`
and reads exactly those attributes (`chunk_len`, `chunk_right_context`,
`fifo_len`, `spkcache_update_period`, `spkcache_len`). Constructing the
streaming factory therefore silently retuned batch diarization, invalidating
any batch-vs-streaming comparison in one process.

The parameters cannot simply be dropped — NeMo reads them at call time — so
they are applied around each model call and restored afterwards, including on
exception. Tier validation at construction uses the same scoping, so it leaves
no residue. This makes construction and teardown safe; it does not make one
model object safe for concurrent use across different latency tiers, which
remains a pre-existing constraint.

## Consequences

Both Diarization Flows resolve the setting once and share it, and both apply
the same gate through one helper, so batch and streaming cannot drift apart in
how identical predictions become segments.

Operators upgrading who had not set anything will see diarization output change
when NeMo diarization is enabled. The change is a measured improvement on the
only corpus where it has been measured, and `CORO_DIARIZATION_POSTPROCESSING=none`
restores the previous behaviour exactly.

## Superseded position

The originally accepted decision was to expose the mechanism and deliberately
ship no default, on the reasoning that coro is a general-purpose server
accepting arbitrary audio, that NeMo's two presets differ substantially from
each other because their domains do, and that there was therefore no basis for
assuming any one benchmark's optimum transfers.

That reasoning is retained for the *choice between the tuned sets* — which is
why the larger-gain `callhome-part1` is not the default. It is no longer
retained for *tuned versus untuned*, because the measurement shows the
unconfigured baseline is not a neutral choice: it is worse than both published
presets, at both collars, on 33 of 34 meetings for the selected default. "No
default" was not the conservative option it appeared to be; it was an
unmeasured one.
