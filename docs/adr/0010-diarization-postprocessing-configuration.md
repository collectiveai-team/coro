---
status: accepted
---

# Diarization Post-Processing Configuration

Sortformer's raw frame-level speaker-activity predictions are turned into
speaker segments by a VAD-style hysteresis post-processing step — six
thresholds (`onset`, `offset`, `pad_onset`, `pad_offset`, `min_duration_on`,
`min_duration_off`) from Medennikov et al.'s TS-VAD scheme. NeMo's own model
card for `diar_streaming_sortformer_4spk-v2` ships two dataset-optimized
presets for this step (`dihard3-dev`, `callhome-part1`) and its own benchmark
table shows tuning them measurably improves DER on every reported domain.
Coro today calls this step with no configuration at all — both the batch
Sortformer adapter (`coro/backends/diarization/nemo/diarization.py`, used by
the Full-Memory Pipeline) and the streaming adapter
(`coro/backends/diarization/nemo/streaming.py`) resolve to
`load_postprocessing_from_yaml(None)`, which is NeMo's raw, unsmoothed
baseline: `onset=0.5, offset=0.5, pad_onset=0.0, pad_offset=0.0,
min_duration_on=0.0, min_duration_off=0.0`. No settings field, CLI flag, or
adapter parameter exists to change this on either pipeline.

A separate investigation isolated NeMo Sortformer's own Diarization Error
Rate from ASR-segmentation artifacts on a 30-meeting AMI ES workload and
found its error is overwhelmingly missed-detection (≈89% of total error,
vs. ≈4% speaker confusion) — exactly the failure mode these six thresholds
are meant to tune away from. That finding motivated this decision, but does
not resolve it: AMI ES is one acoustic domain (English business meetings,
close-talking headset mics), and is not established to be representative of
this project's actual deployments.

## Considered options

**Tune the six thresholds against a benchmark workload (AMI ES, or any
other single dataset) and ship the tuned values as coro's new default.**
Rejected. NeMo's own two published presets make the risk concrete:
`dihard3-dev`'s optimum (`onset=0.56, offset=1.0, min_duration_off=0.151`)
and `callhome-part1`'s (`onset=0.641, offset=0.561,
min_duration_off=0.296`) differ substantially from each other, because
"diverse challenging recordings" and "telephone conversations" are
genuinely different acoustic domains that want different thresholds. Coro
is a general-purpose, self-hosted server that accepts arbitrary uploaded
audio; there is no basis for assuming any one benchmark's optimum
transfers to an arbitrary deployment's real traffic, and shipping one
benchmark's numbers as the new global default would launder a
single-domain overfit as a general improvement.

**Leave post-processing unconfigurable, as it is today.** Rejected. NeMo
already built the mechanism this problem needs — a YAML-driven threshold
override, accepted directly by `SortformerEncLabelModel.diarize(...,
postprocessing_yaml=...)` and by the streaming path's
`load_postprocessing_from_yaml` — and coro exposes none of it. An operator
who *does* know their deployment's acoustic domain, and has (however
small) a representative labelled sample to validate against, currently has
no way to apply that knowledge; the raw defaults are not a deliberate
choice, they are just what happens when nothing is configured.

**Expose the existing NeMo mechanism as an operator-facing coro setting,
without coro computing or recommending specific numbers.** Accepted. A new
setting accepts either a named preset (`dihard3-dev`, `callhome-part1` —
vendored verbatim from NeMo's own repository, Apache-2.0, with their
original attribution comments kept intact) or a path to a custom YAML in
the same schema. Coro's own default stays unchanged (no post-processing
override, i.e. today's raw baseline) so behavior does not shift for anyone
who does not opt in. This is documentation and plumbing, not tuning: coro
ships the *capability*, and defers the *number* to whoever has the
domain-specific data to validate one — including, eventually, coro's own
maintainers, if a representative sample of this project's actual traffic
ever becomes available to validate against instead of a public benchmark.

## Consequences

The setting must resolve identically for both diarization flows: the batch
Sortformer adapter threads the resolved path straight into
`diarize(..., postprocessing_yaml=path)`, and the streaming adapter's
`StreamingDiarizer` must stop hardcoding `None` in `_default_post_process`
and use the same resolved path instead, so the Full-Memory Pipeline and the
Streaming Pipeline apply the same post-processing configuration for
identical settings — the same parity principle already established for
other Diarization Flow behavior.

An unrecognized preset name or a custom path that does not exist or does
not parse fails Strict Startup Validation loudly, the same as any other
Server Startup Selection value — never a silent fallback to the raw
default.

This decision does not endorse either vendored preset for any specific
deployment. Choosing one over the other, or supplying a custom YAML, is a
per-deployment operator decision this ADR deliberately does not make.

## Target scoring collar as preset provenance

Each vendored preset records the DER scoring collar it was optimised against
(`dihard3-dev` at 0 s, `callhome-part1` at 0.25 s) alongside the parameters
themselves. Zero-collar scoring rewards boundary precision and near-zero
padding; collar-tolerant scoring rewards generous padding and aggressive
short-segment deletion. Scoring a set against a collar it was not tuned for
measures the mismatch rather than the model, so the pairing governs *selection*
— `coro-bench-diar` picks the set matching its own `--collar` by default. It is
recorded as provenance and is explicitly **not** asserted as a prediction of
which set will score best.

## Speaker-count gate

NVIDIA's v2 results report this post-processing improving DER for four or fewer
speakers and *degrading* it at five or more (+0.26 to +0.66 absolute):
short-segment deletion removes the brief, fragmentary evidence the model has
for the additional speakers, so applying it unconditionally makes the worst
case worse.

When a post-processing configuration is in force, the gate estimates the
speaker count from the raw activity matrix — a speaker counts as present when
it exceeds a fixed onset threshold for at least 0.5 s in total — and bypasses
the tuned thresholds above the ceiling. The estimate uses fixed thresholds
rather than the configured ones, so the gate decision does not depend on the
parameters it is gating.

**The gate cannot fire on any currently shipped Sortformer revision.** They are
all 4-speaker models emitting a `T x 4` matrix, so the estimate can never
exceed 4. It is built now, with the ceiling as a setting rather than a
constant, so the behaviour is already correct when a >4-speaker Diarization
Model Selection is configured. This is a deliberate acceptance of presently
unreachable code, not an oversight — and it is why the batch adapter asks NeMo
for `include_tensor_outputs=True`: the gate must be evaluable without a second
inference pass.

## Shared-state hazard

`NemoStreamingDiarizerFactory.__init__` used to write the latency tier onto
`model.sortformer_modules` permanently. That object is shared with the batch
Diarization Adapter, and the streaming Sortformer revisions set
`streaming_mode=True`, so batch `diarize()` runs through `forward_streaming`
and reads exactly those attributes (`chunk_len`, `chunk_right_context`,
`fifo_len`, `spkcache_update_period`, `spkcache_len`). Constructing the
streaming factory therefore silently retuned batch diarization, invalidating
any batch-vs-streaming comparison made in one process.

The parameters cannot simply be dropped — NeMo reads them at call time — so
they are applied around each model call and restored afterwards, including on
exception. Tier validation at construction uses the same scoping, so it leaves
no residue. This makes construction and teardown safe; it does not make one
model object safe for concurrent use across different latency tiers, which
remains a pre-existing constraint.

## Measurement — recorded as evidence, not as grounds for a default

An A/B was run: 34 AMI meetings (Mix-Headset, 15.3 h),
`diar_streaming_sortformer_4spk-v2`, diarization only. One inference pass
cached the raw activity matrices and each arm was applied offline to those
cached predictions, so the arms differ only in post-processing. DER via
MeetEval, `regions=all`, duration-weighted.

| Arm | collar 0.0 | vs baseline | collar 0.25 | vs baseline |
|---|---|---|---|---|
| baseline (unconfigured) | 28.97% | — | 25.89% | — |
| `dihard3-dev` | 28.10% | −3.02% | 25.28% | −2.35% |
| `callhome-part1` | 26.40% | −8.89% | 20.75% | −19.86% |

| Arm | collar | better | worse | worst single-meeting regression |
|---|---|---|---|---|
| `dihard3-dev` | 0.0 | 33 / 34 | 1 | +1.54 pts |
| `dihard3-dev` | 0.25 | 32 / 34 | 2 | +1.62 pts |
| `callhome-part1` | 0.0 | 30 / 34 | 4 | +6.82 pts |
| `callhome-part1` | 0.25 | 31 / 34 | 3 | +0.55 pts |

**This does not change the default, for a reason beyond the original
single-domain-overfit argument.** Which preset is preferable turns on the
*decomposition* of the error — how much is missed detection versus false alarm
— because that is what padding and minimum-duration parameters act on. That
decomposition is a property of the reference the timelines were scored against,
not of the model alone. The same Sortformer timelines scored against different
defensible AMI references produce total DERs within about one point of each
other while disagreeing about whether the model over-detects or under-detects
speech. A preset chosen from the total would look validated under either; a
preset chosen from the decomposition is only as sound as the reference. Until
that is settled, this measurement is recorded and the default stays unset.

Two further scope limits: it is one corpus, one language and one microphone
condition, with no measurement of transfer; and the reference speaker-count
census over those 34 meetings is 33 meetings with exactly 4 speakers and 1 with
3 — none with 5 or more — so the ≥5-speaker case the gate exists for is
unmeasured here rather than estimated.

## Further consequences

Both Diarization Flows resolve the setting once and share it, and both apply
the same gate through one helper, so batch and streaming cannot drift apart in
how identical predictions become segments.

`None`, the empty string and the literal `none` all resolve to NeMo's
unconfigured baseline. The default is already the baseline, but an operator
templating the environment variable can only unset it by writing something, and
that must resolve rather than fail Strict Startup Validation.
