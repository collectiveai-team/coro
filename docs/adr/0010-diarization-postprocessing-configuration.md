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
