# Streaming Diarizer Reads the Latency Tier Off a Tier-Bound Model View

## Status

Accepted. Amends ADR 0010 ("Shared-state hazard"): the per-call
apply-and-restore scoping is replaced on the request path.

## Context

NeMo's `forward_streaming_step` reads the latency-tier parameters
(`chunk_len`, `chunk_right_context`, `fifo_len`, `spkcache_update_period`,
`spkcache_len`) off `self.sortformer_modules` at call time. ADR 0010 stopped
the streaming factory from writing them onto the shared model permanently by
applying them around each call and restoring afterwards.

That scoping is not safe under concurrency. Streaming requests run chunks on
worker threads, and two overlapping scopes interleave:

1. A saves the checkpoint values and writes the tier.
2. B saves **the tier** as its "previous" values and writes the tier.
3. A returns and restores the checkpoint values while B is still in the call.
4. B's step reads the checkpoint values — `fifo_len=0`,
   `spkcache_update_period=188` on `diar_streaming_sortformer_4spk-v2`.
5. B returns and restores the tier values: the shared model stays retuned.

Reproduced on the real model: one pair of concurrent `low`-tier streams left
the shared model at the `low` tier permanently. A deterministic unit test
forces steps 1–5 and shows B's chunk running with the checkpoint's values.

In a server, every streaming request shares one tier, so after the first
overlap the model is stuck at the tier and later overlaps swap tier for tier:
the damage is the one misconfigured chunk plus the permanent retune, which
breaks the batch-vs-streaming isolation ADR 0010 exists to guarantee.

## Decision

`NemoStreamingDiarizerFactory` builds a **tier-bound view** of the model once,
with `bind_latency_tier`: a shallow copy of the model whose
`sortformer_modules` is a shallow copy carrying the tier. Every request of the
factory calls `forward_streaming_step` on that view. Nothing on the request
path writes to any model object, so no lock is needed.

`applied_streaming_params` stays for the single-threaded diarization bench,
which applies a tier around sequential batch calls; its docstring now says it
is not for concurrent use.

## Consequences

- Concurrent streaming requests read their own tier and never retune the
  shared model; the batch Diarization Adapter is fully isolated.
- The view shares every parameter tensor and submodule (encoder,
  transformer, the sortformer linear layers); only two small Python objects
  are copied. Verified on the real 471 MB model: the view's encoder and
  weights are the original objects, not copies.
- `StreamingDiarizer` loses its `tier_params` argument: the model it receives
  already carries the tier.
- New dependency on NeMo internals: `nn.Module` resolving submodules through
  `_modules`, which the view replaces with its own dict. Covered by the
  env-gated real-model test comparing concurrent to sequential streams.
- The view is a snapshot of the model's Python attributes at construction; a
  later `model.train()`/`eval()` or `.to(device)` on the original is not
  mirrored on the view's own attributes. Neither happens after startup.
- The Conformer encoder can grow its positional-encoding buffer mid-call
  (`update_max_seq_length`), another write to shared state. Streaming steps
  stay far below the preallocated 5000 frames (at most spkcache + FIFO +
  chunk + right context = 608 at the `very-high` tier), so it is not reached; a non-streaming Sortformer on very long batch
  audio could reach it, and that is not addressed here.

## Alternatives Considered

- **Lock around each streaming call.** Rejected, as for the batch flow (ADR
  0021): it serialises every streaming request on the model.
- **One tier value written permanently at startup when the pipeline is
  streaming.** Rejected: it reintroduces the batch/streaming coupling ADR 0010
  removed, and silently depends on no process ever using both flows.
- **A full model copy per tier.** Rejected: 471 MB per copy to isolate five
  integers.
