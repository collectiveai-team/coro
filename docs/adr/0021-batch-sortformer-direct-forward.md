# Batch Sortformer Calls `forward` Directly, Without a Concurrency Limit

## Status

Accepted. Amends ADR 0005 (the batch adapter no longer uses `diarize()`) and
ADR 0010 (the post-processing configuration reaches the batch flow through the
shared gated helper, not through `diarize(..., postprocessing_yaml=...)`).

## Context

Two concurrent Full-Memory Pipeline requests with `--backend-diarization nemo`
made one fail mid-stream with an error message that was only a temp-file stem
(`'coro-nemo-f8xyxoxy'`, issue #81). The cause is NeMo's `diarize()`
convenience wrapper, which keeps per-call state on the shared model instance:

- `_diarize_audio_rttm_map`, keyed by the input file stem, written before
  inference and read back after it. An overlapping call replaces the map and
  the first call raises `KeyError('<its temp file stem>')`.
- Preprocessor `dither` and `pad_to`, saved and zeroed on entry, restored on
  exit — so one call's exit restores values under another call still running.
- The model's train/eval mode and NeMo's process-global log verbosity, handled
  the same way.

The inference itself does not share state. In eval mode, `forward` →
`process_signal` → `forward_streaming`/`forward_infer` only reads the model;
streaming state is a per-call local and the modules write attributes only in
`__init__` or in training-only branches. This is unchanged in NeMo 3.0.0.

## Decision

`NemoDiarizationAdapter` calls `SortformerEncLabelModel.forward` directly on
in-memory PCM, and turns the raw activity matrix into segments with
`apply_gated_postprocessing` — the same helper the Streaming Diarizer uses.
The fixed state `diarize()` sets per call (eval mode, `dither = 0`,
`pad_to = 0`) is set once at load by `prepare_for_inference`.

There is **no concurrency limit** on batch diarization: no lock, no semaphore,
no setting. Overlapping requests run in parallel on the shared model.

## Consequences

- Concurrent batch diarization is correct rather than serialised or racing.
- The temp WAV (≈115 MB for a 60-minute recording) and NeMo's manifest +
  lhotse reload disappear from every request.
- Batch and streaming now share the post-processing implementation outright,
  closing the drift ADR 0010 was guarding against.
- Segment boundaries are no longer rounded to 2 decimals by NeMo's RTTM-line
  formatting before parsing; `convert_diarization_segments` rounds to 3.
- The batch adapter now depends on `forward(audio_signal, audio_signal_length)`
  in addition to the APIs listed in ADR 0005. The env-gated real-model parity
  test (`CORO_RUN_REAL_MODEL_TESTS=1`) compares it against `diarize()` and must
  run on any NeMo upgrade.
- Unbounded concurrency means overlapping requests share the host's cores:
  PyTorch already parallelises one call across intra-op threads, so on CPU a
  second concurrent call mostly lowers per-request speed rather than raising
  throughput. Capacity is an operator/deployment concern, not an adapter one.
- `diarize()`'s global log-verbosity toggle no longer happens.
- The streaming flow's own shared-state race (ADR 0010, "Shared-state
  hazard") is resolved separately by ADR 0022.

## Alternatives Considered

- **Serialise `diarize()` behind a lock.** Rejected: correct but turns
  diarization into a queue; a second 60-minute request waits for the first.
- **A configurable diarization concurrency limit.** Rejected: pushes a tuning
  knob onto operators to compensate for a wrapper bug instead of removing it.
- **One model instance per concurrent request.** Rejected: multiplies model
  memory to avoid state that inference does not actually need.
- **Pass numpy arrays to `diarize()`.** Rejected: that path still writes
  `_diarize_audio_rttm_map` (always under `numpy_0`) and still toggles
  dither/pad_to/log level; it hides the `KeyError` without removing the race.
