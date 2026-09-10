# Model Slug Registry

Picking Canary-1b-v2 (ADR 0019) as the default ASR Model Selection meant `model_asr` now needs to resolve four things together — backend, model id, encoder quantization, decoder quantization — not just the one field `onnx-asr`'s single-repo model catalogue needed. `MODEL_SLUGS` (`coro/settings.py`) is a small registry mapping a short slug to a complete, known-good configuration, resolved inside `ServerSettings` so CLI flags, env vars and `.env` all benefit identically:

| slug | `backend_asr` | `model_asr` | `asr_quantization` | `asr_decoder_quantization` |
|---|---|---|---|---|
| `canary-1b-v2` (default) | `onnx-canary-split` | `collectiveai/canary-1b-v2-onnx-split-int8` | `static_qdq_v4_pct_excl` | `dynamic_v1_quint8` |
| `parakeet-tdt-0.6b-v3` | `onnx-asr` | `nemo-parakeet-tdt-0.6b-v3` | — | — |
| `whisper-large-v3-turbo` | `faster-whisper` | `large-v3-turbo` | — | — |
| `whisper-large-v3` | `faster-whisper` | `large-v3` | — | — |

`coro serve` / `coro run` with zero flags now runs the full Canary INT8/INT8 stack; `--model-asr parakeet-tdt-0.6b-v3` (or the equivalent `CORO_MODEL_ASR`) switches to the previous default in one flag instead of four.

## Precedence is per field, not per request

The rule: an operator-set value always wins over the slug's own default; the slug only fills fields the operator left unset. This is deliberately finer-grained than "slug vs. no slug" — `ServerSettings(backend_asr="onnx-asr")` (no `model_asr` given) still resolves `model_asr` to the default slug's *model id*, `collectiveai/canary-1b-v2-onnx-split-int8`, and fills both quantization fields from that same slug, because only `backend_asr` was operator-set. That is very likely not what an operator setting `backend_asr=onnx-asr` alone actually wants, but per-field precedence is the only rule simple enough to specify and test exhaustively — a "coherent slug or coherent explicit set, no partial mixing" rule would need to define what counts as partial, and the acceptance criteria (`ServerSettings(backend_asr="onnx-asr", model_asr="nemo-parakeet-tdt-0.6b-v3")` must be an exact no-op versus pre-registry behaviour) already fixes the answer for the one case that matters in practice: giving both fields explicitly.

A `model_asr` that is not a registered slug key passes through **verbatim** as the raw model id/path for the given `backend_asr`, which must then be given explicitly — this is the pre-registry behaviour (`--backend-asr onnx-asr --model-asr nemo-parakeet-tdt-0.6b-v3`), kept byte-for-byte as a regression test. A non-slug `model_asr` with no `backend_asr` is a Strict Startup Validation error naming both fields, not a silent guess at intent.

## The `fp32` sentinel, not `none`

A slug's quantization defaults need an explicit way to say "off," distinct from `None` meaning "unset, let the slug or backend decide." `none` was rejected as that sentinel because the codebase already uses the string `"none"` for an unrelated, pre-existing convention (`backend_diarization: "none"` disables diarization outright) — reusing it here for "explicitly no quantization" on a *different* field would train two incompatible readings of the same literal into one settings module. `fp32` was chosen instead: it names what the operator actually gets (the unquantized graph) rather than what they are turning off, reads correctly in `--asr-quantization fp32`, and collapses to `None` unconditionally after slug/passthrough resolution (`FP32_QUANTIZATION_SENTINEL`) so no backend ever receives the literal string — every backend's own quantization branch only ever sees `None` or a real selector name.

`asr_quantization="fp32"` and `asr_decoder_quantization="fp32"` are independent: overriding one to fp32 does not touch the other, so `canary-1b-v2` with `asr_decoder_quantization=fp32` still gets the INT8 encoder.

## A pydantic v2 trap: mutation inside a `mode="after"` validator taints `model_fields_set`

`resolve_model_slug` is a `model_validator(mode="after")`, and `warn_ignored_asr_settings` (`coro/backends/asr/factory.py`) needs to know which fields the *operator* actually set, to avoid warning that a slug-filled quantization value is "ignored by this backend" when the operator never asked for it. The obvious tool is pydantic's own `model_fields_set`. It does not work here: assigning `self.x = y` inside a `mode="after"` validator retroactively adds `x` to `model_fields_set`, indistinguishable from the operator having passed `x` as a constructor kwarg. This is empirically verified pydantic v2 behaviour, not documented as a contract to rely on.

The fix is a snapshot taken at the very top of the validator, before any of its own mutations: `self._explicit_fields = frozenset(self.model_fields_set)` (a `PrivateAttr`, so it is not itself a pydantic field and cannot recurse into the same trap). `warn_ignored_asr_settings` reads that snapshot, never `model_fields_set` directly, and never after the validator has run — which is every successful construction, since it always runs unconditionally.

## Resolved values, always

After `resolve_model_slug` runs, every consumer downstream — the Backend Adapter Factory, the ASR Window Cache fingerprint (ADR 0016), CLI/log report lines — sees only the *resolved* values, never the slug string. This was a hard invariant from the PRD, not a convenience: the cache fingerprint in particular must key on what the backend actually loaded (a concrete model id and concrete quantization selectors), because two different slugs could theoretically resolve to configurations that collide or diverge in ways the slug name alone would hide from a cache key built on the string `"canary-1b-v2"` instead.

## Backward compatibility is a regression test, not a promise in prose

The PRD's non-negotiable invariant — `--backend-asr onnx-asr --model-asr nemo-parakeet-tdt-0.6b-v3` must be a no-op change across the whole registry landing — is asserted directly (`tests/test_settings.py::TestModelSlugRegistry`), not just documented. The same file also asserts that an unknown `model_asr` with an explicit `backend_asr` still passes both through verbatim, and that both these facts hold independently of anything the *default* slug ships with, so a future change to the default's own quantization defaults cannot silently break either guarantee.
