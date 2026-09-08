# Make Canary-1b-v2 (INT8/INT8) the default ASR model, with sticky language detection

PRD: `.scratch/canary-default/PRD.md`. Per-ticket detail with acceptance criteria:
`.scratch/canary-default/issues/01..08`. Those files are the contract; this file
is the planner's index into them. Read `CONVENTIONS.md`, `AGENTS.md` and the
relevant module docstrings before planning — the facts below are load-bearing.

## Outcome

`uv run --extra cpu coro serve` with no model flags loads Canary-1b-v2 from
Hugging Face (`collectiveai/canary-1b-v2-onnx-split-int8`, INT8 encoder
`static_qdq_v4_pct_excl` + INT8 decoder `dynamic_v1_quint8`, ≈1.29 GB download),
serves concurrent requests, forces the request `language` when given, otherwise
detects the language once per request with Canary's own partial-prompt LID and
holds it for every window of that request, falling back to
`CORO_ASR_FALLBACK_LANGUAGE` (default `en`) only if no window yields a language.
Operators select models by slug (`canary-1b-v2` default, `parakeet-tdt-0.6b-v3`,
`whisper-large-v3-turbo`, `whisper-large-v3`) and may override any resolved
parameter; `--asr-quantization fp32` turns a slug's quantization off.

## Baseline (branch `canary-encoder-int8-redo`, HEAD after `chore: orq-lite ignore rules`)

- Default today: `coro/settings.py` `backend_asr="onnx-asr"`,
  `model_asr="nemo-parakeet-tdt-0.6b-v3"`; tests asserting it:
  `tests/test_settings.py`, `tests/test_offline_run.py`.
- `coro/backends/asr/onnx_canary_split.py`: split-decode Canary adapter.
  `_kv_cache` is instance state → one-permit serialised admission. The decode
  prefix hardcodes `<|en|>` when no language is passed; `<|{language}|>` lookup
  raises `KeyError` for locales (`es-US`) and unsupported codes.
- `coro/backends/asr/factory.py`: provider dispatch, `PROVIDER_HONOURS_PROMPT`,
  `_PROVIDER_SPECIFIC_SETTINGS`, `warn_ignored_asr_settings` (compares to field
  defaults — will need the operator's explicit values once slugs fill fields).
- `coro/backends/asr/onnx_genai.py` already does "local path else
  `snapshot_download`" — reuse that pattern.
- `coro/pipelines/windowing.py` drives every pipeline (full-memory, streaming,
  live) through `ASRWindowing`; it is where request-scoped language state belongs.
- `coro/api/openai/render.py` reports `language or "unknown"` in verbose_json.
- Vocab (`.tmp/onnx_export/canary_split/vocab.txt`) has `<|emo:*|>`, 25 language
  tokens, `<|unklang|>`, `<|predict_lang|>`, `<|nopredict_lang|>`. NeMo's
  `Canary2PromptFormatter` `user_partial` role = `<|startofcontext|><|startoftranscript|>`
  for two decoder steps → emotion token, then source-language token.
- Quality history: encoder INT8 v3 rejected (+34.5% cpWER), v4_pct rejected
  (+19.9%), v4_pct_excl accepted (0.0508 vs fp32 0.0513); decoder static-QDQ
  rejected, dynamic quint8 accepted (0.0529, +43.8% RTFx). Combined INT8/INT8
  never measured cleanly in one process.

## Settled design — do not re-litigate

See the PRD's "Settled decisions" table. In short: speed gap accepted;
thread-local K/V cache; forced > sticky auto-LID (first successful window) >
fallback `en`; HF default with local-path override; fp32 encoder from
`istupakov/canary-1b-v2-onnx`; slug registry with explicit-override precedence;
`fp32` sentinel (not `none`); window geometry unchanged.

## Non-goals

Window-size changes, VAD, diarization, other backends' behaviour, beating
Parakeet's RTFx, removing any backend, publishing new HF artifacts, per-window
language switching.

## Invariants that must survive every ticket

- `uv run --extra cpu --group bench pytest -q` exits 0 and `uvx prek run
  --all-files` exits 0 at the end of every ticket. A test that cannot pass yet is
  landed as `@pytest.mark.xfail(strict=True)` and un-marked by the ticket that
  makes it pass.
- `--backend-asr onnx-asr --model-asr nemo-parakeet-tdt-0.6b-v3` behaves exactly
  as today.
- Adapters stay shared singletons; request-scoped state lives in pipelines.
- Cache fingerprint keys on resolved model/quantization/language, never on slugs.
- No hardcoded list of Canary languages — derive from the loaded vocab.
- Heavy runs go through `systemd-run` + `/usr/bin/time -v` (see CONVENTIONS.md).

---

## Ticket A — LID probe (`issues/01-lid-probe.md`)

Research only. Script under `.tmp/`; findings to
`.scratch/canary-default/findings-lid-probe.md` with a per-language table,
silence behaviour, and a recommendation: partial-prompt LID vs `<|unklang|>`.

Acceptance criteria: findings file exists with table + recommendation; tokens
decoded via `vocab.txt`; silence behaviour reported; no product code changed.

## Ticket B — Concurrent `onnx-canary-split` (`issues/02-concurrent-canary-split.md`)

Thread-local `_kv_cache`; auto-sized permits like `onnx-asr`;
`asr_max_concurrency` honoured; docstrings updated.

Acceptance criteria: real-thread race test (two different windows concurrently ==
sequentially); `serialized=True` removed for this backend;
`_PROVIDER_SPECIFIC_SETTINGS` updated + tested; gates green.

## Ticket C — HF artifact resolution (`issues/03-hf-artifact-resolution.md`)

`model_asr` = local dir or HF repo id; filtered `snapshot_download`; `fp32`
encoder from `istupakov/canary-1b-v2-onnx`; `hf_token` honoured.

Acceptance criteria: mocked-hub tests assert exact file lists for INT8/INT8 and
for fp32 (two repos); local dir makes no hub call; one real smoke transcript
pasted in the issue's Comments; docstring updated; gates green.

## Ticket D — Forced-language hardening (`issues/04-forced-language-hardening.md`)

Locale → base code; unsupported → 400 on all surfaces; `asr_fallback_language`
setting (default `en`) used when no language and no detection; warmup passes it;
fingerprint sees the resolved language.

Acceptance criteria: resolver unit tests (`es-US` ≡ `es`); 400 test for `ja`;
settings test for the fallback; warmup log shows language; fingerprint test;
gates green.

## Ticket E — Auto-LID, sticky per request (`issues/05-auto-lid-sticky.md`)

Implement the design Ticket A recommends. Request-scoped sticky state in the
windowing/pipeline layer for all three pipelines; `detected_language` on the
result; `verbose_json.language` reports it; explicit `language` skips detection.

Acceptance criteria: fake-adapter pipeline test (window 1 none → fallback,
window 2 `es`, window 3 forced `es`); verbose_json e2e; zero LID calls when
language given; real 3-window smoke pasted in Comments; streaming + live covered;
gates green.

## Ticket F — Slug registry + flip default (`issues/06-model-slug-registry-and-default.md`)

Registry as tabled in the issue; resolution in `ServerSettings`; explicit >
slug > unset; `fp32` sentinel; unknown id passthrough; `backend_asr` derived;
`warn_ignored_asr_settings` uses operator-set values; tests and CLI report line
updated.

Acceptance criteria: the six settings tests listed in the issue; empty-cache
`coro serve` log excerpt (≈1.29 GB, warmup OK) pasted in Comments; gates green.

## Ticket G — Combined gate (`issues/07-combined-quality-rtfx-gate.md`)

Three interleaved arms in one process on the 48-window mTEDx gate under
`systemd-run`: fp32/fp32 forced-es; INT8/INT8 forced-es; INT8/INT8 auto-LID.
Table appended to `docs/benchmark.md`.

Acceptance criteria: wall-stats log exists; table with cpWER/RTFx/LID hits;
misses listed with emitted token; thresholds stated with explicit pass/fail;
no product code changed (file a follow-up issue instead).

## Ticket H — Docs (`issues/08-docs-and-adrs.md`)

ADR 0019 (Canary default) + ADR 0020 (slug registry); `CONTEXT.md`; README and
`docs/benchmark.md` quick-starts; purge Canary "never the default /
comparative reference" statements in code docstrings and recipe READMEs; mark
the old PRD non-goal superseded; PRD status done.

Acceptance criteria: both ADRs exist; `rg "never the default|comparative.reference"`
has no Canary hits; README quick-start verified by running it; prek green.

## Dependency order

A ∥ B ∥ C (independent) → D (after B) → E (after A, D) → F (after C, E) → G → H.
Do not start E before A's findings file exists. Do not start G before F's
`coro serve` smoke is pasted.

## Evidence required

For every ticket: the two gate commands' exit status. For C, E, F: the smoke
command and output in the issue's `## Comments`. For G: the wall-stats log path
and the three-arm table. Result JSON is a claim; transcripts and logs are the
evidence.
