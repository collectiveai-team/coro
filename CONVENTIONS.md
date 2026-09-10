# Conventions for agent-driven changes

Read `AGENTS.md` and `CONTEXT.md` first; this file is the short operational
version that every orq-lite role receives verbatim.

## Commands

- Sync once: `uv sync --extra cpu --group bench`.
- **Every** `uv run` must carry `--extra cpu`. A bare `uv run` re-syncs to the
  default extras and uninstalls `onnxruntime`, breaking unrelated tests.
- Tests: `uv run --extra cpu --group bench pytest -q`.
- Lint/format/types: `uvx prek run --all-files` (ruff-format, ruff, pyrefly,
  hygiene hooks). CI runs the identical config.
- Heavy jobs (model loads, 48-window gates, quantization) run under
  `systemd-run --user --scope -p MemoryMax=… -p CPUQuota=…` with
  `/usr/bin/time -v -o .tmp/logs/<cmd>/<date>_wall-stats.log`, in the background
  with a saved PID, printing flushed per-step progress. Read the latest wall-stats
  log before choosing limits.
- Temporary files, scripts, downloads and artifacts go under `.tmp/`; never
  `/tmp`. Plans and issue notes go under `.scratch/`.

## Style

- Python 3.12, `from __future__ import annotations`, type hints everywhere,
  pydantic v2 for settings/models.
- Files 200–300 lines max; split by responsibility. Block docstring at the top of
  every public function/class; inline comments sparingly and only for the
  non-obvious "why".
- Reuse before creating: search for an existing helper/pattern (`rg`) and extend
  it; extract shared code instead of copy-pasting between backends.
- Adapter modules under `coro/backends/asr/` do not import each other's private
  helpers; runtime imports of `onnxruntime`/`onnx_asr`/`nemo` stay inside the
  builder functions so importing the module never requires them.
- Use the domain vocabulary from `CONTEXT.md` (ASR Backend Provider, ASR Model
  Selection, Adapter Concurrency Policy, Server Warmup, ASR Window Cache,
  Strict Startup Validation, OpenAI-Style Error).
- Conventional Commits.

## Architectural boundaries

- `coro/settings.py` is the single source of truth for runtime configuration;
  CLI flags and `CORO_*` env vars flow through `ServerSettings`. Validation
  happens at startup (Strict Startup Validation), not lazily.
- `coro/backends/asr/factory.py` is the only place that maps a provider name to
  an adapter builder.
- Pipelines (`coro/pipelines/`) own request-scoped state; adapters are shared
  singletons and must be safe to call concurrently unless their documented
  Adapter Concurrency Policy says otherwise.
- The ASR Window Cache fingerprint (`docs/adr/0016`) must include everything that
  can change a prediction — model, quantization, resolved language — and nothing
  that cannot.
- Request surfaces (`coro/api/openai`, `coro/api/deepgram`, live websocket) map
  backend errors to the OpenAI-Style Error shape; backends never raise HTTP.

## Canary-specific facts (load-bearing)

- `onnx-canary-split` artifacts: `encoder-model[.<q>].onnx(+.data)`,
  `xattn_kv.onnx`, `decoder_step[.<q>].onnx`, `vocab.txt`, `config.json`.
  Accepted selectors: encoder `static_qdq_v4_pct_excl`, decoder
  `dynamic_v1_quint8`. Do not "simplify" the encoder's node-exclusion list.
- HF repos: `collectiveai/canary-1b-v2-onnx-split-int8` (everything except the
  fp32 encoder); `istupakov/canary-1b-v2-onnx` (fp32 encoder, unmodified).
- Decoder prefix: `" " <|startofcontext|> <|startoftranscript|> <|emo:undefined|>
  <|src|> <|tgt|> <|pnc|> <|noitn|> <|notimestamp|> <|nodiarize|>`; for ASR
  `src == tgt`. Partial LID prefix is the first three tokens only.
- Never trust short-clip quality/speed numbers for Canary; validate on the
  48-window mTEDx gate (`.tmp/mtedx-drift`, template
  `.tmp/run_mtedx_drift_canary_encoder_quant.py`).
- Local artifact dir for experiments: `.tmp/onnx_export/canary_split`.

## Tests

- `tests/` mirrors `coro/`; backend tests under `tests/backends/asr/`.
- Prefer real threads and real (tiny) sessions for concurrency tests; mock the
  Hugging Face hub (`snapshot_download`), never the code under test.
- A test that should fail until a later ticket lands is committed as
  `@pytest.mark.xfail(strict=True)` and the marker is removed in the ticket that
  fixes it, so the test gate never goes red between tickets.

## Changes

- Fix the cause, not the symptom. Never weaken a gate to make a run green.
- Keep the diff scoped to the ticket; preserve unrelated user changes.
- Do not edit version strings, `uv.lock` by hand, or git config; no force-push.
- When a ticket asks for evidence (command + output), paste it under the issue's
  `## Comments` heading in `.scratch/canary-default/issues/`.
