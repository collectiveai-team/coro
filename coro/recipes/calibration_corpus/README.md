# calibration_corpus

Builds a real, multi-language, multi-duration audio calibration corpus for
static-QDQ encoder quantization — the corpus
[parakeet_prompt_encoder_static_qdq](../parakeet_prompt_encoder_static_qdq/README.md)
consumes.

## What it does

1. **Spanish** (`es`): resolves the FLEURS `es_419` test shard via
   `huggingface_hub.snapshot_download` (cached locally after the first
   run), then scans `test.tar.gz` in place — no full-dataset extraction —
   picking clips within a target duration window (3-15s).
2. **English, French, German, Portuguese**: streamed directly from HF's
   auto-converted Parquet export via `coro.bench.utils.hf_parquet`, the
   same mechanism `coro-bench`'s own Spanish workload materialization
   uses, just pointed at `hf_config` values not registered in
   `SPANISH_CORPORA`. Only as many Parquet row-groups as needed to satisfy
   the per-language target are fetched — no full dataset download.
3. Writes `<out-dir>/<lang>-<id>.wav` (16 kHz mono) plus a
   `calibration_manifest.json` recording language/duration/source per
   clip.

## Worth knowing

- **The encoder never sees `target_lang` — so "cross-language forced
  calibration passes" are not needed.** This was a real hypothesis from an
  earlier `diagnose`-skill root-cause session that got retracted after
  reading `onnx_asr.models.nemo.NemoConformerRnnt._encode`: the encoder's
  ONNX input contract is `audio_signal`/`length` only. What actually
  matters for calibration quality is **acoustic diversity** (languages,
  speakers, durations, recording conditions) so MinMax activation ranges
  are representative of what the encoder sees in production, since these
  encoders serve audio in up to 13 languages.

- **Fixed two portability bugs from the original `.tmp/build_calibration_corpus.py`
  when this was promoted out of `.tmp/`.** The original script (a) did
  `sys.path.insert(0, "/home/.../evaluate-language-constrained-onnx-asr-qwen3-asr")`
  to reach `coro.bench.utils` from a sibling worktree, and (b) resolved the
  FLEURS Spanish snapshot via a hardcoded blob-hash path
  (`.../snapshots/70bb2e84.../data/es_419`) specific to whatever host
  happened to have downloaded it first. Neither survives moving to a new
  machine or a fresh cache. This version imports `coro.bench.utils`
  directly (a real first-party package, no path hack needed — this recipe
  *is* part of `coro` now) and resolves the FLEURS snapshot through
  `huggingface_hub.snapshot_download` (downloads once, reuses the cache
  thereafter, works on any host).

- **Best-effort on the remote languages.** A network hiccup fetching one
  remote language (`en`/`fr`/`de`/`pt`) is caught and logged, not fatal —
  the recipe still produces a usable (if smaller) corpus rather than
  aborting entirely on a transient failure.

## Usage

```sh
uv run --extra recipes --extra cpu -m coro.recipes.calibration_corpus
uv run --extra recipes --extra cpu -m coro.recipes.calibration_corpus \
    --out-dir recipe-artifacts/calibration_corpus --per-language-target 8
```

Requires the `recipes` extra's `pyarrow` dependency (streaming Parquet
reads via `coro.bench.utils.hf_parquet`) in addition to `recipes` itself.

## Consumers

[`parakeet_prompt_encoder_static_qdq`](../parakeet_prompt_encoder_static_qdq/README.md)'s
`--calibration-manifest` defaults to this recipe's
`calibration_manifest.json` output location.
