# parakeet_prompt_kernel_cache

Extracts `parakeet-rnnt-1.1b-multilingual-prompt`'s `prompt_kernel` MLP
weights and prompt/vocab metadata from the original NeMo checkpoint — a
required artifact of `onnx-parakeet-prompt`'s directory contract, not an
optional quantization.

## What it does

Loads the checkpoint via `nemo.collections.asr.models.ASRModel.restore_from`,
then extracts:

- `model.prompt_kernel.state_dict()` — the small
  `Linear(hidden+num_prompts, 2*hidden) -> ReLU -> Linear(2*hidden, hidden)`
  MLP that conditions the encoder output on a forced language, saved to
  `prompt_kernel_cache.npz`.
- `vocab_size`, `blank_id`, and `prompt_dictionary` (the
  `{language: prompt_id}` mapping), saved to `prompt_kernel_cache.json`.

## Worth knowing

- **This is not a quantization recipe.** Every other recipe in this
  package produces an *optional* artifact a backend selector can swap in.
  This one produces a *required* part of `onnx-parakeet-prompt`'s artifact
  directory contract — the backend cannot start without it.
- **Why extraction is cached instead of loading the checkpoint every
  time.** Loading the full ~4 GB NeMo checkpoint just to read a small MLP's
  `state_dict()` costs 40-250s. Run this recipe once per checkpoint; the
  backend and every other recipe that touches this checkpoint's outputs
  read only the small `.npz`/`.json` this writes, not the checkpoint
  itself.
- **License-driven design constraint.** `parakeet-rnnt-1.1b-multilingual-prompt`
  is licensed under the NVIDIA Community Model License and NIM-gated for
  production use (see `coro/backends/asr/nemo.py`'s module docstring for
  the full summary). This recipe never downloads or redistributes the
  checkpoint — `--checkpoint` must already point at a `.nemo` file the
  caller has legitimately acquired — and its *output* (a small numpy MLP's
  weights, plus non-sensitive vocabulary-size/prompt-dictionary metadata)
  inherits the same redistribution restriction as the source checkpoint.
  Do not upload/publish `prompt_kernel_cache.npz`/`.json`.

## Usage

```sh
uv run --extra recipes --extra cpu -m coro.recipes.parakeet_prompt_kernel_cache \
    --checkpoint /path/to/parakeet-rnnt-1.1b-multilingual-prompt.nemo
uv run --extra recipes --extra cpu -m coro.recipes.parakeet_prompt_kernel_cache \
    --checkpoint /path/to/checkpoint.nemo --out-dir recipe-artifacts/parakeet_prompt
```

## Consumers

`coro/backends/asr/onnx_parakeet_prompt.py` loads
`prompt_kernel_cache.npz`/`.json` unconditionally as part of its artifact
directory contract.
