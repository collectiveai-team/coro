"""Reproducible ONNX export/quantization recipes for comparative-reference ASR backends.

``coro/backends/asr/onnx_canary_split.py`` and ``coro/backends/asr/onnx_parakeet_prompt.py``
both drive artifact-directory contracts (fp32 + optionally-quantized ONNX
graphs, extracted checkpoint state) that some script has to produce first.
Before this package existed, those scripts lived ad hoc under ``.tmp/``
(gitignored, several with paths hardcoded into a *different*, no-longer-
guaranteed-to-exist worktree) -- reproducing a backend's artifacts meant
archaeology through journal entries, not running a command. This package is
that gap closed: every recipe referenced by a shipped backend's artifact
contract lives here, versioned, with parameterized paths.

Only *accepted* techniques live here -- a recipe module existing is itself a
claim "this is what the backend referenced by its docstring actually
expects." Rejected experiments (e.g. Canary's static-QDQ decoder INT8,
real word-level WER damage -- see
``.journals/2026-09-04/2026-09-04_canary-decode-loop-rtf_decoder-quant-static-qdq-rejected-plus-research/``)
stay in ``.tmp/`` as research artifacts, not promoted here. Extend this rule
forward: a recipe only moves into this package once the backend it feeds
has actually adopted its output as a real, wired selector -- do not migrate
speculatively ahead of that.

The rule cuts both ways, and ``canary_encoder_static_qdq`` is the worked
example: Canary's static-QDQ *encoder* INT8 was rejected once (+34.5%
relative cpWER at full 48-window mTEDx scale) and correctly left in ``.tmp/``,
then redone with percentile calibration and a measured ``nodes_to_exclude``
list, re-validated on the same gate, and only *then* promoted. A rejection
here is a statement about a technique, not about a target.

Recipes (each its own subpackage -- see the layout note below):

- ``canary_split_decoder``: graph-surgery split of the fused Canary decoder
  into ``xattn_kv.onnx`` + ``decoder_step.onnx`` (issue #64's core RTF fix).
- ``canary_decoder_dynamic_quantization``: dynamic INT8 quantization of
  ``decoder_step.onnx`` (this session's accepted decoder-quantization
  result).
- ``canary_encoder_static_qdq``: static-QDQ INT8 quantization of the Canary
  encoder (percentile calibration + a measured per-node exclusion list; the
  artifact ``onnx-canary-split``'s ``quantization`` selector loads).
- ``calibration_corpus``: multi-language acoustic calibration corpus builder
  (FLEURS), used by both static-QDQ encoder quantization recipes.
- ``parakeet_prompt_kernel_cache``: extracts ``model.prompt_kernel``'s
  weights + prompt/vocab metadata from the original NeMo checkpoint (numpy
  arrays only -- the checkpoint itself is never redistributed, see the
  License section below).
- ``parakeet_prompt_encoder_static_qdq``: static-QDQ INT8 quantization of
  the Parakeet-Prompt encoder (the current-best artifact
  ``onnx-parakeet-prompt`` actually loads).

Layout: each recipe above is its own subpackage
(``coro/recipes/<name>/``), not a flat module, so the narrative that
belongs with it -- why this technique, what bug its parameters avoid
re-introducing, what the numbers were -- has an obvious home next to the
code instead of competing for space in a module docstring:

- ``__init__.py``: the recipe's code (``main()`` plus any helpers), with a
  short docstring (what ``--help`` shows).
- ``__main__.py``: a thin ``from coro.recipes.<name> import main`` so
  ``python -m coro.recipes.<name>`` keeps working exactly as if it were
  still a flat module.
- ``README.md``: the full write-up -- what it does, what's worth knowing
  (diagnosed bugs its parameters avoid, why a technique was chosen over an
  alternative, measured numbers), usage, and consumers. Read this before
  a module's source when you need context, not just the code.

Conventions:

- Every recipe is a standalone, argparse-driven script with a ``main()``
  entry point (``uv run --extra recipes --extra cpu -m coro.recipes.<name>``
  -- see each recipe's own README.md for the exact invocation and required
  arguments). None are wired into ``coro``'s FastAPI server or CLI: these
  are export-time tooling, not serving-time code.
- Generated artifacts (ONNX graphs, calibration corpora, extracted weights)
  default to writing under :data:`coro.recipes.paths.RECIPE_ARTIFACTS_DIR`
  (``recipe-artifacts/`` at the repo root) -- never ``.tmp/`` (this package
  exists precisely so these artifacts are not one-off scratch output) and
  never committed to git (hundreds of MB to several GB; see ``.gitignore``).
  A recipe that needs another recipe's output takes it as a CLI argument
  with that convention as the default, never a hardcoded absolute path.

License:
    ``parakeet_prompt_kernel_cache`` and ``parakeet_prompt_encoder_static_qdq``
    both operate on ``parakeet-rnnt-1.1b-multilingual-prompt`` (NVIDIA
    Community Model License, NIM-gated for production use -- see
    ``coro/backends/asr/nemo.py``'s module docstring). Neither recipe
    downloads or redistributes the checkpoint itself; both require a local
    ``.nemo`` path the caller has already acquired. Their *outputs* (a small
    numpy MLP's weights, and a derivative quantized ONNX encoder) inherit
    the same redistribution restriction as the source checkpoint -- do not
    upload/publish them, same policy ``onnx_parakeet_prompt.py`` documents
    for the backend that loads them.
"""

from __future__ import annotations
