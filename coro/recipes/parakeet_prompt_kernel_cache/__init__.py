r"""Extract parakeet-rnnt-1.1b-multilingual-prompt's ``prompt_kernel`` weights.

See ``README.md`` in this package for the full write-up (why this exists,
the license constraint on its output, and the checkpoint-loading cost it
avoids paying twice). This docstring stays short on purpose -- it is what
``--help`` shows.

Usage:
    uv run --extra recipes --extra cpu -m coro.recipes.parakeet_prompt_kernel_cache \\
        --checkpoint /path/to/parakeet-rnnt-1.1b-multilingual-prompt.nemo
    uv run --extra recipes --extra cpu -m coro.recipes.parakeet_prompt_kernel_cache \\
        --checkpoint /path/to/checkpoint.nemo --out-dir recipe-artifacts/parakeet_prompt
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from coro.cli_support import invoke, new_app, out_console
from coro.core.logger import get_logger
from coro.recipes.paths import RECIPE_ARTIFACTS_DIR

log = get_logger(__name__)


app = new_app("parakeet_prompt_kernel_cache", help=__doc__.splitlines()[0])


@app.default
def _extract(
    *,
    checkpoint: Path,
    out_dir: Path = RECIPE_ARTIFACTS_DIR / "parakeet_prompt",
) -> None:
    """Extract the prompt_kernel weights and metadata into a cache.

    Args:
        checkpoint: Path to a local parakeet-rnnt-1.1b-multilingual-prompt.nemo checkpoint.
        out_dir: Directory to write prompt_kernel_cache.npz/.json into.

    """
    import nemo.collections.asr as nemo_asr
    import torch

    log.info("loading checkpoint", checkpoint=str(checkpoint))
    t0 = time.time()
    model = nemo_asr.models.ASRModel.restore_from(
        str(checkpoint), map_location=torch.device("cpu")
    ).eval()
    log.info("loaded checkpoint", seconds=round(time.time() - t0, 1))

    tokenizer = model.tokenizer
    vocab_size = tokenizer.vocab_size
    blank_id = model.joint.num_classes_with_blank - 1
    prompt_dictionary = dict(model.cfg.model_defaults.get("prompt_dictionary") or {})
    weights = {
        name: param.detach().numpy() for name, param in model.prompt_kernel.state_dict().items()
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_npz = out_dir / "prompt_kernel_cache.npz"
    cache_json = out_dir / "prompt_kernel_cache.json"
    np.savez(cache_npz, **weights)
    cache_json.write_text(
        json.dumps(
            {
                "vocab_size": vocab_size,
                "blank_id": blank_id,
                "prompt_dictionary": prompt_dictionary,
            },
            indent=2,
        )
    )
    out_console.out(f"cached weights -> {cache_npz}")
    out_console.out(f"cached meta -> {cache_json}")
    out_console.out("prompt_dictionary:", prompt_dictionary)
    out_console.out("blank_id:", blank_id, "vocab_size:", vocab_size)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the prompt kernel cache recipe."""
    invoke(app, argv)
