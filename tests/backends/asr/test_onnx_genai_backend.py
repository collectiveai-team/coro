"""onnx-genai (nemotron) backend pure-logic tests.

The streaming loop requires the onnxruntime-genai runtime + model and is covered by the
benchmark harness; here we test the language mapping and language-tag stripping, plus the
reuse of the onnx-asr word reconstruction for GenAI-style decoded text pieces.
"""

from __future__ import annotations

from types import SimpleNamespace

from coro.backends.asr.onnx_asr import convert_onnx_asr_result
from coro.backends.asr.onnx_genai import _LANG_TAG_RE, _lang_id_for


def test_lang_id_known_codes():
    """Known language/locale codes map to the model's lang_id."""
    assert _lang_id_for("en") == 0
    assert _lang_id_for("en-GB") == 1
    assert _lang_id_for("es") == 3
    assert _lang_id_for("auto") == 101


def test_lang_id_falls_back_to_base_then_default():
    """Unknown locale falls back to base code, then to English default."""
    assert _lang_id_for("es-AR") == 3  # base 'es'
    assert _lang_id_for("xx") == 0  # unknown -> default English
    assert _lang_id_for(None) == 0


def test_language_tag_stripping():
    """Inline language-tag tokens are removed from decoded text."""
    assert _LANG_TAG_RE.sub("", "hello <en-US> world") == "hello  world"
    assert _LANG_TAG_RE.sub("", "<es> hola") == " hola"
    assert _LANG_TAG_RE.sub("", "no tags here") == "no tags here"


def test_genai_pieces_reconstruct_words():
    """GenAI decoded pieces (space-prefixed) reconstruct into word tokens."""
    result = SimpleNamespace(
        tokens=[" hello", " world", "."],
        timestamps=[0.0, 0.56, 1.12],
        logprobs=None,
    )
    tokens = convert_onnx_asr_result(result)
    assert [t.text for t in tokens] == [" hello", " world."]
