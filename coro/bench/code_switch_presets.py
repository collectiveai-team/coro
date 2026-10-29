"""Registry of the source corpora and splice presets of the code-switch corpus.

:mod:`coro.bench.code_switch` materialises a preset; this module only names
them, the way ``bench.spanish`` names its corpora.
"""

from __future__ import annotations

from coro.bench.models.code_switch import CodeSwitchCorpus, CodeSwitchPreset

CODE_SWITCH_CORPORA: dict[str, CodeSwitchCorpus] = {
    "fleurs_es": CodeSwitchCorpus(
        key="fleurs_es",
        name="FLEURS (es_419, test split)",
        hf_dataset="google/fleurs",
        hf_config="es_419",
        hf_split="test",
        licence="CC-BY-4.0",
        licence_url="https://creativecommons.org/licenses/by/4.0/",
        homepage="https://huggingface.co/datasets/google/fleurs",
        language="es",
        id_column="id",
        text_column="raw_transcription",
        notes="Read Spanish speech; the carrier-language source for the code-switch corpus.",
    ),
    "fleurs_en": CodeSwitchCorpus(
        key="fleurs_en",
        name="FLEURS (en_us, test split)",
        hf_dataset="google/fleurs",
        hf_config="en_us",
        hf_split="test",
        licence="CC-BY-4.0",
        licence_url="https://creativecommons.org/licenses/by/4.0/",
        homepage="https://huggingface.co/datasets/google/fleurs",
        language="en",
        id_column="id",
        text_column="raw_transcription",
        notes="Read English speech; the switch-language source for the code-switch corpus.",
    ),
}

CODE_SWITCH_PRESETS: dict[str, CodeSwitchPreset] = {
    "sparse-splice": CodeSwitchPreset(
        key="sparse-splice",
        pattern="sparse",
        carrier_corpus="fleurs_es",
        switch_corpus="fleurs_en",
        carrier_clips_per_item=4,
        switch_clips_per_item=1,
        items=20,
        gap_seconds=1.0,
        description=(
            "One short English clip inserted once, evenly amid four Spanish "
            "carrier clips per item -- simulates an occasional embedded "
            "foreign term/phrase, not a full language change."
        ),
    ),
    "paragraph-level": CodeSwitchPreset(
        key="paragraph-level",
        pattern="paragraph",
        carrier_corpus="fleurs_es",
        switch_corpus="fleurs_en",
        carrier_clips_per_item=2,
        switch_clips_per_item=2,
        items=20,
        gap_seconds=1.0,
        description=(
            "Whole Spanish/English clips alternated one-for-one -- "
            "simulates a full topic/speaker-language change."
        ),
    ),
}


def resolve_code_switch_preset(preset_key: str) -> CodeSwitchPreset:
    """Return the named code-switch preset, or fail with the known preset names."""
    try:
        return CODE_SWITCH_PRESETS[preset_key]
    except KeyError:
        known = ", ".join(sorted(CODE_SWITCH_PRESETS))
        raise ValueError(
            f"Unknown code-switch preset {preset_key!r}; known presets: {known}"
        ) from None
