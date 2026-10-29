"""An in-memory NeMo Prompt ASR checkpoint: the ``NemoTranscriber`` the ``nemo`` adapter calls.

``FakeNemoPromptModel.transcribe`` keeps the NeMo behaviours the adapter works
around, as verified against ``EncDecHybridRNNTCTCBPEModelWithPrompt`` (see the
module docstring of ``coro/backends/asr/nemo.py``):

- a bare ``target_lang`` keyword is ignored; the language reaches the model
  only through ``override_config`` with ``use_lhotse=False``;
- the forced-language path crashes unless ``timestamps=False`` is passed at
  the top level and in ``override_config``, and a ``DataLoader`` with workers
  crashes under long checkout paths;
- the forced-language path returns raw frame indices as ``hyp.timestamp``,
  and the default path returns seconds;
- without ``return_hypotheses`` it returns plain strings.

Its typed ``override_config`` parameter is what ``resolve_override_config_type``
introspects, as on the real checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import soundfile as sf

PIECES = {11: "▁hola", 12: ",", 13: "▁mundo", 14: ".", 21: "▁hello", 22: "▁world"}


class FakeSubwordTokenizer:
    """A ``SubwordTokenizer`` over the SentencePiece pieces of ``PIECES``."""

    def ids_to_tokens(self, ids: list[int]) -> list[str]:
        """Return the piece of each id."""
        return [PIECES[i] for i in ids]


@dataclass(frozen=True)
class Utterance:
    """What the model hears in one language: ids, per-token times and text."""

    ids: tuple[int, ...]
    seconds: tuple[float, ...]
    frames: tuple[int, ...]
    text: str


SPANISH = Utterance((11, 12, 13, 14), (0.1, 0.2, 0.5, 0.6), (7, 9, 10, 11), "hola, mundo.")
ENGLISH = Utterance((21, 22), (0.1, 0.5), (7, 10), "hello world")
SILENCE = Utterance((), (), (), "")


@dataclass
class FakeTranscribeConfig:
    """The fields of NeMo's (base and Prompt subclass) ``TranscribeConfig``."""

    use_lhotse: bool = True
    batch_size: int = 4
    return_hypotheses: bool = False
    num_workers: int | None = None
    timestamps: bool | None = None
    verbose: bool = True
    target_lang: str = "en-US"


@dataclass(frozen=True)
class HeardAudio:
    """The WAV file the model read for one call."""

    samplerate: int
    frames: int


@dataclass
class FakeNemoPromptModel:
    """A Prompt checkpoint that says ``utterances[language]``; unforced, ``default_language``."""

    utterances: dict[str, Utterance] = field(
        default_factory=lambda: {"es-US": SPANISH, "es-ES": SPANISH, "en-US": ENGLISH}
    )
    default_language: str = "es-US"
    emits_timestamps: bool = True
    tokenizer: FakeSubwordTokenizer = field(default_factory=FakeSubwordTokenizer)
    cfg: object = field(default_factory=dict)
    encoder: object | None = None
    heard: list[HeardAudio] = field(default_factory=list)

    def eval(self) -> FakeNemoPromptModel:
        """Return the model, as ``nn.Module.eval`` does."""
        return self

    def to(self, device: str) -> FakeNemoPromptModel:
        """Return the model, as ``nn.Module.to`` does."""
        return self

    def transcribe(
        self,
        paths: list[str],
        *,
        override_config: FakeTranscribeConfig | None = None,
        timestamps: bool | None = None,
        return_hypotheses: bool = False,
        **_kwargs: object,
    ) -> list[object]:
        """Transcribe the first WAV of ``paths`` with NeMo's quirks (see the module docstring)."""
        info = sf.info(paths[0])
        self.heard.append(HeardAudio(int(info.samplerate), int(info.frames)))
        if override_config is None:
            utterance = self.utterances[self.default_language]
            times = utterance.seconds if timestamps and self.emits_timestamps else None
            wants_hypotheses = return_hypotheses
        else:
            if override_config.num_workers != 0:
                raise OSError("AF_UNIX path too long")
            if timestamps is not False or override_config.timestamps is not False:
                raise RuntimeError("Tensor.__contains__ only supports Tensor or scalar")
            language = self.default_language
            if not override_config.use_lhotse:
                language = override_config.target_lang
            utterance = self.utterances[language]
            times = utterance.frames
            wants_hypotheses = override_config.return_hypotheses
        if not wants_hypotheses:
            return [utterance.text]
        return [
            SimpleNamespace(
                y_sequence=utterance.ids,
                timestamp=list(times) if times else None,
                text=utterance.text,
            )
        ]
