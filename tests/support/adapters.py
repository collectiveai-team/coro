"""In-memory adapters of coro's Core Boundary ports, for pipeline and app tests."""

from __future__ import annotations

from coro.core.models import SpeakerSegment, TranscriptToken


class RecordingASR:
    """An ASR Adapter returning fixed tokens and keeping what each window asked for.

    With ``error`` set, every window raises it instead, as a failing model does.
    """

    honours_prompt = True

    def __init__(
        self, tokens: list[TranscriptToken] | None = None, *, error: Exception | None = None
    ) -> None:
        self.tokens = list(tokens or [])
        self.error = error
        self.window_sizes: list[int] = []
        self.languages: list[str | None] = []
        self.prompts: list[str | None] = []

    async def transcribe_pcm(
        self, pcm: bytes, *, language: str | None = None, prompt: str | None = None
    ) -> list[TranscriptToken]:
        """Keep the window size, language and prompt, then answer the fixed tokens."""
        self.window_sizes.append(len(pcm))
        self.languages.append(language)
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return list(self.tokens)


class FixedDiarization:
    """A Diarization Adapter that answers the same speaker timeline for any PCM."""

    def __init__(self, timeline: list[SpeakerSegment] | None = None) -> None:
        self.timeline = list(timeline or [])
        self.diarized_bytes: list[int] = []

    async def diarize_pcm(self, pcm: bytes) -> list[SpeakerSegment]:
        """Keep the PCM size and return the fixed timeline."""
        self.diarized_bytes.append(len(pcm))
        return list(self.timeline)


class RecordingStreamingDiarizer:
    """A Streaming Diarizer that keeps the size of every chunk it ingests."""

    def __init__(self, timeline: list[SpeakerSegment]) -> None:
        self.timeline = timeline
        self.chunk_sizes: list[int] = []

    def ingest_pcm_chunk(self, pcm: bytes) -> None:
        """Keep the chunk size."""
        self.chunk_sizes.append(len(pcm))

    def finalize(self) -> list[SpeakerSegment]:
        """Return the fixed timeline."""
        return list(self.timeline)


class RecordingStreamingDiarizerFactory:
    """A Streaming Diarizer Factory that keeps every diarizer it hands out."""

    def __init__(self, timeline: list[SpeakerSegment] | None = None) -> None:
        self.timeline = list(timeline or [])
        self.diarizers: list[RecordingStreamingDiarizer] = []

    def __call__(self) -> RecordingStreamingDiarizer:
        """Hand out a fresh diarizer for one request."""
        diarizer = RecordingStreamingDiarizer(self.timeline)
        self.diarizers.append(diarizer)
        return diarizer
