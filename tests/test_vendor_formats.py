"""Vendor-shaped response formats carry per-word speaker labels.

The point of these formats is the one thing no OpenAI-compatible format can
express: a speaker on every word. These tests pin that, the unit conversions,
and the mapping of the ``-1`` unknown sentinel onto each vendor's native
"no speaker" representation.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from coro.api.schemas import WhisperWord
from coro.api.vendor.utterances import group_words_into_utterances, mean_confidence
from coro.core.models import (
    ResponseSegment,
    TranscriptionResult,
    TranscriptItem,
    TranscriptWord,
)
from conftest import make_app, make_wav

# (word, start, end, score, speaker) — two speakers plus one unsupported word.
_WORDS = [
    ("hola", 0.0, 0.5, 0.91, "1"),
    ("mundo", 0.5, 1.0, 0.83, "1"),
    ("si", 1.2, 1.6, 0.77, "2"),
    ("claro", 1.6, 2.0, 0.66, "-1"),
]


def _transcript_words() -> list[TranscriptWord]:
    return [
        TranscriptWord(word=word, start=start, end=end, score=score, speaker=speaker)
        for word, start, end, score, speaker in _WORDS
    ]


class _FakePipeline:
    async def transcribe(self, audio, *, language=None, prompt=None):
        words = _transcript_words()
        return TranscriptionResult(
            segments=[
                ResponseSegment(
                    start=0.0, end=1.0, text="hola mundo", speaker="1", words=words[:2]
                ),
                ResponseSegment(start=1.2, end=1.6, text="si", speaker="2", words=words[2:3]),
                ResponseSegment(start=1.6, end=2.0, text="claro", speaker="-1", words=words[3:]),
            ],
            word_segments=words,
            transcript=[
                TranscriptItem(start=0.0, end=1.0, text="hola mundo"),
                TranscriptItem(start=1.2, end=1.6, text="si"),
                TranscriptItem(start=1.6, end=2.0, text="claro"),
            ],
            diarization=[],
            raw_words=[],
        )


async def _transcribe(fmt: str, *, filename: str = "meeting.wav") -> Any:
    app = make_app(_FakePipeline())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": (filename, make_wav(), "audio/wav")},
            data={"response_format": fmt},
        )
    assert response.status_code == 200, response.text
    return response.json()


# MARK: Utterance grouping
def _word(text: str, speaker: str, score: float = 1.0) -> WhisperWord:
    return WhisperWord(word=text, start=0.0, end=1.0, score=score, speaker=speaker)


class TestUtteranceGrouping:
    def test_consecutive_same_speaker_words_form_one_utterance(self):
        words = [_word("a", "1"), _word("b", "1"), _word("c", "1")]
        utterances = group_words_into_utterances(words)
        assert len(utterances) == 1
        assert utterances[0].speaker == "1"
        assert utterances[0].text == "a b c"

    def test_speaker_change_starts_a_new_utterance(self):
        words = [_word("a", "1"), _word("b", "2"), _word("c", "1")]
        utterances = group_words_into_utterances(words)
        assert [u.speaker for u in utterances] == ["1", "2", "1"]

    def test_recurring_speaker_is_not_merged_across_a_turn(self):
        # A speaker returning later is a separate turn, not the same utterance.
        words = [_word("a", "1"), _word("b", "2"), _word("c", "1")]
        assert len(group_words_into_utterances(words)) == 3

    def test_empty_input_yields_no_utterances(self):
        assert group_words_into_utterances([]) == []

    def test_mean_confidence_of_no_words_is_zero(self):
        assert mean_confidence([]) == 0.0

    def test_utterance_confidence_is_the_mean_of_its_words(self):
        words = [_word("a", "1", 0.5), _word("b", "1", 1.0)]
        assert group_words_into_utterances(words)[0].confidence == pytest.approx(0.75)


# MARK: AssemblyAI shape
@pytest.mark.asyncio
class TestAssemblyAIFormat:
    async def test_every_word_carries_a_speaker(self):
        body = await _transcribe("assemblyai_json")
        assert [w["speaker"] for w in body["words"]] == ["1", "1", "2", None]

    async def test_words_are_nested_inside_utterances_with_speakers(self):
        body = await _transcribe("assemblyai_json")
        assert [u["speaker"] for u in body["utterances"]] == ["1", "2", None]
        assert [w["speaker"] for w in body["utterances"][0]["words"]] == ["1", "1"]

    async def test_timestamps_are_integer_milliseconds(self):
        body = await _transcribe("assemblyai_json")
        first = body["words"][0]
        assert (first["start"], first["end"]) == (0, 500)
        assert all(isinstance(w["start"], int) for w in body["words"])

    async def test_confidence_is_the_real_per_word_score(self):
        body = await _transcribe("assemblyai_json")
        assert [w["confidence"] for w in body["words"]] == [0.91, 0.83, 0.77, 0.66]

    async def test_unknown_speaker_becomes_null_not_the_sentinel_string(self):
        body = await _transcribe("assemblyai_json")
        assert body["words"][-1]["speaker"] is None
        assert "-1" not in {w["speaker"] for w in body["words"]}

    async def test_status_is_completed_and_id_is_the_request_id(self):
        body = await _transcribe("assemblyai_json")
        assert body["status"] == "completed"
        assert body["id"]

    async def test_audio_url_carries_the_upload_filename(self):
        body = await _transcribe("assemblyai_json", filename="hearing-42.wav")
        assert body["audio_url"] == "hearing-42.wav"


# MARK: Deepgram shape
@pytest.mark.asyncio
class TestDeepgramFormat:
    async def test_channel_words_carry_speakers(self):
        body = await _transcribe("deepgram_json")
        words = body["results"]["channels"][0]["alternatives"][0]["words"]
        assert [w["speaker"] for w in words] == [1, 1, 2, None]

    async def test_utterances_carry_speakers_and_their_words(self):
        body = await _transcribe("deepgram_json")
        utterances = body["results"]["utterances"]
        assert [u["speaker"] for u in utterances] == [1, 2, None]
        assert [w["speaker"] for w in utterances[0]["words"]] == [1, 1]

    async def test_timestamps_are_float_seconds(self):
        body = await _transcribe("deepgram_json")
        first = body["results"]["channels"][0]["alternatives"][0]["words"][0]
        assert (first["start"], first["end"]) == (0.0, 0.5)

    async def test_speaker_confidence_is_omitted(self):
        # Explicit decision (ADR 0010): the diarizer's per-frame posteriors are
        # binarized before the Core Boundary, so the field is not derivable.
        body = await _transcribe("deepgram_json")
        words = body["results"]["channels"][0]["alternatives"][0]["words"]
        assert all("speaker_confidence" not in word for word in words)

    async def test_speaker_numbering_is_passed_through_not_renumbered(self):
        # Labels stay comparable with diarized_json and with the scoring STM.
        body = await _transcribe("deepgram_json")
        words = body["results"]["channels"][0]["alternatives"][0]["words"]
        assert {w["speaker"] for w in words if w["speaker"] is not None} == {1, 2}

    async def test_metadata_digest_matches_the_uploaded_bytes(self):
        import hashlib

        body = await _transcribe("deepgram_json")
        assert body["metadata"]["sha256"] == hashlib.sha256(make_wav()).hexdigest()

    async def test_metadata_reports_the_configured_model(self):
        body = await _transcribe("deepgram_json")
        assert body["metadata"]["models"] == ["openai/whisper-medium"]
        assert body["metadata"]["channels"] == 1
