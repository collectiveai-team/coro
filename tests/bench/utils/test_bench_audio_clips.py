"""Tests for shared benchmark audio-clip helpers."""

from __future__ import annotations

from pathlib import Path
import wave

import pytest
from support.corpus import write_silent_wav

from coro.bench.utils.audio_clips import concat_wav_clips


def _read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as handle:
        return handle.readframes(handle.getnframes())


class TestConcatWavClips:
    def test_concatenates_in_order_with_silence_gaps(self, tmp_path: Path):
        clip_a = tmp_path / "a.wav"
        clip_b = tmp_path / "b.wav"
        write_silent_wav(clip_a, seconds=1.0)
        write_silent_wav(clip_b, seconds=2.0)
        dst = tmp_path / "out.wav"

        spans = concat_wav_clips([clip_a, clip_b], dst, gap_seconds=0.5)

        assert spans == [(0.0, 1.0), (1.5, 3.5)]
        with wave.open(str(dst), "rb") as handle:
            assert handle.getnframes() == int(16000 * 3.5)
            assert handle.getframerate() == 16000
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2

    def test_no_gap_before_first_or_after_last_clip(self, tmp_path: Path):
        clip = tmp_path / "solo.wav"
        write_silent_wav(clip, seconds=1.0)
        dst = tmp_path / "out.wav"

        spans = concat_wav_clips([clip], dst, gap_seconds=5.0)

        assert spans == [(0.0, 1.0)]

    def test_three_clips_accumulate_gaps_between_each_pair(self, tmp_path: Path):
        clips = [tmp_path / f"{i}.wav" for i in range(3)]
        for clip in clips:
            write_silent_wav(clip, seconds=1.0)
        dst = tmp_path / "out.wav"

        spans = concat_wav_clips(clips, dst, gap_seconds=1.0)

        assert spans == [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]

    def test_rejects_a_clip_that_is_not_16khz_mono_pcm16(self, tmp_path: Path):
        bad = tmp_path / "bad.wav"
        with wave.open(str(bad), "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x00\x00\x00" * 100)
        dst = tmp_path / "out.wav"

        with pytest.raises(ValueError, match="16 kHz mono PCM16"):
            concat_wav_clips([bad], dst)

    def test_output_pcm_bytes_preserve_each_clip_verbatim(self, tmp_path: Path):
        clip_a = tmp_path / "a.wav"
        clip_b = tmp_path / "b.wav"
        write_silent_wav(clip_a, seconds=0.001)
        write_silent_wav(clip_b, seconds=0.001)
        dst = tmp_path / "out.wav"

        concat_wav_clips([clip_a, clip_b], dst, gap_seconds=0.0)

        assert _read_pcm(dst) == _read_pcm(clip_a) + _read_pcm(clip_b)
