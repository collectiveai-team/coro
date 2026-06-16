"""Tests for coro.bench.stm and vendored warmup audio.

Issue 01: Vendor JFK warmup audio and extract STM library module.
"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest


class TestWarmupAudioAsset:
    """Vendored JFK WAV is loadable as 16 kHz mono."""

    def test_jfk_wav_exists(self):
        from coro.bench.data import WARMUP_AUDIO_PATH

        assert WARMUP_AUDIO_PATH.exists(), f"Warmup audio not found at {WARMUP_AUDIO_PATH}"

    def test_jfk_wav_is_valid_16khz_mono(self):
        from coro.bench.data import WARMUP_AUDIO_PATH

        with wave.open(str(WARMUP_AUDIO_PATH), "rb") as wf:
            assert wf.getnchannels() == 1, "Expected mono WAV"
            assert wf.getframerate() == 16000, "Expected 16 kHz sample rate"
            assert wf.getsampwidth() == 2, "Expected 16-bit (2 bytes per sample)"
            assert wf.getnframes() > 0, "Expected non-zero duration"

    def test_jfk_wav_duration_positive(self):
        from coro.bench.data import WARMUP_AUDIO_PATH

        with wave.open(str(WARMUP_AUDIO_PATH), "rb") as wf:
            duration = wf.getnframes() / wf.getframerate()
            assert duration > 0, "Duration must be positive"


class TestHypSegmentsToStm:
    """hyp_segments_to_stm converts diarized_json segments to STM text."""

    def test_basic_segments_produce_stm_lines(self):
        from coro.bench.stm import hyp_segments_to_stm

        segments = [
            {"start": 0.0, "end": 1.5, "text": "hello world", "speaker": "A"},
            {"start": 1.5, "end": 3.0, "text": "goodbye", "speaker": "B"},
        ]
        result = hyp_segments_to_stm(segments, "meeting001")
        lines = result.strip().split("\n")
        assert len(lines) == 2
        assert lines[0] == "meeting001 1 A 0.000 1.500 hello world"
        assert lines[1] == "meeting001 1 B 1.500 3.000 goodbye"

    def test_speaker_labels_passed_through_unchanged(self):
        from coro.bench.stm import hyp_segments_to_stm

        segments = [
            {"start": 0.0, "end": 1.0, "text": "hi", "speaker": "Speaker_0"},
        ]
        result = hyp_segments_to_stm(segments, "rec01")
        assert "Speaker_0" in result
        assert "SPEAKER_Speaker_0" not in result

    def test_empty_text_segments_skipped(self):
        from coro.bench.stm import hyp_segments_to_stm

        segments = [
            {"start": 0.0, "end": 1.0, "text": "", "speaker": "A"},
            {"start": 1.0, "end": 2.0, "text": "actual content", "speaker": "A"},
        ]
        result = hyp_segments_to_stm(segments, "rec01")
        lines = result.strip().split("\n")
        assert len(lines) == 1

    def test_zero_duration_segments_skipped(self):
        from coro.bench.stm import hyp_segments_to_stm

        segments = [
            {"start": 1.0, "end": 1.0, "text": "same time", "speaker": "A"},
            {"start": 2.0, "end": 1.0, "text": "inverted", "speaker": "A"},
            {"start": 0.0, "end": 1.0, "text": "valid", "speaker": "A"},
        ]
        result = hyp_segments_to_stm(segments, "rec01")
        lines = result.strip().split("\n")
        assert len(lines) == 1
        assert "valid" in lines[0]

    def test_segments_sorted_by_start_time_then_speaker(self):
        from coro.bench.stm import hyp_segments_to_stm

        segments = [
            {"start": 1.0, "end": 2.0, "text": "second a", "speaker": "B"},
            {"start": 0.0, "end": 1.0, "text": "first", "speaker": "A"},
            {"start": 1.0, "end": 2.0, "text": "second b", "speaker": "A"},
        ]
        result = hyp_segments_to_stm(segments, "rec01")
        lines = result.strip().split("\n")
        assert len(lines) == 3
        assert "first" in lines[0]
        assert "second b" in lines[1]
        assert "second a" in lines[2]

    def test_missing_start_or_end_skipped(self):
        from coro.bench.stm import hyp_segments_to_stm

        segments = [
            {"end": 1.0, "text": "no start", "speaker": "A"},
            {"start": 0.0, "text": "no end", "speaker": "A"},
            {"start": 0.0, "end": 1.0, "text": "valid", "speaker": "A"},
        ]
        result = hyp_segments_to_stm(segments, "rec01")
        lines = result.strip().split("\n")
        assert len(lines) == 1

    def test_whitespace_cleaned(self):
        from coro.bench.stm import hyp_segments_to_stm

        segments = [
            {"start": 0.0, "end": 1.0, "text": "  hello   world  ", "speaker": "A"},
        ]
        result = hyp_segments_to_stm(segments, "rec01")
        assert "hello world" in result


class TestAmiMeetingToStm:
    """ami_meeting_to_stm produces Reference STM from AMI annotation tree."""

    @pytest.fixture()
    def ami_fixture(self, tmp_path: Path) -> Path:
        """Create a minimal AMI annotation tree for meeting TS3003a."""
        root = tmp_path / "amicorpus"
        words_dir = root / "TS3003a" / "words"
        segments_dir = root / "TS3003a" / "segments"
        words_dir.mkdir(parents=True)
        segments_dir.mkdir(parents=True)

        words_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<nite:root xmlns:nite="http://nite.sourceforge.net/">\n'
            '  <w nite:id="TS3003a.A.words0" starttime="0.0" endtime="0.5">Hello</w>\n'
            '  <w nite:id="TS3003a.A.words1" starttime="0.5" endtime="1.0">world</w>\n'
            '  <w nite:id="TS3003a.B.words0" starttime="1.0" endtime="1.5">Good</w>\n'
            '  <w nite:id="TS3003a.B.words1" starttime="1.5" endtime="2.0">morning</w>\n'
            "</nite:root>\n"
        )
        (words_dir / "TS3003a.A.words.xml").write_text(words_xml)
        (words_dir / "TS3003a.B.words.xml").write_text(
            words_xml.replace("TS3003a.A", "TS3003a.B")
        )

        seg_a_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<nite:root xmlns:nite="http://nite.sourceforge.net/">\n'
            '  <segment nite:id="s0">\n'
            '    <child href="TS3003a.A.words.xml#id(TS3003a.A.words0)"/>\n'
            '    <child href="TS3003a.A.words.xml#id(TS3003a.A.words1)"/>\n'
            '  </segment>\n'
            "</nite:root>\n"
        )
        (segments_dir / "TS3003a.A.segments.xml").write_text(seg_a_xml)

        seg_b_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<nite:root xmlns:nite="http://nite.sourceforge.net/">\n'
            '  <segment nite:id="s0">\n'
            '    <child href="TS3003a.B.words.xml#id(TS3003a.B.words0)"/>\n'
            '    <child href="TS3003a.B.words.xml#id(TS3003a.B.words1)"/>\n'
            '  </segment>\n'
            "</nite:root>\n"
        )
        (segments_dir / "TS3003a.B.segments.xml").write_text(seg_b_xml)
        return root

    def test_produces_stm_with_correct_speakers(self, ami_fixture: Path):
        from coro.bench.stm import ami_meeting_to_stm

        result = ami_meeting_to_stm(ami_fixture, "TS3003a")
        assert "A" in result
        assert "B" in result

    def test_stm_lines_have_correct_format(self, ami_fixture: Path):
        from coro.bench.stm import ami_meeting_to_stm

        result = ami_meeting_to_stm(ami_fixture, "TS3003a")
        lines = result.strip().split("\n")
        for line in lines:
            parts = line.split()
            assert parts[0] == "TS3003a", f"Recording ID mismatch: {parts[0]}"
            assert parts[1] == "1", f"Channel mismatch: {parts[1]}"
            assert parts[2] in ("A", "B"), f"Speaker mismatch: {parts[2]}"

    def test_stm_lines_sorted_by_time(self, ami_fixture: Path):
        from coro.bench.stm import ami_meeting_to_stm

        result = ami_meeting_to_stm(ami_fixture, "TS3003a")
        lines = result.strip().split("\n")
        times = [float(line.split()[3]) for line in lines]
        assert times == sorted(times)
