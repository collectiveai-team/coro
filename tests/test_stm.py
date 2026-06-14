"""Tests for asr_diar_server.bench.stm and vendored warmup audio.

Issue 01: Vendor JFK warmup audio and extract STM library module.
"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest


class TestWarmupAudioAsset:
    """Vendored JFK WAV is loadable as 16 kHz mono."""

    def test_jfk_wav_exists(self):
        from asr_diar_server.bench.data import WARMUP_AUDIO_PATH

        assert WARMUP_AUDIO_PATH.exists(), f"Warmup audio not found at {WARMUP_AUDIO_PATH}"

    def test_jfk_wav_is_valid_16khz_mono(self):
        from asr_diar_server.bench.data import WARMUP_AUDIO_PATH

        with wave.open(str(WARMUP_AUDIO_PATH), "rb") as wf:
            assert wf.getnchannels() == 1, "Expected mono WAV"
            assert wf.getframerate() == 16000, "Expected 16 kHz sample rate"
            assert wf.getsampwidth() == 2, "Expected 16-bit (2 bytes per sample)"
            assert wf.getnframes() > 0, "Expected non-zero duration"

    def test_jfk_wav_duration_positive(self):
        from asr_diar_server.bench.data import WARMUP_AUDIO_PATH

        with wave.open(str(WARMUP_AUDIO_PATH), "rb") as wf:
            duration = wf.getnframes() / wf.getframerate()
            assert duration > 0, "Duration must be positive"


class TestHypSegmentsToStm:
    """hyp_segments_to_stm converts diarized_json segments to STM text."""

    def test_basic_segments_produce_stm_lines(self):
        from asr_diar_server.bench.stm import hyp_segments_to_stm

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
        from asr_diar_server.bench.stm import hyp_segments_to_stm

        segments = [
            {"start": 0.0, "end": 1.0, "text": "hi", "speaker": "Speaker_0"},
        ]
        result = hyp_segments_to_stm(segments, "rec01")
        assert "Speaker_0" in result
        assert "SPEAKER_Speaker_0" not in result

    def test_empty_text_segments_skipped(self):
        from asr_diar_server.bench.stm import hyp_segments_to_stm

        segments = [
            {"start": 0.0, "end": 1.0, "text": "", "speaker": "A"},
            {"start": 1.0, "end": 2.0, "text": "actual content", "speaker": "A"},
        ]
        result = hyp_segments_to_stm(segments, "rec01")
        lines = result.strip().split("\n")
        assert len(lines) == 1

    def test_zero_duration_segments_skipped(self):
        from asr_diar_server.bench.stm import hyp_segments_to_stm

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
        from asr_diar_server.bench.stm import hyp_segments_to_stm

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
        from asr_diar_server.bench.stm import hyp_segments_to_stm

        segments = [
            {"end": 1.0, "text": "no start", "speaker": "A"},
            {"start": 0.0, "text": "no end", "speaker": "A"},
            {"start": 0.0, "end": 1.0, "text": "valid", "speaker": "A"},
        ]
        result = hyp_segments_to_stm(segments, "rec01")
        lines = result.strip().split("\n")
        assert len(lines) == 1

    def test_whitespace_cleaned(self):
        from asr_diar_server.bench.stm import hyp_segments_to_stm

        segments = [
            {"start": 0.0, "end": 1.0, "text": "  hello   world  ", "speaker": "A"},
        ]
        result = hyp_segments_to_stm(segments, "rec01")
        assert "hello world" in result


class TestRttmToStm:
    """rttm_to_stm converts RTTM SPEAKER turns to a diarization-only STM."""

    _RTTM = (
        "SPEAKER rec 1 2.50 1.50 <NA> <NA> spkB <NA> <NA>\n"
        "SPEAKER rec 1 0.00 2.00 <NA> <NA> spkA <NA> <NA>\n"
    )

    def test_speaker_turns_become_sorted_stm_with_sentinel_text(self):
        from asr_diar_server.bench.stm import DIARIZATION_ONLY_TEXT, rttm_to_stm

        lines = rttm_to_stm(self._RTTM, "rec").strip().split("\n")
        # Sorted by start time; end = onset + duration; sentinel text.
        assert lines[0] == f"rec 1 spkA 0.000 2.000 {DIARIZATION_ONLY_TEXT}"
        assert lines[1] == f"rec 1 spkB 2.500 4.000 {DIARIZATION_ONLY_TEXT}"

    def test_non_speaker_and_nonpositive_duration_rows_dropped(self):
        from asr_diar_server.bench.stm import rttm_to_stm

        rttm = (
            "SPKR-INFO rec 1 <NA> <NA> <NA> unknown spkA <NA> <NA>\n"
            "SPEAKER rec 1 1.00 0.00 <NA> <NA> spkA <NA> <NA>\n"
            "SPEAKER rec 1 1.00 0.50 <NA> <NA> spkA <NA> <NA>\n"
        )
        assert rttm_to_stm(rttm, "rec").strip().split("\n") == [
            "rec 1 spkA 1.000 1.500 <sd>"
        ]

    def test_empty_rttm_yields_empty_string(self):
        from asr_diar_server.bench.stm import rttm_to_stm

        assert rttm_to_stm("\n# comment\n", "rec") == ""


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
        from asr_diar_server.bench.stm import ami_meeting_to_stm

        result = ami_meeting_to_stm(ami_fixture, "TS3003a")
        assert "A" in result
        assert "B" in result

    def test_stm_lines_have_correct_format(self, ami_fixture: Path):
        from asr_diar_server.bench.stm import ami_meeting_to_stm

        result = ami_meeting_to_stm(ami_fixture, "TS3003a")
        lines = result.strip().split("\n")
        for line in lines:
            parts = line.split()
            assert parts[0] == "TS3003a", f"Recording ID mismatch: {parts[0]}"
            assert parts[1] == "1", f"Channel mismatch: {parts[1]}"
            assert parts[2] in ("A", "B"), f"Speaker mismatch: {parts[2]}"

    def test_stm_lines_sorted_by_time(self, ami_fixture: Path):
        from asr_diar_server.bench.stm import ami_meeting_to_stm

        result = ami_meeting_to_stm(ami_fixture, "TS3003a")
        lines = result.strip().split("\n")
        times = [float(line.split()[3]) for line in lines]
        assert times == sorted(times)

    def test_clip_reference_stm_windows_and_rebases(self, ami_fixture: Path):
        from asr_diar_server.bench.ami import clip_reference_stm

        # Full meeting: A 0.0-1.0 "Hello world", B 1.0-2.0 "Good morning".
        clip = clip_reference_stm(ami_fixture, "TS3003a", start=0.5, duration=1.0)
        lines = [line.split(maxsplit=5) for line in clip.splitlines()]

        assert lines[0][2] == "A"
        assert lines[0][3:5] == ["0.000", "0.500"]
        assert lines[0][5] == "Hello world"
        assert lines[1][2] == "B"
        assert lines[1][3:5] == ["0.500", "1.000"]
        assert lines[1][5] == "Good morning"


class TestSliceStmWindow:
    """slice_stm_window keeps overlapping lines, clamps, and rebases times."""

    SAMPLE = (
        "m 1 A 0.000 2.000 hello world\n"
        "m 1 B 2.000 4.000 foo bar\n"
        "m 1 A 4.000 6.000 baz qux\n"
        "m 1 B 6.000 8.000 out of window\n"
    )

    def test_keeps_only_overlapping_lines(self):
        from asr_diar_server.bench.stm import slice_stm_window

        out = slice_stm_window(self.SAMPLE, 2.0, 6.0)
        texts = [line.split(maxsplit=5)[5] for line in out.splitlines()]
        assert texts == ["foo bar", "baz qux"]

    def test_rebases_times_to_window_start(self):
        from asr_diar_server.bench.stm import slice_stm_window

        out = slice_stm_window(self.SAMPLE, 2.0, 6.0)
        first = out.splitlines()[0].split()
        # "foo bar" was 2.0-4.0; rebased to 0.0-2.0.
        assert first[3] == "0.000"
        assert first[4] == "2.000"

    def test_rebase_false_preserves_absolute_times(self):
        from asr_diar_server.bench.stm import slice_stm_window

        out = slice_stm_window(self.SAMPLE, 2.0, 6.0, rebase=False)
        first = out.splitlines()[0].split()
        assert first[3] == "2.000"

    def test_clamps_partial_overlap_at_boundaries(self):
        from asr_diar_server.bench.stm import slice_stm_window

        # Window 1.0-5.0 partially overlaps the first and third lines.
        out = slice_stm_window(self.SAMPLE, 1.0, 5.0, rebase=False)
        lines = {line.split(maxsplit=5)[5]: line.split() for line in out.splitlines()}
        assert lines["hello world"][3:5] == ["1.000", "2.000"]
        assert lines["baz qux"][3:5] == ["4.000", "5.000"]

    def test_empty_for_window_outside_all_lines(self):
        from asr_diar_server.bench.stm import slice_stm_window

        assert slice_stm_window(self.SAMPLE, 100.0, 200.0) == ""

    def test_empty_for_nonpositive_window(self):
        from asr_diar_server.bench.stm import slice_stm_window

        assert slice_stm_window(self.SAMPLE, 5.0, 5.0) == ""

    def test_recording_id_override_rewrites_session_column(self):
        from asr_diar_server.bench.stm import slice_stm_window

        out = slice_stm_window(self.SAMPLE, 0.0, 8.0, recording_id="clip_0_8")
        sessions = {line.split()[0] for line in out.splitlines()}
        assert sessions == {"clip_0_8"}

    def test_output_sorted_by_time_then_speaker(self):
        from asr_diar_server.bench.stm import slice_stm_window

        out = slice_stm_window(self.SAMPLE, 0.0, 8.0)
        times = [float(line.split()[3]) for line in out.splitlines()]
        assert times == sorted(times)
