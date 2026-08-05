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


class TestRttmToStm:
    """rttm_to_stm converts RTTM SPEAKER turns to a diarization-only STM."""

    _RTTM = (
        "SPEAKER rec 1 2.50 1.50 <NA> <NA> spkB <NA> <NA>\n"
        "SPEAKER rec 1 0.00 2.00 <NA> <NA> spkA <NA> <NA>\n"
    )

    def test_speaker_turns_become_sorted_stm_with_sentinel_text(self):
        from coro.bench.stm import DIARIZATION_ONLY_TEXT, rttm_to_stm

        lines = rttm_to_stm(self._RTTM, "rec").strip().split("\n")
        # Sorted by start time; end = onset + duration; sentinel text.
        assert lines[0] == f"rec 1 spkA 0.000 2.000 {DIARIZATION_ONLY_TEXT}"
        assert lines[1] == f"rec 1 spkB 2.500 4.000 {DIARIZATION_ONLY_TEXT}"

    def test_non_speaker_and_nonpositive_duration_rows_dropped(self):
        from coro.bench.stm import rttm_to_stm

        rttm = (
            "SPKR-INFO rec 1 <NA> <NA> <NA> unknown spkA <NA> <NA>\n"
            "SPEAKER rec 1 1.00 0.00 <NA> <NA> spkA <NA> <NA>\n"
            "SPEAKER rec 1 1.00 0.50 <NA> <NA> spkA <NA> <NA>\n"
        )
        assert rttm_to_stm(rttm, "rec").strip().split("\n") == ["rec 1 spkA 1.000 1.500 <sd>"]

    def test_empty_rttm_yields_empty_string(self):
        from coro.bench.stm import rttm_to_stm

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
        (words_dir / "TS3003a.B.words.xml").write_text(words_xml.replace("TS3003a.A", "TS3003a.B"))

        seg_a_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<nite:root xmlns:nite="http://nite.sourceforge.net/">\n'
            '  <segment nite:id="s0">\n'
            '    <child href="TS3003a.A.words.xml#id(TS3003a.A.words0)"/>\n'
            '    <child href="TS3003a.A.words.xml#id(TS3003a.A.words1)"/>\n'
            "  </segment>\n"
            "</nite:root>\n"
        )
        (segments_dir / "TS3003a.A.segments.xml").write_text(seg_a_xml)

        seg_b_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<nite:root xmlns:nite="http://nite.sourceforge.net/">\n'
            '  <segment nite:id="s0">\n'
            '    <child href="TS3003a.B.words.xml#id(TS3003a.B.words0)"/>\n'
            '    <child href="TS3003a.B.words.xml#id(TS3003a.B.words1)"/>\n'
            "  </segment>\n"
            "</nite:root>\n"
        )
        (segments_dir / "TS3003a.B.segments.xml").write_text(seg_b_xml)
        return root

    @pytest.fixture()
    def ami_realistic_fixture(self, tmp_path: Path) -> Path:
        """An AMI tree shaped like the real corpus.

        Real AMI word files interleave ``<w>`` with ``<disfmarker>``,
        ``<vocalsound>`` and ``<gap>``, and a segment's ``<child>`` href is a
        RANGE whose endpoints may be any of those — not necessarily a word.
        Real segments carry ``transcriber_start``/``transcriber_end``, not
        ``starttime``/``endtime``.
        """
        root = tmp_path / "amicorpus"
        words_dir = root / "TS3003a" / "words"
        segments_dir = root / "TS3003a" / "segments"
        words_dir.mkdir(parents=True)
        segments_dir.mkdir(parents=True)

        (words_dir / "TS3003a.A.words.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<nite:root xmlns:nite="http://nite.sourceforge.net/">\n'
            '  <w nite:id="TS3003a.A.words0" starttime="0.0" endtime="0.5">Hello</w>\n'
            '  <w nite:id="TS3003a.A.words1" starttime="0.5" endtime="1.0">everyone</w>\n'
            '  <disfmarker nite:id="TS3003a.A.words2" starttime="1.0" endtime="1.1"/>\n'
            '  <w nite:id="TS3003a.A.words3" starttime="2.0" endtime="2.5">second</w>\n'
            '  <w nite:id="TS3003a.A.words4" starttime="2.5" endtime="3.0">turn</w>\n'
            '  <vocalsound nite:id="TS3003a.A.words5" starttime="3.0" endtime="3.2"/>\n'
            "</nite:root>\n"
        )
        (segments_dir / "TS3003a.A.segments.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<nite:root xmlns:nite="http://nite.sourceforge.net/">\n'
            '  <segment nite:id="s0" transcriber_start="0.0" transcriber_end="1.1">\n'
            '    <child href="TS3003a.A.words.xml#id(TS3003a.A.words0)..'
            'id(TS3003a.A.words2)"/>\n'
            "  </segment>\n"
            '  <segment nite:id="s1" transcriber_start="2.0" transcriber_end="3.2">\n'
            '    <child href="TS3003a.A.words.xml#id(TS3003a.A.words3)..'
            'id(TS3003a.A.words5)"/>\n'
            "  </segment>\n"
            "</nite:root>\n"
        )
        return root

    def test_href_range_ending_on_a_non_word_keeps_its_words(self, ami_realistic_fixture: Path):
        """A range terminating on <disfmarker>/<vocalsound> must not drop the segment.

        AMI href ranges routinely end on a non-word element. Resolving the range
        against a word-only index loses every word in it — 28.5% of segments and
        ~30k reference words across the AMI ES meetings.
        """
        from coro.bench.stm import ami_meeting_to_stm

        result = ami_meeting_to_stm(ami_realistic_fixture, "TS3003a")

        assert "Hello everyone" in result
        assert "second turn" in result
        assert len(result.strip().splitlines()) == 2

    def test_window_keeps_only_words_inside_it(self, ami_realistic_fixture: Path):
        """A segment straddling the window contributes only its in-window words.

        Clipping rendered STM text can only clamp a segment's TIMES; the text
        column comes along whole, so a segment crossing the edge donates every
        word to a clip whose audio holds just a few of them. Those words are
        unspeakable and score as deletions. Word times are known at build time,
        so the window is applied there instead.
        """
        from coro.bench.stm import ami_meeting_to_stm

        # Window ends mid-segment s1 ("second" at 2.0, "turn" at 2.5).
        result = ami_meeting_to_stm(ami_realistic_fixture, "TS3003a", window=(0.0, 2.4))

        assert "Hello everyone" in result
        assert "second" in result
        assert "turn" not in result

    def test_window_drops_segments_entirely_outside(self, ami_realistic_fixture: Path):
        """Segments with no word starting inside the window are dropped."""
        from coro.bench.stm import ami_meeting_to_stm

        result = ami_meeting_to_stm(ami_realistic_fixture, "TS3003a", window=(1.9, 3.5))

        assert "Hello" not in result
        assert "second turn" in result
        assert len(result.strip().splitlines()) == 1

    def test_window_uses_half_open_word_start_membership(self, ami_realistic_fixture: Path):
        """Membership is by word START on ``[lo, hi)`` — one window owns each word.

        Matches the Overlap Token Acceptance convention, so adjacent windows
        partition the words rather than sharing or dropping any at the seam.
        """
        from coro.bench.stm import ami_meeting_to_stm

        lower = ami_meeting_to_stm(ami_realistic_fixture, "TS3003a", window=(0.0, 2.0))
        upper = ami_meeting_to_stm(ami_realistic_fixture, "TS3003a", window=(2.0, 4.0))

        # "second" starts exactly at 2.0: it belongs to the upper window only.
        assert "second" not in lower
        assert "second" in upper

    def test_no_window_is_unchanged(self, ami_realistic_fixture: Path):
        """Omitting the window keeps the full-meeting behaviour byte for byte."""
        from coro.bench.stm import ami_meeting_to_stm

        assert ami_meeting_to_stm(ami_realistic_fixture, "TS3003a", window=None) == (
            ami_meeting_to_stm(ami_realistic_fixture, "TS3003a")
        )

    def test_segment_falls_back_to_transcriber_times(self, tmp_path: Path):
        """When no href resolves, the segment's own transcriber_* window is used."""
        root = tmp_path / "amicorpus"
        words_dir = root / "TS3003a" / "words"
        segments_dir = root / "TS3003a" / "segments"
        words_dir.mkdir(parents=True)
        segments_dir.mkdir(parents=True)
        (words_dir / "TS3003a.A.words.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<nite:root xmlns:nite="http://nite.sourceforge.net/">\n'
            '  <w nite:id="TS3003a.A.words0" starttime="0.0" endtime="0.5">Hello</w>\n'
            "</nite:root>\n"
        )
        (segments_dir / "TS3003a.A.segments.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<nite:root xmlns:nite="http://nite.sourceforge.net/">\n'
            '  <segment nite:id="s0" transcriber_start="0.0" transcriber_end="1.0">\n'
            '    <child href="TS3003a.A.words.xml#id(TS3003a.A.unknown)"/>\n'
            "  </segment>\n"
            "</nite:root>\n"
        )

        from coro.bench.stm import ami_meeting_to_stm

        assert "Hello" in ami_meeting_to_stm(root, "TS3003a")

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

    def test_clip_reference_stm_windows_and_rebases(self, ami_fixture: Path):
        """A clip's reference holds only the words its audio contains.

        Previously this asserted the whole segment text survived the clip edge
        ("Hello world", "Good morning"), which is the defect: those references
        credit a clip with words spoken outside it, and the ASR is charged a
        deletion for each. Membership is now per word on ``[lo, hi)``.
        """
        from coro.bench.ami import clip_reference_stm

        # Full meeting: A "Hello"(0.0-0.5) "world"(0.5-1.0),
        #               B "Good"(1.0-1.5) "morning"(1.5-2.0).
        # Clip [0.5, 1.5) holds "world" and "Good" only.
        clip = clip_reference_stm(ami_fixture, "TS3003a", start=0.5, duration=1.0)
        lines = [line.split(maxsplit=5) for line in clip.splitlines()]

        assert lines[0][2] == "A"
        assert lines[0][3:5] == ["0.000", "0.500"]
        assert lines[0][5] == "world"
        assert lines[1][2] == "B"
        assert lines[1][3:5] == ["0.500", "1.000"]
        assert lines[1][5] == "Good"


class TestSliceStmWindow:
    """slice_stm_window keeps overlapping lines, clamps, and rebases times."""

    SAMPLE = (
        "m 1 A 0.000 2.000 hello world\n"
        "m 1 B 2.000 4.000 foo bar\n"
        "m 1 A 4.000 6.000 baz qux\n"
        "m 1 B 6.000 8.000 out of window\n"
    )

    def test_keeps_only_overlapping_lines(self):
        from coro.bench.stm import slice_stm_window

        out = slice_stm_window(self.SAMPLE, 2.0, 6.0)
        texts = [line.split(maxsplit=5)[5] for line in out.splitlines()]
        assert texts == ["foo bar", "baz qux"]

    def test_rebases_times_to_window_start(self):
        from coro.bench.stm import slice_stm_window

        out = slice_stm_window(self.SAMPLE, 2.0, 6.0)
        first = out.splitlines()[0].split()
        # "foo bar" was 2.0-4.0; rebased to 0.0-2.0.
        assert first[3] == "0.000"
        assert first[4] == "2.000"

    def test_rebase_false_preserves_absolute_times(self):
        from coro.bench.stm import slice_stm_window

        out = slice_stm_window(self.SAMPLE, 2.0, 6.0, rebase=False)
        first = out.splitlines()[0].split()
        assert first[3] == "2.000"

    def test_clamps_partial_overlap_at_boundaries(self):
        from coro.bench.stm import slice_stm_window

        # Window 1.0-5.0 partially overlaps the first and third lines.
        out = slice_stm_window(self.SAMPLE, 1.0, 5.0, rebase=False)
        lines = {line.split(maxsplit=5)[5]: line.split() for line in out.splitlines()}
        assert lines["hello world"][3:5] == ["1.000", "2.000"]
        assert lines["baz qux"][3:5] == ["4.000", "5.000"]

    def test_empty_for_window_outside_all_lines(self):
        from coro.bench.stm import slice_stm_window

        assert slice_stm_window(self.SAMPLE, 100.0, 200.0) == ""

    def test_empty_for_nonpositive_window(self):
        from coro.bench.stm import slice_stm_window

        assert slice_stm_window(self.SAMPLE, 5.0, 5.0) == ""

    def test_recording_id_override_rewrites_session_column(self):
        from coro.bench.stm import slice_stm_window

        out = slice_stm_window(self.SAMPLE, 0.0, 8.0, recording_id="clip_0_8")
        sessions = {line.split()[0] for line in out.splitlines()}
        assert sessions == {"clip_0_8"}

    def test_output_sorted_by_time_then_speaker(self):
        from coro.bench.stm import slice_stm_window

        out = slice_stm_window(self.SAMPLE, 0.0, 8.0)
        times = [float(line.split()[3]) for line in out.splitlines()]
        assert times == sorted(times)


class TestSpeakerTimelineToStm:
    """speaker_timeline_to_stm converts a SpeakerSegment timeline to a DER-only STM."""

    def _timeline(self):
        from coro.core.models import SpeakerSegment

        return [
            SpeakerSegment(start=0.5, end=1.0, speaker=2),
            SpeakerSegment(start=0.0, end=0.5, speaker=1),
        ]

    def test_emits_diarization_only_sentinel_text(self):
        from coro.bench.stm import DIARIZATION_ONLY_TEXT, speaker_timeline_to_stm

        out = speaker_timeline_to_stm(self._timeline(), "MEET1")
        for line in out.splitlines():
            assert line.split(maxsplit=5)[5] == DIARIZATION_ONLY_TEXT

    def test_lines_use_recording_id_and_speaker_label(self):
        from coro.bench.stm import speaker_timeline_to_stm

        out = speaker_timeline_to_stm(self._timeline(), "MEET1")
        first = out.splitlines()[0].split()
        assert first[0] == "MEET1"
        assert first[2] == "spk1"
        assert first[3] == "0.000"
        assert first[4] == "0.500"

    def test_output_sorted_by_time_then_speaker(self):
        from coro.bench.stm import speaker_timeline_to_stm

        out = speaker_timeline_to_stm(self._timeline(), "MEET1")
        times = [float(line.split()[3]) for line in out.splitlines()]
        assert times == sorted(times)

    def test_drops_zero_and_negative_duration_segments(self):
        from coro.bench.stm import speaker_timeline_to_stm
        from coro.core.models import SpeakerSegment

        timeline = [SpeakerSegment(start=1.0, end=1.0, speaker=1)]
        assert speaker_timeline_to_stm(timeline, "MEET1") == ""
