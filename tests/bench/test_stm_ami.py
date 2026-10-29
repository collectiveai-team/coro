"""Tests for coro.bench.stm_ami: Reference STM from the AMI annotation tree."""

from __future__ import annotations

from pathlib import Path

import pytest


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

    def test_produces_stm_with_correct_speakers(self, ami_fixture: Path):
        from coro.bench.stm_ami import ami_meeting_to_stm

        result = ami_meeting_to_stm(ami_fixture, "TS3003a")
        assert "A" in result
        assert "B" in result

    def test_stm_lines_have_correct_format(self, ami_fixture: Path):
        from coro.bench.stm_ami import ami_meeting_to_stm

        result = ami_meeting_to_stm(ami_fixture, "TS3003a")
        rows = [line.split()[:3] for line in result.strip().split("\n")]
        assert [row[:2] for row in rows] == [["TS3003a", "1"], ["TS3003a", "1"]]
        assert sorted(row[2] for row in rows) == ["A", "B"]

    def test_stm_lines_sorted_by_time(self, ami_fixture: Path):
        from coro.bench.stm_ami import ami_meeting_to_stm

        result = ami_meeting_to_stm(ami_fixture, "TS3003a")
        lines = result.strip().split("\n")
        times = [float(line.split()[3]) for line in lines]
        assert times == sorted(times)

    def test_clip_reference_stm_windows_and_rebases(self, ami_fixture: Path):
        from coro.bench.ami import clip_reference_stm

        # Full meeting: A 0.0-1.0 "Hello world", B 1.0-2.0 "Good morning".
        clip = clip_reference_stm(ami_fixture, "TS3003a", start=0.5, duration=1.0)
        lines = [line.split(maxsplit=5) for line in clip.splitlines()]

        assert lines[0][2] == "A"
        assert lines[0][3:5] == ["0.000", "0.500"]
        assert lines[0][5] == "Hello world"
        assert lines[1][2] == "B"
        assert lines[1][3:5] == ["0.500", "1.000"]
        assert lines[1][5] == "Good morning"
