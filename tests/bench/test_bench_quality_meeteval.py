"""Tests for the MeetEval adapter of the Quality Benchmark."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from support.paths import PYPROJECT


class TestPyprojectBenchExtra:
    def test_pyproject_declares_bench_optional_extra(self):
        text = PYPROJECT.read_text()
        assert "[project.optional-dependencies]" in text
        assert "bench = [" in text
        assert "meeteval" in text
        assert "rich" in text


class TestRequireMeeteval:
    def test_require_meeteval_exits_with_helpful_message(self):
        with patch.dict(sys.modules, {"meeteval": None}):
            from coro.bench.quality_meeteval import require_meeteval

            with pytest.raises(SystemExit) as exc_info:
                require_meeteval()
            assert exc_info.value.code == 1

    def test_require_meeteval_returns_module_when_available(self):
        mock_meeteval = MagicMock()
        with patch.dict(sys.modules, {"meeteval": mock_meeteval}):
            from coro.bench.quality_meeteval import require_meeteval

            result = require_meeteval()
            assert result is mock_meeteval


class TestMeetevalInstallHint:
    """The printed install hint must be a command that actually works."""

    def _hint(self, capsys) -> str:
        from coro.bench.quality_meeteval import require_meeteval

        # A None entry in sys.modules makes `import meeteval` raise ImportError.
        with patch.dict(sys.modules, {"meeteval": None}), pytest.raises(SystemExit):
            require_meeteval()
        return capsys.readouterr().err

    def test_hint_uses_the_dependency_group_command(self, capsys):
        assert "uv sync --group bench" in self._hint(capsys)

    def test_hint_does_not_claim_an_installable_extra(self, capsys):
        assert "coro[bench]" not in self._hint(capsys)

    def test_bench_is_a_dependency_group_not_an_extra(self):
        """Guards the hint against the packaging metadata drifting back."""
        import tomllib

        pyproject = tomllib.loads(PYPROJECT.read_text())
        assert "bench" in pyproject["dependency-groups"]
        assert "bench" not in pyproject["project"].get("optional-dependencies", {})
