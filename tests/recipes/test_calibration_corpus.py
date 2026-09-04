"""coro.recipes.calibration_corpus: duration filtering, harvesting, and best-effort main loop.

``harvest_local_fleurs_es`` is exercised against a real tar.gz of real (tiny,
synthetic) WAV clips -- this is the recipe's actual duration-filtering
contract, worth testing for real rather than mocking `tarfile` itself.
Network-touching pieces (``huggingface_hub``, ``coro.bench.utils.hf_parquet``)
are mocked.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from coro.recipes.calibration_corpus import (
    MAX_S,
    MIN_S,
    clip_duration,
    harvest_local_fleurs_es,
    harvest_remote_fleurs,
    main,
)

_SAMPLE_RATE = 16000


def _write_wav(path: Path, seconds: float) -> None:
    sf.write(str(path), np.zeros(int(seconds * _SAMPLE_RATE), dtype=np.float32), _SAMPLE_RATE)


def _build_fleurs_snapshot(tmp_path: Path, durations: dict[str, float]) -> Path:
    """Build a real fleurs_es_dir (`<root>/audio/test.tar.gz`) of real WAV clips.

    Mirrors `_resolve_fleurs_es_snapshot`'s return contract exactly, so the
    mock in these tests can return the root directly.
    """
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    with tarfile.open(audio_dir / "test.tar.gz", "w:gz") as tar:
        for name, seconds in durations.items():
            wav_path = wav_dir / f"{name}.wav"
            _write_wav(wav_path, seconds)
            tar.add(wav_path, arcname=f"test/{name}.wav")
    return tmp_path


class TestClipDuration:
    def test_returns_seconds_from_frames_and_samplerate(self, tmp_path):
        wav_path = tmp_path / "clip.wav"
        _write_wav(wav_path, 2.5)
        assert clip_duration(wav_path) == pytest.approx(2.5, rel=1e-6)


class TestHarvestLocalFleursEs:
    def test_keeps_only_clips_within_the_duration_window(self, tmp_path):
        snapshot_root = _build_fleurs_snapshot(
            tmp_path,
            {
                "too_short": MIN_S - 1.0,
                "in_range_a": MIN_S + 1.0,
                "in_range_b": MAX_S - 1.0,
                "too_long": MAX_S + 1.0,
            },
        )
        out_dir = tmp_path / "out"

        with patch(
            "coro.recipes.calibration_corpus._resolve_fleurs_es_snapshot",
            autospec=True,
            return_value=snapshot_root,
        ):
            picked = harvest_local_fleurs_es(out_dir, target=10)

        picked_names = {Path(p["path"]).stem for p in picked}
        assert picked_names == {"es-in_range_a", "es-in_range_b"}
        assert all(p["lang"] == "es" for p in picked)
        assert all(p["source"] == "fleurs-es_419" for p in picked)

    def test_stops_once_the_target_count_is_reached(self, tmp_path):
        snapshot_root = _build_fleurs_snapshot(
            tmp_path, {f"clip_{i}": MIN_S + 1.0 for i in range(5)}
        )
        out_dir = tmp_path / "out"

        with patch(
            "coro.recipes.calibration_corpus._resolve_fleurs_es_snapshot",
            autospec=True,
            return_value=snapshot_root,
        ):
            picked = harvest_local_fleurs_es(out_dir, target=2)

        assert len(picked) == 2

    def test_discards_probe_files_outside_the_window(self, tmp_path):
        """Rejected clips must not leave `_probe_*` files behind in out_dir."""
        snapshot_root = _build_fleurs_snapshot(tmp_path, {"too_short": MIN_S - 1.0})
        out_dir = tmp_path / "out"

        with patch(
            "coro.recipes.calibration_corpus._resolve_fleurs_es_snapshot",
            autospec=True,
            return_value=snapshot_root,
        ):
            harvest_local_fleurs_es(out_dir, target=10)

        assert list((out_dir / "es").glob("_probe_*")) == []


class TestHarvestRemoteFleurs:
    def test_filters_by_duration_and_stops_at_target(self, tmp_path):
        rows = [
            {"id": "a", "audio": {"bytes": b"fake"}},
            {"id": "b", "audio": {"bytes": b"fake"}},
            {"id": "c", "audio": {"bytes": b"fake"}},
        ]
        durations = iter([MIN_S - 1.0, MIN_S + 1.0, MIN_S + 2.0])

        def _fake_transcode(_data: bytes, dst: Path) -> None:
            _write_wav(dst, next(durations))

        with (
            patch(
                "coro.bench.utils.hf_parquet.resolve_shard_urls",
                autospec=True,
                return_value=["shard.parquet"],
            ),
            patch(
                "coro.bench.utils.hf_parquet.iter_parquet_rows",
                autospec=True,
                return_value=iter(rows),
            ),
            patch(
                "coro.bench.utils.audio_clips.transcode_bytes_to_wav",
                autospec=True,
                side_effect=_fake_transcode,
            ),
        ):
            picked = harvest_remote_fleurs("en", "en_us", tmp_path / "out", target=2)

        assert len(picked) == 2
        assert {p["lang"] for p in picked} == {"en"}
        assert {p["source"] for p in picked} == {"fleurs-en_us"}


class TestMain:
    def test_aggregates_local_and_remote_harvests_into_one_manifest(self, tmp_path):
        out_dir = tmp_path / "out"
        local = [{"path": "es-1.wav", "lang": "es", "duration_s": 5.0, "source": "fleurs-es_419"}]
        remote = [{"path": "en-1.wav", "lang": "en", "duration_s": 6.0, "source": "fleurs-en_us"}]

        with (
            patch(
                "coro.recipes.calibration_corpus.harvest_local_fleurs_es",
                autospec=True,
                return_value=local,
            ),
            patch(
                "coro.recipes.calibration_corpus.harvest_remote_fleurs",
                autospec=True,
                return_value=remote,
            ),
        ):
            main(["--out-dir", str(out_dir), "--per-language-target", "3"])

        manifest = json.loads((out_dir / "calibration_manifest.json").read_text())
        assert manifest.count(local[0]) == 1
        # 4 remote languages, each contributing the same fake `remote` list.
        assert manifest.count(remote[0]) == 4

    def test_a_failed_remote_language_does_not_abort_the_run(self, tmp_path):
        out_dir = tmp_path / "out"
        local = [{"path": "es-1.wav", "lang": "es", "duration_s": 5.0, "source": "fleurs-es_419"}]

        with (
            patch(
                "coro.recipes.calibration_corpus.harvest_local_fleurs_es",
                autospec=True,
                return_value=local,
            ),
            patch(
                "coro.recipes.calibration_corpus.harvest_remote_fleurs",
                autospec=True,
                side_effect=RuntimeError("network hiccup"),
            ),
        ):
            main(["--out-dir", str(out_dir)])

        manifest = json.loads((out_dir / "calibration_manifest.json").read_text())
        assert manifest == local
