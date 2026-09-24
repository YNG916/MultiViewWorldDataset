from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "archive_completed_shard.py"
    spec = importlib.util.spec_from_file_location("archive_completed_shard", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"content")


def _complete_root(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "dataset"
    shard = root / "shards" / "Rs_int"
    _json(root / "global" / "production_manifest.json", {
        "selected_scenes": ["Rs_int"],
        "configuration_fingerprint": "fingerprint",
        "schema_version": "1.1.0",
    })
    _json(shard / "shard_status.json", {
        "status": "complete", "accepted_configurations": 1, "accepted_episodes": 1,
    })
    _json(shard / "generation_status.json", {"status": "pass"})
    _json(shard / "dataset_meta.json", {
        "configuration_fingerprint": "fingerprint", "schema_version": "1.1.0",
    })
    _json(shard / "configurations" / "Rs_int" / "config_000" / "config_meta.json", {})
    episode = shard / "episodes" / "Rs_int" / "config_000" / "episode_000"
    _json(episode / "meta.json", {})
    _json(episode / "qa.json", [{"check": "paired_trajectory_equality", "passed": True}])
    _json(episode / "generation_metrics.json", {})
    _json(episode / "events.json", [])
    _json(episode / "state_before.json", {})
    _json(episode / "state_after.json", {})
    _file(episode / "trajectories.npz")
    for name in ("world_before", "world_after", "environment_after"):
        _file(episode / "bev" / f"{name}.npz")
    for phase in ("before", "after"):
        reference = f"robot_views/{phase}/robot_00.npz"
        _file(episode / reference)
        _json(episode / f"observations_{phase}.json", [
            {"modality_refs": {"rgb": f"{reference}::rgb[0]"}}
        ])
    return root, episode


def test_archive_requires_complete_qa_and_confirmed_eviction(tmp_path):
    archive = _module()
    root, episode = _complete_root(tmp_path)
    destination = tmp_path / "archive"
    result = archive.archive_scene(root, destination, "Rs_int")
    assert not result["evicted"]
    assert episode.is_dir()
    assert (destination / "shards" / "Rs_int" / "episodes" / "Rs_int" / "config_000" / "episode_000" / "meta.json").is_file()
    with pytest.raises(ValueError, match="confirm-scene"):
        archive.archive_scene(root, destination, "Rs_int", evict=True)
    result = archive.archive_scene(
        root, destination, "Rs_int", evict=True, confirm_scene="Rs_int"
    )
    assert result["evicted"]
    assert not (root / "shards" / "Rs_int").exists()
    assert archive.restore_scene(root, destination, "Rs_int")["restored"]
    assert episode.is_dir()
    assert archive._tree_manifest(root / "shards" / "Rs_int") == archive._tree_manifest(
        destination / "shards" / "Rs_int"
    )


def test_archive_rejects_partial_or_missing_modality(tmp_path):
    archive = _module()
    root, episode = _complete_root(tmp_path)
    destination = tmp_path / "archive"
    (episode / "robot_views" / "before" / "robot_00.npz").unlink()
    with pytest.raises(RuntimeError, match="modality archive missing"):
        archive.archive_scene(root, destination, "Rs_int")
    assert episode.is_dir()
    _file(episode / "robot_views" / "before" / "robot_00.npz")
    _json(root / "shards" / "Rs_int" / "shard_status.json", {"status": "running"})
    with pytest.raises(RuntimeError, match="not fully complete"):
        archive.archive_scene(root, destination, "Rs_int")
    assert episode.is_dir()
