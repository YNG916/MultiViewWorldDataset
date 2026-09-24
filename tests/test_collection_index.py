from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_collection_index.py"
    spec = importlib.util.spec_from_file_location("build_collection_index", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _root(base: Path, name: str, scene: str, fingerprint: str) -> Path:
    root = base / name
    splits = {"Rs_int": "train", "Beechwood_1_int": "val"}
    _json(root / "global" / "production_manifest.json", {
        "schema_version": "1.1.0",
        "robot_asset_fingerprint": "same-robot",
        "configuration_fingerprint": fingerprint,
        "generator_source_fingerprint": fingerprint,
        "split_mapping": splits,
        "selected_scenes": list(splits),
    })
    shard = root / "shards" / scene
    _json(shard / "dataset_meta.json", {"configuration_fingerprint": fingerprint})
    _json(shard / "shard_status.json", {"status": "complete"})
    episode = shard / "episodes" / scene / "config_000" / "episode_000"
    _json(episode / "meta.json", {})
    _json(episode / "qa.json", [{"passed": True}])
    _json(episode / "generation_metrics.json", {})
    (episode / "trajectories.npz").write_bytes(b"trajectory")
    return root


def test_collection_keeps_versions_and_disjoint_scenes(tmp_path):
    old = _root(tmp_path, "old", "Rs_int", "old-fingerprint")
    new = _root(tmp_path, "new", "Beechwood_1_int", "new-fingerprint")
    output = tmp_path / "collection.json"
    index = _module().build_collection_index([old, new], output)
    assert output.is_file()
    assert index["episode_count"] == 2
    assert {item["version_fingerprint"] for item in index["episodes"]} == {
        "old-fingerprint", "new-fingerprint",
    }
    assert {item["split"] for item in index["episodes"]} == {"train", "val"}


def test_collection_refuses_duplicate_scene_or_incompatible_robot(tmp_path):
    old = _root(tmp_path, "old", "Rs_int", "old-fingerprint")
    duplicate = _root(tmp_path, "duplicate", "Rs_int", "new-fingerprint")
    with pytest.raises(ValueError, match="multiple producer versions"):
        _module().build_collection_index([old, duplicate], tmp_path / "index.json")
    new = _root(tmp_path, "new", "Beechwood_1_int", "new-fingerprint")
    manifest_path = new / "global" / "production_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["robot_asset_fingerprint"] = "different-robot"
    _json(manifest_path, manifest)
    with pytest.raises(ValueError, match="Incompatible"):
        _module().build_collection_index([old, new], tmp_path / "index.json")
    assert not (tmp_path / "index.json").exists()
