from __future__ import annotations

import importlib.util
from pathlib import Path


def _compat_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "resume_overlay" / "compat.py"
    spec = importlib.util.spec_from_file_location("mvwd_resume_overlay_compat", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resume_overlay_skips_only_committed_complete_configurations(tmp_path):
    can_skip = _compat_module().can_skip_navigation_rebuild
    shard = tmp_path / "shard"
    scene = "Rs_int"
    configuration = "config_000"
    config_root = shard / "configurations" / scene / configuration
    config_root.mkdir(parents=True)
    (config_root / "config_meta.json").write_text("{}", encoding="utf-8")
    assert not can_skip(shard, scene, configuration, 3)

    (config_root / "navigation_context.json").write_text("{}", encoding="utf-8")
    for index in (0, 1):
        episode = shard / "episodes" / scene / configuration / f"episode_{index:03d}"
        episode.mkdir(parents=True)
        (episode / "meta.json").write_text("{}", encoding="utf-8")
    assert not can_skip(shard, scene, configuration, 3)

    staging = shard / "episodes" / scene / configuration / ".episode_002.staging"
    staging.mkdir()
    (staging / "meta.json").write_text("{}", encoding="utf-8")
    assert not can_skip(shard, scene, configuration, 3)

    final = shard / "episodes" / scene / configuration / "episode_002"
    final.mkdir()
    (final / "meta.json").write_text("{}", encoding="utf-8")
    assert can_skip(shard, scene, configuration, 3)
