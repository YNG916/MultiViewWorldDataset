"""Read-only completeness check for the frozen-producer resume overlay."""
from __future__ import annotations

from pathlib import Path


def can_skip_navigation_rebuild(
    shard: Path, scene_id: str, configuration_id: str, requested_episodes: int
) -> bool:
    """Only skip when a configuration and every requested episode were committed."""
    configuration = shard / "configurations" / scene_id / configuration_id
    if not (configuration / "config_meta.json").is_file():
        return False
    if not (configuration / "navigation_context.json").is_file():
        return False
    for index in range(requested_episodes):
        episode = shard / "episodes" / scene_id / configuration_id / f"episode_{index:03d}"
        if not (episode / "meta.json").is_file():
            return False
    return True
