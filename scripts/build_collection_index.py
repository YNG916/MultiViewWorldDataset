"""Build a provenance-preserving logical index across production versions.

This never copies or rewrites episode data. Different producer fingerprints
remain separate roots and every collection entry retains its source version.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_collection_index(
    roots: list[Path], output: Path, *, include_partial: bool = False
) -> dict:
    if len(roots) < 2:
        raise ValueError("A collection needs at least two production roots")
    resolved = [root.resolve() for root in roots]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Production roots must be distinct")
    output = output.resolve()
    versions = []
    episodes = []
    seen_scenes: set[str] = set()
    schema = None
    robot_asset = None
    split_mapping = None
    for root in resolved:
        manifest = _read_json(root / "global" / "production_manifest.json")
        if schema is None:
            schema = manifest["schema_version"]
            robot_asset = manifest["robot_asset_fingerprint"]
            split_mapping = manifest["split_mapping"]
        elif (
            manifest["schema_version"] != schema
            or manifest["robot_asset_fingerprint"] != robot_asset
            or manifest["split_mapping"] != split_mapping
        ):
            raise ValueError(f"Incompatible schema, robot, or split mapping: {root}")
        fingerprint = manifest["configuration_fingerprint"]
        version = {
            "root": str(root),
            "configuration_fingerprint": fingerprint,
            "generator_source_fingerprint": manifest["generator_source_fingerprint"],
            "robot_asset_fingerprint": manifest["robot_asset_fingerprint"],
            "scenes": [],
        }
        for scene_id in manifest["selected_scenes"]:
            shard = root / "shards" / scene_id
            if not shard.is_dir():
                continue
            shard_meta_path = shard / "dataset_meta.json"
            if not shard_meta_path.is_file():
                continue
            shard_meta = _read_json(shard_meta_path)
            if shard_meta["configuration_fingerprint"] != fingerprint:
                raise ValueError(f"Shard fingerprint mismatch: {shard}")
            status_path = shard / "shard_status.json"
            complete = (
                status_path.is_file()
                and _read_json(status_path).get("status") == "complete"
            )
            if not complete and not include_partial:
                continue
            committed = sorted(
                (shard / "episodes" / scene_id).glob("config_*/episode_*/meta.json")
            )
            if not committed:
                continue
            if scene_id in seen_scenes:
                raise ValueError(f"Scene occurs in multiple producer versions: {scene_id}")
            seen_scenes.add(scene_id)
            for meta_path in committed:
                episode = meta_path.parent
                qa = _read_json(episode / "qa.json")
                if not isinstance(qa, list) or not qa or not all(
                    isinstance(check, dict) and check.get("passed") is True
                    for check in qa
                ):
                    raise ValueError(f"Committed episode lacks passing QA: {episode}")
                for name in ("generation_metrics.json", "trajectories.npz"):
                    if not (episode / name).is_file():
                        raise ValueError(f"Committed episode is incomplete: {episode}")
                episodes.append({
                    "scene_id": scene_id,
                    "configuration_id": episode.parent.name,
                    "episode_id": episode.name,
                    "split": manifest["split_mapping"][scene_id],
                    "version_fingerprint": fingerprint,
                    "root": str(root),
                    "relative_path": str(episode.relative_to(root)),
                })
            version["scenes"].append({
                "scene_id": scene_id,
                "status": "complete" if complete else "partial",
                "episode_count": len(committed),
            })
        versions.append(version)
    if not episodes:
        raise ValueError("No QA-passing committed episodes to index")
    index = {
        "format": "mvwd-multi-provenance-collection-v1",
        "schema_version": schema,
        "robot_asset_fingerprint": robot_asset,
        "split_mapping": split_mapping,
        "versions": versions,
        "episode_count": len(episodes),
        "episodes": sorted(
            episodes,
            key=lambda item: (item["scene_id"], item["configuration_id"], item["episode_id"]),
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-partial", action="store_true")
    arguments = parser.parse_args()
    result = build_collection_index(
        arguments.root, arguments.output, include_partial=arguments.include_partial
    )
    print(json.dumps({
        "output": str(arguments.output.resolve()),
        "versions": len(result["versions"]),
        "episodes": result["episode_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
