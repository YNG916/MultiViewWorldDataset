"""Read-only check that a saved configuration rebuilds its original route bank."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np

from multi_view_world_dataset.adapters.omnigibson import OmniGibsonAdapter
from multi_view_world_dataset.generator import _configuration_navigation_seed
from multi_view_world_dataset.utils.config import load_yaml_config
from multi_view_world_dataset.utils.runtime import resolve_runtime_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--configuration-dir", type=Path, required=True)
    parser.add_argument("--behavior-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--skip-navigation", action="store_true")
    parser.add_argument("--probe-fabric-sync", action="store_true")
    parser.add_argument("--probe-physics-step", action="store_true")
    parser.add_argument("--inspect-links", action="store_true")
    parser.add_argument("--refresh-collision-cache", action="store_true")
    args = parser.parse_args()
    saved = args.configuration_dir
    metadata = json.loads((saved / "config_meta.json").read_text(encoding="utf-8"))
    original = json.loads((saved / "navigation_context.json").read_text(encoding="utf-8"))
    config = load_yaml_config(args.config)
    runtime = resolve_runtime_paths(
        config, behavior_root=args.behavior_root, cache_root=args.cache_root
    )
    adapter = OmniGibsonAdapter(runtime, config)
    print("diagnostic_stage=adapter_start", flush=True)
    adapter.start()
    try:
        print("diagnostic_stage=load_scene", flush=True)
        adapter.load_scene(
            args.scene,
            robot_count=3,
            development_robot=(
                config["robot"]["final_model"]
                if config["robot"]["use_final_robot"]
                else config["robot"]["development_model"]
            ),
        )
        print("diagnostic_stage=load_snapshot", flush=True)
        snapshot = np.load(saved / "simulator_state.npy", allow_pickle=False)
        adapter.load_snapshot(snapshot)
        if args.refresh_collision_cache:
            adapter.refresh_collision_geometry_cache()
        restored_snapshot = adapter.dump_snapshot()
        snapshot_differences = np.abs(
            restored_snapshot.astype(np.float64) - snapshot.astype(np.float64)
        )
        expected_objects = {
            obj["instance_id"]: obj
            for obj in metadata["world_state"]["objects"]
        }
        aabb_differences = []
        pose_differences = []
        for obj in adapter.object_catalog():
            original_obj = expected_objects.get(obj.instance_id)
            if original_obj is None:
                continue
            delta = max(
                float(np.max(np.abs(np.asarray(getattr(obj, field)) - original_obj[field])))
                for field in ("bbox_min_world", "bbox_max_world")
            )
            if delta > 1.0e-4:
                aabb_differences.append((obj.instance_id, obj.category, delta))
            pose_delta = float(np.max(np.abs(
                np.asarray(obj.object_to_world) - original_obj["object_to_world"]
            )))
            if pose_delta > 1.0e-4:
                pose_differences.append((obj.instance_id, obj.category, pose_delta))
        refresh_probes = {}
        if args.probe_fabric_sync or args.probe_physics_step:
            actions = [
                ("sync_physx_to_fabric", adapter._og.sim.sync_physx_to_fabric),
                ("render", adapter._og.sim.render),
            ] if args.probe_fabric_sync else []
            if args.probe_physics_step:
                actions.append((
                    "step_physics_and_sync",
                    lambda: (
                        adapter._og.sim.step_physics(),
                        adapter._og.sim.sync_physx_to_fabric(),
                    ),
                ))
            for action, callback in actions:
                callback()
                refreshed = {}
                refreshed_poses = {}
                for obj in adapter.object_catalog():
                    original_obj = expected_objects.get(obj.instance_id)
                    if original_obj is None:
                        continue
                    delta = max(
                        float(np.max(np.abs(np.asarray(getattr(obj, field)) - original_obj[field])))
                        for field in ("bbox_min_world", "bbox_max_world")
                    )
                    if delta > 1.0e-4:
                        refreshed[obj.instance_id] = delta
                    pose_delta = float(np.max(np.abs(
                        np.asarray(obj.object_to_world) - original_obj["object_to_world"]
                    )))
                    if pose_delta > 1.0e-4:
                        refreshed_poses[obj.instance_id] = pose_delta
                refresh_probes[action] = {
                    "aabb_differences": refreshed,
                    "pose_differences": refreshed_poses,
                }
        link_diagnostics = {}
        if args.inspect_links:
            changed_ids = {item[0] for item in aabb_differences}
            catalog_by_path = {
                obj.native_path: obj for obj in adapter.object_catalog()
                if obj.instance_id in changed_ids
            }
            for native in adapter._require_scene().objects:
                record = catalog_by_path.get(str(native.prim_path))
                if record is None:
                    continue
                links = []
                for name, link in native.links.items():
                    position, orientation = link.get_position_orientation()
                    links.append({
                        "name": name,
                        "position": adapter._native_value(position).tolist(),
                        "orientation": adapter._native_value(orientation).tolist(),
                        "scale": adapter._native_value(link.scale).tolist(),
                    })
                link_diagnostics[record.instance_id] = {
                    "native_path": record.native_path,
                    "current_bbox_min_world": record.bbox_min_world,
                    "current_bbox_max_world": record.bbox_max_world,
                    "saved_bbox_min_world": expected_objects[record.instance_id]["bbox_min_world"],
                    "saved_bbox_max_world": expected_objects[record.instance_id]["bbox_max_world"],
                    "current_scale": record.scale,
                    "saved_scale": expected_objects[record.instance_id]["scale"],
                    "root_link_name": native.root_link.name,
                    "links": links[:12],
                }
        if args.skip_navigation:
            print(json.dumps({
                "changed_object_aabb_count": len(aabb_differences),
                "largest_object_aabb_differences": sorted(aabb_differences, key=lambda item: -item[2])[:10],
                "changed_object_pose_count": len(pose_differences),
                "largest_object_pose_differences": sorted(pose_differences, key=lambda item: -item[2])[:10],
                "snapshot_shape_equal": restored_snapshot.shape == snapshot.shape,
                "snapshot_max_absolute_difference": float(np.max(snapshot_differences)) if snapshot_differences.size else None,
                "snapshot_different_values": int(np.count_nonzero(snapshot_differences > 1.0e-5)),
                "restore_findings": {
                    key: value for key, value in adapter._runtime_findings.items()
                    if key.startswith("snapshot_")
                },
                "refresh_probes": refresh_probes,
                "link_diagnostics": link_diagnostics,
            }, sort_keys=True), flush=True)
            return 0
        print("diagnostic_stage=build_navigation", flush=True)
        adapter.prepare_navigation_context(
            str(metadata["exact_state_hash"]),
            _configuration_navigation_seed(metadata),
        )
        rebuilt = adapter.navigation_context_metadata()
        original_seeds = {
            floor: [route["route_seed"] for route in data["routes"]]
            for floor, data in original.items()
        }
        rebuilt_seeds = {
            floor: [route["route_seed"] for route in data["routes"]]
            for floor, data in rebuilt.items()
        }
        diagnostic_keys = (
            "point_free_cell_count", "permissive_footprint_cell_count",
            "footprint_safe_cell_count", "old_omnigibson_eroded_cell_count",
            "raw_route_attempts", "route_reject_counts",
        )
        floor_diagnostics = {}
        for floor in sorted(set(original) | set(rebuilt)):
            before = original.get(floor, {})
            after = rebuilt.get(floor, {})
            before_polygon = np.asarray(
                before.get("robot_footprint", {}).get("polygon_xy", [])
            )
            after_polygon = np.asarray(
                after.get("robot_footprint", {}).get("polygon_xy", [])
            )
            floor_diagnostics[floor] = {
                "original": {key: before.get("diagnostics", {}).get(key) for key in diagnostic_keys},
                "rebuilt": {key: after.get("diagnostics", {}).get(key) for key in diagnostic_keys},
                "footprint_polygon_max_difference_m": (
                    float(np.max(np.abs(before_polygon - after_polygon)))
                    if before_polygon.shape == after_polygon.shape and before_polygon.size
                    else None
                ),
                "original_first_route_seeds": original_seeds.get(floor, [])[:8],
                "rebuilt_first_route_seeds": rebuilt_seeds.get(floor, [])[:8],
            }
        result = {
            "scene_id": args.scene,
            "configuration_id": metadata["configuration_id"],
            "navigation_seed": _configuration_navigation_seed(metadata),
            "original_route_counts": {
                floor: data["route_bank_size"] for floor, data in original.items()
            },
            "rebuilt_route_counts": {
                floor: data["route_bank_size"] for floor, data in rebuilt.items()
            },
            "route_seeds_identical": original_seeds == rebuilt_seeds,
            "floor_diagnostics": floor_diagnostics,
            "changed_object_aabb_count": len(aabb_differences),
            "largest_object_aabb_differences": sorted(aabb_differences, key=lambda item: -item[2])[:10],
            "changed_object_pose_count": len(pose_differences),
            "largest_object_pose_differences": sorted(pose_differences, key=lambda item: -item[2])[:10],
            "snapshot_shape_equal": restored_snapshot.shape == snapshot.shape,
            "snapshot_max_absolute_difference": float(np.max(snapshot_differences)) if snapshot_differences.size else None,
            "snapshot_different_values": int(np.count_nonzero(snapshot_differences > 1.0e-5)),
            "restore_findings": {
                key: value for key, value in adapter._runtime_findings.items()
                if key.startswith("snapshot_")
            },
        }
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["route_seeds_identical"] else 1
    except BaseException:
        print("diagnostic_error=" + traceback.format_exc(), flush=True)
        raise
    finally:
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
