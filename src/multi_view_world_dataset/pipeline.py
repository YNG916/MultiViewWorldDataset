from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from multi_view_world_dataset.adapters.omnigibson import OmniGibsonAdapter
from multi_view_world_dataset.cameras.calibration import PinholeCalibration
from multi_view_world_dataset.cameras.overlap import build_overlap_graph
from multi_view_world_dataset.rendering.inspection import save_rgb, write_html_summary
from multi_view_world_dataset.sampling.splits import assign_scene_family_splits
from multi_view_world_dataset.utils.runtime import RuntimePaths
from multi_view_world_dataset.utils.serialization import dump_json, to_jsonable


def inspect_simulator_runtime(runtime: RuntimePaths, config: dict[str, Any], verify_assets: bool = True) -> dict[str, Any]:
    adapter = OmniGibsonAdapter(runtime, config)
    try:
        adapter.start()
        scenes = adapter.discover_scenes()
        nova = adapter.resolve_nova_carter_asset(verify=verify_assets)
        return {
            **adapter.runtime_report(),
            "scenes": scenes,
            "nova_carter": nova,
            "vision_modalities": [
                "rgb", "depth", "depth_linear", "normal", "seg_semantic", "seg_instance",
                "seg_instance_id", "flow", "bbox_2d_tight", "bbox_2d_loose", "bbox_3d",
                "camera_params", "pointcloud",
            ],
            "orthographic_api": "pxr.UsdGeom.Camera.GetProjectionAttr().Set(UsdGeom.Tokens.orthographic)",
        }
    finally:
        adapter.close()


def run_simulator_probe(
    runtime: RuntimePaths,
    config: dict[str, Any],
    *,
    scene_id: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    output_root = runtime.require_output()
    adapter = OmniGibsonAdapter(runtime, config)
    started = time.perf_counter()
    image_names: list[str] = []
    try:
        adapter.start()
        scenes = adapter.discover_scenes()
        if not scenes:
            raise RuntimeError("No installed BEHAVIOR scenes were discovered")
        selected_scene = scene_id or scenes[0]
        if selected_scene not in scenes:
            raise ValueError(f"Requested scene is not installed: {selected_scene}")
        probe_dir = output_root / "smoke_probe" / selected_scene
        if probe_dir.exists():
            raise FileExistsError(f"Refusing to overwrite prior smoke probe: {probe_dir}")
        splits = assign_scene_family_splits(
            scenes, config["dataset"]["splits"], int(config["dataset"]["scene_family_split_seed"])
        )
        adapter.load_scene(selected_scene, robot_count=3, development_robot=config["robot"]["development_model"])
        load_seconds = time.perf_counter() - started
        sampled_heights = adapter.place_development_robots(int(config["seed"]))

        catalog_before = adapter.object_catalog()
        snapshot = adapter.dump_snapshot()
        adapter.load_snapshot(snapshot)
        snapshot_after = adapter.dump_snapshot()
        snapshot_max_error = (
            float(np.max(np.abs(snapshot.astype(np.float64) - snapshot_after.astype(np.float64))))
            if snapshot.shape == snapshot_after.shape and snapshot.size
            else float("inf")
        )
        catalog_after = adapter.object_catalog()
        stable_ids = [obj.instance_id for obj in catalog_before] == [obj.instance_id for obj in catalog_after]

        bev_config = config["bev"]
        environment_calibration = adapter.calibrated_floor_bounds(
            0, float(bev_config["environment_meters_per_pixel"]), float(bev_config["bounds_margin_m"])
        )
        environment_bev = adapter.render_floor_bev(
            0, environment_calibration, include_robots=False, modalities=tuple(bev_config["modalities"])
        )
        world_calibration = adapter.calibrated_floor_bounds(
            0, float(bev_config["world_meters_per_pixel"]), float(bev_config["bounds_margin_m"])
        )
        world_bev = adapter.render_floor_bev(
            0, world_calibration, include_robots=True, modalities=tuple(bev_config["world_modalities"])
        )

        observations = adapter.robot_observations()
        camera_cfg = config["camera"]
        geometry_calibration = PinholeCalibration(
            camera_cfg["geometry_width"], camera_cfg["geometry_height"], camera_cfg["hfov_deg"],
            camera_cfg["near_m"], camera_cfg["far_m"],
        )
        depths: dict[str, np.ndarray] = {}
        intrinsics: dict[str, np.ndarray] = {}
        camera_poses: dict[str, np.ndarray] = {}
        for robot_id, record in observations.items():
            depth = np.asarray(record["modalities"]["depth_linear"]).squeeze()
            depths[robot_id] = depth[::2, ::2]
            intrinsics[robot_id] = geometry_calibration.pixel_intrinsics
            camera_poses[robot_id] = record["camera_to_world"]
        overlap_cfg = config["overlap"]
        overlap = build_overlap_graph(
            tuple(observations), depths, intrinsics, camera_poses,
            edge_threshold=float(overlap_cfg["edge_threshold"]),
            near_duplicate_threshold=float(overlap_cfg["near_duplicate_threshold"]),
            stride=int(overlap_cfg["depth_sample_stride"]),
            tolerance_m=float(overlap_cfg["reprojection_tolerance_m"]),
        )

        probe_dir.mkdir(parents=True)
        save_rgb(probe_dir / "b_env_rgb.png", environment_bev.modalities["rgb"])
        save_rgb(probe_dir / "b_world_t000_rgb.png", world_bev.modalities["rgb"])
        image_names.extend(["b_env_rgb.png", "b_world_t000_rgb.png"])
        for index, (robot_id, record) in enumerate(observations.items()):
            name = f"robot_{index:02d}_t000_rgb.png"
            save_rgb(probe_dir / name, record["modalities"]["rgb"])
            image_names.append(name)

        hard_checks = {
            "stable_instance_ids": stable_ids,
            "exact_snapshot_restore": snapshot_max_error <= float(config["trajectory"]["validation_position_tolerance_m"]),
            "orthographic_environment_bev": environment_bev.projection_token == "orthographic",
            "orthographic_world_bev": world_bev.projection_token == "orthographic",
            "connected_overlap_graph": overlap.connected,
            "no_near_duplicate_views": not overlap.near_duplicate_pairs,
        }
        findings: dict[str, Any] = {
            "status": "pass" if all(hard_checks.values()) else "fail",
            "hard_checks": hard_checks,
            "scene_id": selected_scene,
            "scene_split": splits[selected_scene],
            "scene_count": len(scenes),
            "load_seconds": load_seconds,
            "total_seconds": time.perf_counter() - started,
            "catalog_object_count": len(catalog_before),
            "stable_instance_ids_after_restore": stable_ids,
            "snapshot_values": int(snapshot.size),
            "snapshot_restore_max_abs_error": snapshot_max_error,
            "development_camera_heights_m": sampled_heights,
            "environment_bev": {
                "shape": [environment_calibration.height, environment_calibration.width],
                "meters_per_pixel": environment_calibration.meters_per_pixel,
                "projection": environment_bev.projection_token,
                "contains_robots": environment_bev.contains_robots,
                "modalities": sorted(environment_bev.modalities),
            },
            "world_bev": {
                "shape": [world_calibration.height, world_calibration.width],
                "meters_per_pixel": world_calibration.meters_per_pixel,
                "projection": world_bev.projection_token,
                "contains_robots": world_bev.contains_robots,
                "modalities": sorted(world_bev.modalities),
            },
            "robot_view_shapes": {
                robot_id: {name: list(np.asarray(array).shape) for name, array in record["modalities"].items()}
                for robot_id, record in observations.items()
            },
            "overlap": to_jsonable(overlap),
            "runtime": adapter.runtime_report(),
            "scope": "Gates 3/4/6/7 probe; this is not a finalized paired episode",
        }
        write_html_summary(probe_dir, "MultiViewWorldDataset smoke probe", findings, image_names)
        dump_json(output_root / "smoke_probe_last_result.json", {
            "status": findings["status"],
            "output": str(probe_dir),
            "summary": str(probe_dir / "summary.json"),
        })
        return probe_dir, findings
    except BaseException as error:
        failure = {
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_seconds": time.perf_counter() - started,
        }
        dump_json(output_root / "smoke_probe_failure.json", failure)
        dump_json(output_root / "smoke_probe_last_result.json", failure)
        print(f"MVWD_SMOKE_ERROR {failure}", flush=True)
        raise
    finally:
        adapter.close()
