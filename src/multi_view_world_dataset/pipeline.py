from __future__ import annotations

import time
import traceback
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



def _bev_geometry_metrics(render: Any) -> tuple[bool, dict[str, Any]]:
    calibration = render.calibration
    expected_shape = (calibration.height, calibration.width)
    shape_matches = {
        name: tuple(np.asarray(array).shape[:2]) == expected_shape
        for name, array in render.modalities.items()
    }
    world_width = calibration.world_bounds[2] - calibration.world_bounds[0]
    world_height = calibration.world_bounds[3] - calibration.world_bounds[1]
    horizontal = float(render.metadata.get("horizontal_aperture", float("nan")))
    vertical = float(render.metadata.get("vertical_aperture", float("nan")))
    aperture_matches = np.isclose(horizontal, 10.0 * world_width) and np.isclose(vertical, 10.0 * world_height)
    metrics = {
        "expected_shape": list(expected_shape),
        "modality_shape_matches": shape_matches,
        "world_span_m": [world_width, world_height],
        "aperture": [horizontal, vertical],
        "expected_aperture": [10.0 * world_width, 10.0 * world_height],
    }
    return bool(all(shape_matches.values()) and aperture_matches), metrics


def _segmentation_counts(render: Any, modality: str) -> dict[str, int]:
    backend_name = {"instance": "seg_instance", "instance_id": "seg_instance_id"}[modality]
    info = render.metadata.get("segmentation_info", {}).get(backend_name, {})
    if not isinstance(info, dict) or modality not in render.modalities:
        return {}
    labels = np.asarray(render.modalities[modality]).squeeze()
    counts: dict[str, int] = {}
    for raw_id, label in info.items():
        pixel_count = int(np.count_nonzero(labels == int(raw_id)))
        if pixel_count:
            counts[str(label)] = counts.get(str(label), 0) + pixel_count
    return counts


def _bev_content_metrics(environment: Any, world: Any, robot_ids: tuple[str, ...]) -> tuple[bool, dict[str, Any]]:
    environment_instances = _segmentation_counts(environment, "instance")
    environment_paths = _segmentation_counts(environment, "instance_id")
    world_paths = _segmentation_counts(world, "instance_id")
    structural = ("background", "unlabelled", "groundplane", "floors", "walls", "ceilings", "roof", "stairs")
    furniture = {
        label: pixels
        for label, pixels in environment_instances.items()
        if pixels >= 4 and not label.lower().startswith(structural) and not any(robot in label for robot in robot_ids)
    }

    def pixels_for_robot(counts: dict[str, int], robot_id: str) -> int:
        return sum(pixels for label, pixels in counts.items() if robot_id in label)

    environment_robot_pixels = {
        robot_id: pixels_for_robot(environment_paths, robot_id) for robot_id in robot_ids
    }
    world_robot_pixels = {robot_id: pixels_for_robot(world_paths, robot_id) for robot_id in robot_ids}
    passed = (
        sum(furniture.values()) >= 8
        and all(pixels == 0 for pixels in environment_robot_pixels.values())
        and all(pixels >= 4 for pixels in world_robot_pixels.values())
    )
    metrics = {
        "visible_nonstructural_instances": furniture,
        "visible_nonstructural_pixels": sum(furniture.values()),
        "environment_robot_pixels": environment_robot_pixels,
        "world_robot_pixels": world_robot_pixels,
    }
    return bool(passed), metrics


def _colorize_labels(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels).squeeze().astype(np.uint64)
    color = np.zeros(values.shape + (3,), dtype=np.uint8)
    color[..., 0] = ((values * 37 + 17) % 251).astype(np.uint8)
    color[..., 1] = ((values * 67 + 29) % 253).astype(np.uint8)
    color[..., 2] = ((values * 97 + 43) % 255).astype(np.uint8)
    color[values == 0] = 0
    return color

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
    stage = "initialize"
    try:
        stage = "launch_simulator"
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
        stage = "load_scene"
        robot_model = (
            config["robot"]["final_model"]
            if config["robot"]["use_final_robot"]
            else config["robot"]["development_model"]
        )
        adapter.load_scene(
            selected_scene, robot_count=3, development_robot=robot_model
        )
        load_seconds = time.perf_counter() - started
        stage = "place_and_settle_robots"
        sampled_heights = adapter.place_development_robots(int(config["seed"]))

        stage = "catalog_and_snapshot"
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
        stage = "render_environment_bev"
        environment_bev = adapter.render_floor_bev(
            0, environment_calibration, include_robots=False, modalities=tuple(bev_config["modalities"])
        )
        world_calibration = adapter.calibrated_floor_bounds(
            0, float(bev_config["world_meters_per_pixel"]), float(bev_config["bounds_margin_m"])
        )
        stage = "render_world_bev"
        world_bev = adapter.render_floor_bev(
            0, world_calibration, include_robots=True, modalities=tuple(bev_config["world_modalities"])
        )

        stage = "capture_robot_observations"
        observations = adapter.robot_observations()
        environment_geometry_ok, environment_geometry = _bev_geometry_metrics(environment_bev)
        world_geometry_ok, world_geometry = _bev_geometry_metrics(world_bev)
        bev_content_ok, bev_content = _bev_content_metrics(
            environment_bev, world_bev, tuple(observations)
        )
        mount_position_tolerance = float(config["trajectory"]["validation_position_tolerance_m"])
        mount_rotation_tolerance = float(config["trajectory"]["validation_rotation_tolerance_rad"])
        camera_mount_metrics = {
            robot_id: {
                "sensor_prim_path": record["sensor_prim_path"],
                "attached_to_robot": bool(record["sensor_attached_to_robot"]),
                "translation_error_m": float(record["mount_translation_error_m"]),
                "rotation_error_rad": float(record["mount_rotation_error_rad"]),
                "orthonormality_error": float(record["mount_orthonormality_error"]),
                "camera_to_base": to_jsonable(record["camera_to_base"]),
                "expected_camera_to_base": to_jsonable(record["expected_camera_to_base"]),
            }
            for robot_id, record in observations.items()
        }
        camera_mounts_ok = all(
            metrics["attached_to_robot"]
            and metrics["translation_error_m"] <= mount_position_tolerance
            and metrics["rotation_error_rad"] <= mount_rotation_tolerance
            and metrics["orthonormality_error"] <= mount_rotation_tolerance
            for metrics in camera_mount_metrics.values()
        )
        depth_metrics: dict[str, dict[str, float]] = {}
        for robot_id, record in observations.items():
            values = np.asarray(record["modalities"]["depth_linear"]).squeeze()
            valid = np.isfinite(values) & (values >= config["camera"]["near_m"]) & (values <= config["camera"]["far_m"])
            depth_metrics[robot_id] = {"valid_pixel_ratio": float(valid.mean())}
        robot_depths_ok = all(metrics["valid_pixel_ratio"] >= 0.50 for metrics in depth_metrics.values())
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
        stage = "compute_overlap"
        overlap = build_overlap_graph(
            tuple(observations), depths, intrinsics, camera_poses,
            edge_threshold=float(overlap_cfg["edge_threshold"]),
            near_duplicate_threshold=float(overlap_cfg["near_duplicate_threshold"]),
            stride=int(overlap_cfg["depth_sample_stride"]),
            tolerance_m=float(overlap_cfg["reprojection_tolerance_m"]),
        )

        stage = "write_probe_artifacts"
        probe_dir.mkdir(parents=True)
        save_rgb(probe_dir / "b_env_rgb.png", environment_bev.modalities["rgb"])
        save_rgb(probe_dir / "b_world_t000_rgb.png", world_bev.modalities["rgb"])
        image_names.extend(["b_env_rgb.png", "b_world_t000_rgb.png"])
        save_rgb(probe_dir / "b_env_instance.png", _colorize_labels(environment_bev.modalities["instance"]))
        save_rgb(probe_dir / "b_world_t000_instance_id.png", _colorize_labels(world_bev.modalities["instance_id"]))
        image_names.extend(["b_env_instance.png", "b_world_t000_instance_id.png"])
        for index, (robot_id, record) in enumerate(observations.items()):
            name = f"robot_{index:02d}_t000_rgb.png"
            save_rgb(probe_dir / name, record["modalities"]["rgb"])
            image_names.append(name)

        hard_checks = {
            "stable_instance_ids": stable_ids,
            "exact_snapshot_restore": snapshot_max_error <= float(config["trajectory"]["validation_position_tolerance_m"]),
            "orthographic_environment_bev": environment_bev.projection_token == "orthographic",
            "orthographic_world_bev": world_bev.projection_token == "orthographic",
            "environment_bev_full_coverage": environment_geometry_ok,
            "world_bev_full_coverage": world_geometry_ok,
            "environment_furniture_and_robot_mask": bev_content_ok,
            "camera_physically_parented_and_pose_stable": camera_mounts_ok,
            "robot_depth_views_valid": robot_depths_ok,
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
            "bev_geometry": {
                "environment": environment_geometry,
                "world": world_geometry,
            },
            "bev_content": bev_content,
            "camera_mounts": camera_mount_metrics,
            "robot_depth": depth_metrics,
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
            "stage": stage,
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        dump_json(output_root / "smoke_probe_failure.json", failure)
        dump_json(output_root / "smoke_probe_last_result.json", failure)
        print(f"MVWD_SMOKE_ERROR {failure}", flush=True)
        raise
    finally:
        adapter.close()
