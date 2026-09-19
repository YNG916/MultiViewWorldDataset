from __future__ import annotations

import json
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from multi_view_world_dataset.adapters.omnigibson import OmniGibsonAdapter
from multi_view_world_dataset.cameras.calibration import PinholeCalibration
from multi_view_world_dataset.cameras.overlap import (
    build_overlap_graph,
    pairwise_shared_surface_centroid,
)
from multi_view_world_dataset.cameras.transforms import invert_transform
from multi_view_world_dataset.errors import ConfigurationError, SampleRejected
from multi_view_world_dataset.pipeline import _bev_geometry_metrics
from multi_view_world_dataset.qa.checks import check_bev_pair, check_paired_trajectories, require_all
from multi_view_world_dataset.rendering.inspection import (
    save_rgb,
    save_trajectory_inspection,
    write_html_summary,
)
from multi_view_world_dataset.rendering.inspection_v11 import (
    save_environment_room_inspection,
    save_intervention_target_crops,
    save_overlap_graph_inspection,
)
from multi_view_world_dataset.sampling.configurations import near_duplicate_configuration
from multi_view_world_dataset.sampling.diversity import stable_seed, temporal_overlap_acceptance
from multi_view_world_dataset.sampling.interventions import eligible_intervention_targets
from multi_view_world_dataset.sampling.splits import assign_scene_family_splits
from multi_view_world_dataset.schema.records import (
    CameraState,
    DynamicConfiguration,
    ObjectState,
    Observation,
    InterventionType,
    QAResult,
    RobotState,
    WorldEpisode,
    WorldState,
)
from multi_view_world_dataset.storage.writer import DatasetWriter
from multi_view_world_dataset.utils.provenance import generator_source_fingerprint
from multi_view_world_dataset.utils.runtime import RuntimePaths, generator_git_commit
from multi_view_world_dataset.utils.serialization import dump_json


def _write_status(root: Path, **values: Any) -> None:
    dump_json(root / "generation_status.json", values)


def _render_environment_floors(
    adapter: OmniGibsonAdapter,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[int, Any]]:
    bev_config = config["bev"]
    arrays: dict[str, np.ndarray] = {}
    calibrations: dict[int, Any] = {}
    floor_count = len(adapter.scene_record("unassigned").floor_ids)
    for floor_index in range(floor_count):
        calibration = adapter.calibrated_floor_bounds(
            floor_index,
            float(bev_config["environment_meters_per_pixel"]),
            float(bev_config["bounds_margin_m"]),
        )
        render = adapter.render_floor_bev(
            floor_index,
            calibration,
            include_robots=False,
            modalities=tuple(bev_config["modalities"]),
        )
        geometry_ok, geometry = _bev_geometry_metrics(render)
        if not geometry_ok or render.projection_token != "orthographic":
            raise SampleRejected(
                "environment_bev_geometry_failed",
                {"floor_index": floor_index, "geometry": geometry},
            )
        occupancy = np.asarray(render.modalities["occupancy"])
        occupancy_fraction = float(np.mean(occupancy > 0))
        maximum_occupancy_fraction = float(
            bev_config["maximum_occupancy_fraction"]
        )
        if occupancy_fraction >= maximum_occupancy_fraction:
            raise SampleRejected(
                "environment_bev_occupancy_saturated",
                {
                    "floor_index": floor_index,
                    "occupancy_fraction": occupancy_fraction,
                    "maximum_occupancy_fraction": maximum_occupancy_fraction,
                },
            )
        instance_info = render.metadata.get("segmentation_info", {}).get("seg_instance_id", {})
        robot_raw_ids = [
            int(raw_id)
            for raw_id, label in instance_info.items()
            if any(f"robot_{index:02d}" in str(label) for index in range(3))
        ]
        labels = np.asarray(render.modalities["instance_id"]).squeeze()
        robot_pixels = int(np.isin(labels, robot_raw_ids).sum())
        if robot_pixels:
            raise SampleRejected(
                "environment_bev_contains_robot",
                {"floor_index": floor_index, "robot_pixels": robot_pixels},
            )
        prefix = f"floor_{floor_index:02d}"
        for name, value in render.modalities.items():
            arrays[f"{prefix}/{name}"] = np.asarray(value)
        navigation_layers = adapter.traversability_bev_layers(
            floor_index, calibration
        )
        for layer_name, layer in navigation_layers.items():
            arrays[f"{prefix}/{layer_name}"] = layer
        # Schema compatibility only. New consumers must use the explicit key.
        arrays[f"{prefix}/traversability"] = navigation_layers[
            "any_yaw_navigable"
        ]
        arrays[f"{prefix}/calibration_world_bounds"] = np.asarray(calibration.world_bounds)
        arrays[f"{prefix}/calibration_pixel_to_world"] = calibration.pixel_to_world_transform
        arrays[f"{prefix}/calibration_world_to_pixel"] = calibration.world_to_pixel_transform
        arrays[f"{prefix}/calibration_meters_per_pixel"] = np.asarray(
            calibration.meters_per_pixel
        )
        arrays[f"{prefix}/calibration_floor_z"] = np.asarray(calibration.floor_z)
        calibrations[floor_index] = calibration
    return arrays, calibrations


def _initial_overlap(
    adapter: OmniGibsonAdapter,
    config: dict[str, Any],
) -> tuple[Any, dict[str, dict[str, Any]]]:
    observations = adapter.robot_depth_observations()
    camera_config = config["camera"]
    calibration = PinholeCalibration(
        int(camera_config["geometry_width"]),
        int(camera_config["geometry_height"]),
        float(camera_config["hfov_deg"]),
        float(camera_config["near_m"]),
        float(camera_config["far_m"]),
    )
    depths = {
        robot_id: np.asarray(record["depth_linear"]).squeeze()[::2, ::2]
        for robot_id, record in observations.items()
    }
    overlap_config = config["overlap"]
    graph = build_overlap_graph(
        tuple(sorted(observations)),
        depths,
        {robot_id: calibration.pixel_intrinsics for robot_id in observations},
        {robot_id: record["camera_to_world"] for robot_id, record in observations.items()},
        edge_threshold=float(overlap_config["edge_threshold"]),
        near_duplicate_threshold=float(overlap_config["near_duplicate_threshold"]),
        stride=int(overlap_config["depth_sample_stride"]),
        tolerance_m=float(overlap_config["reprojection_tolerance_m"]),
    )
    return graph, observations



def _temporal_overlap_preflight(
    adapter: OmniGibsonAdapter,
    config: dict[str, Any],
    trajectories: tuple[Any, ...],
    *,
    catalog: tuple[ObjectState, ...] | None = None,
    requested_regime: str | None = None,
) -> dict[str, Any]:
    """Validate sparse GT overlap and target visibility before dense capture."""
    preflight = config["trajectory"]["overlap_preflight"]
    frame_count = trajectories[0].frames
    keyframe_indices = np.unique(
        np.rint(np.linspace(0, frame_count - 1, int(preflight["keyframe_count"]))).astype(int)
    )
    width = int(preflight["geometry_width"])
    height = int(preflight["geometry_height"])
    camera_config = config["camera"]
    calibration = PinholeCalibration(
        width,
        height,
        float(camera_config["hfov_deg"]),
        float(camera_config["near_m"]),
        float(camera_config["far_m"]),
    )
    overlap_config = config["overlap"]
    robot_ids = tuple(sorted(trajectory.robot_id for trajectory in trajectories))
    connected_count = 0
    isolation_runs = {robot_id: 0 for robot_id in robot_ids}
    maximum_isolation_runs = {robot_id: 0 for robot_id in robot_ids}
    keyframes: list[dict[str, Any]] = []
    try:
        for frame_index in keyframe_indices:
            adapter.place_robots_at_trajectory_frame(trajectories, int(frame_index))
            observations = adapter.robot_depth_observations()
            depths = {}
            for robot_id, record in observations.items():
                depth = np.asarray(record["depth_linear"]).squeeze()
                rows = np.rint(np.linspace(0, depth.shape[0] - 1, height)).astype(np.int64)
                columns = np.rint(np.linspace(0, depth.shape[1] - 1, width)).astype(np.int64)
                depths[robot_id] = depth[rows[:, None], columns[None, :]]
            graph = build_overlap_graph(
                robot_ids,
                depths,
                {robot_id: calibration.pixel_intrinsics for robot_id in robot_ids},
                {robot_id: observations[robot_id]["camera_to_world"] for robot_id in robot_ids},
                edge_threshold=float(overlap_config["edge_threshold"]),
                near_duplicate_threshold=float(overlap_config["near_duplicate_threshold"]),
                stride=int(preflight["depth_sample_stride"]),
                tolerance_m=float(overlap_config["reprojection_tolerance_m"]),
            )
            shared_surface_centroids_world = {}
            for left, right in graph.edges:
                centroid = pairwise_shared_surface_centroid(
                    depths[left],
                    calibration.pixel_intrinsics,
                    observations[left]["camera_to_world"],
                    depths[right],
                    calibration.pixel_intrinsics,
                    observations[right]["camera_to_world"],
                    stride=int(preflight["depth_sample_stride"]),
                    tolerance_m=float(
                        overlap_config["reprojection_tolerance_m"]
                    ),
                )
                if centroid is not None:
                    shared_surface_centroids_world[
                        f"{left}|{right}"
                    ] = centroid.tolist()
            connected_count += int(graph.connected)
            incident = {robot_id: False for robot_id in robot_ids}
            for left, right in graph.edges:
                incident[left] = True
                incident[right] = True
            isolated = []
            for robot_id in robot_ids:
                isolation_runs[robot_id] = 0 if incident[robot_id] else isolation_runs[robot_id] + 1
                maximum_isolation_runs[robot_id] = max(
                    maximum_isolation_runs[robot_id], isolation_runs[robot_id]
                )
                if not incident[robot_id]:
                    isolated.append(robot_id)
            keyframes.append(
                {
                    "frame_index": int(frame_index),
                    "connected": bool(graph.connected),
                    "edges": [list(edge) for edge in graph.edges],
                    "shared_surface_centroids_world": (
                        shared_surface_centroids_world
                    ),
                    "isolated_robot_ids": isolated,
                    "near_duplicate_pairs": [list(pair) for pair in graph.near_duplicate_pairs],
                    "overlap_matrix": [
                        [
                            1.0 if left == right else float(
                                graph.overlaps.get(tuple(sorted((left, right))), 0.0)
                            )
                            for right in robot_ids
                        ]
                        for left in robot_ids
                    ],
                    "overlaps": {
                        f"{left}|{right}": float(value)
                        for (left, right), value in graph.overlaps.items()
                    },
                }
            )
    finally:
        adapter.place_robots_at_trajectory_frame(trajectories, 0)
    regime = str(trajectories[0].metadata.get("observation_regime", "partial_chain"))
    metrics = temporal_overlap_acceptance(
        robot_ids,
        keyframes,
        regime=regime,
        regime_connected_fraction_target=preflight["regime_connected_fraction_target"],
        regime_shared_keyframe_fraction_target=preflight[
            "regime_shared_keyframe_fraction_target"
        ],
        regime_maximum_consecutive_isolated_keyframes=preflight[
            "regime_maximum_consecutive_isolated_keyframes"
        ],
    )
    metrics.update({
        "keyframe_indices": keyframe_indices.tolist(),
        "geometry_resolution": [width, height],
        "robot_ids": list(robot_ids),
        "keyframes": keyframes,
    })
    if requested_regime is not None:
        metrics["requested_regime"] = str(requested_regime)
        metrics["regime_target_match"] = bool(
            metrics["realized_regime"] == requested_regime
        )
    if not metrics["passed"]:
        raise SampleRejected("trajectory_temporal_overlap_failed", metrics)
    if catalog is not None:
        sparse_instance_views: dict[str, dict[str, list[np.ndarray]]] = {
            robot_id: {"instance": []} for robot_id in robot_ids
        }
        try:
            for frame_index in keyframe_indices:
                adapter.place_robots_at_trajectory_frame(
                    trajectories, int(frame_index)
                )
                observations = adapter.robot_observations()
                for robot_id, record in observations.items():
                    modalities = record["modalities"]
                    instance = modalities.get(
                        "seg_instance_id", modalities.get("seg_instance")
                    )
                    if instance is None:
                        raise SampleRejected(
                            "intervention_visibility_preflight_missing_instance",
                            {"robot_id": robot_id, "frame_index": int(frame_index)},
                        )
                    sparse_instance_views[robot_id]["instance"].append(
                        np.asarray(instance).squeeze()
                    )
        finally:
            adapter.place_robots_at_trajectory_frame(trajectories, 0)
        visibility = _sparse_intervention_visibility_preflight(
            catalog,
            {
                robot_id: {"instance": np.stack(values["instance"], axis=0)}
                for robot_id, values in sparse_instance_views.items()
            },
            config,
            sampled_frame_count=len(keyframe_indices),
            full_frame_count=frame_count,
        )
        metrics["intervention_visibility_preflight"] = visibility
        if not visibility["passed"]:
            raise SampleRejected(
                "no_visible_intervention_target_preflight", visibility
            )
    return metrics


def _soft_requested_overlap_regime(
    weights: dict[str, float],
    global_counts: Counter[str],
    split_counts: Counter[str],
) -> str:
    """Prefer current global/split deficits without making them hard gates."""
    global_total = sum(global_counts.values()) + 1
    split_total = sum(split_counts.values()) + 1
    return max(
        sorted(weights),
        key=lambda name: (
            global_total * float(weights[name]) - global_counts[name]
            + split_total * float(weights[name]) - split_counts[name]
        ),
    )
def _robot_states(
    config: dict[str, Any],
    heights: dict[str, float],
    trajectories: tuple[Any, ...],
) -> tuple[RobotState, ...]:
    by_id = {trajectory.robot_id: trajectory for trajectory in trajectories}
    model = (
        config["robot"]["final_model"]
        if config["robot"]["use_final_robot"]
        else config["robot"]["development_model"]
    )
    return tuple(
        RobotState(
            robot_id=robot_id,
            model=model,
            base_to_world=by_id[robot_id].base_to_world[0],
            camera_height_m=float(heights[robot_id]),
        )
        for robot_id in sorted(by_id)
    )


def _observation_records(
    config: dict[str, Any],
    trajectories: tuple[Any, ...],
    branch: str,
    view_refs: dict[str, str],
    capture_metadata: dict[str, dict[str, np.ndarray]],
) -> tuple[Observation, ...]:
    camera_config = config["camera"]
    rgb_calibration = PinholeCalibration(
        int(camera_config["rgb_width"]), int(camera_config["rgb_height"]),
        float(camera_config["hfov_deg"]), float(camera_config["near_m"]),
        float(camera_config["far_m"]),
    )
    geometry_calibration = PinholeCalibration(
        int(camera_config["geometry_width"]), int(camera_config["geometry_height"]),
        float(camera_config["hfov_deg"]), float(camera_config["near_m"]),
        float(camera_config["far_m"]),
    )
    records: list[Observation] = []
    for trajectory in trajectories:
        for frame_index in range(trajectory.frames):
            base_to_world = trajectory.base_to_world[frame_index]
            camera_to_world = trajectory.camera_to_world[frame_index]
            camera_to_base = invert_transform(base_to_world) @ camera_to_world
            capture = capture_metadata[trajectory.robot_id]
            mounted_camera_to_world = capture["mounted_camera_to_world"][frame_index]
            capture_camera_to_world = capture["capture_camera_to_world"][frame_index]
            modality_refs = {
                modality: f"{view_refs[trajectory.robot_id]}::{modality}[{frame_index}]"
                for modality in ("rgb", "depth_linear", "semantic", "instance", "normal")
            }
            records.append(
                Observation(
                    robot_id=trajectory.robot_id,
                    physical_time_index=frame_index,
                    camera=CameraState(
                        camera_id=f"{trajectory.robot_id}_{branch}",
                        robot_id=trajectory.robot_id,
                        width=int(camera_config["rgb_width"]),
                        height=int(camera_config["rgb_height"]),
                        pixel_intrinsics=rgb_calibration.pixel_intrinsics,
                        normalized_intrinsics=rgb_calibration.normalized_intrinsics,
                        camera_to_world=camera_to_world,
                        world_to_camera=invert_transform(camera_to_world),
                        robot_base_to_world=base_to_world,
                        camera_to_robot_base=camera_to_base,
                        near_m=float(camera_config["near_m"]),
                        far_m=float(camera_config["far_m"]),
                        camera_height_m=float(camera_to_base[2, 3]),
                        geometry_width=int(camera_config["geometry_width"]),
                        geometry_height=int(camera_config["geometry_height"]),
                        geometry_pixel_intrinsics=geometry_calibration.pixel_intrinsics,
                        geometry_normalized_intrinsics=geometry_calibration.normalized_intrinsics,
                        mounted_camera_to_world=mounted_camera_to_world,
                        capture_camera_to_world=capture_camera_to_world,
                        modality_camera_to_world={
                            "rgb": mounted_camera_to_world,
                            "depth_linear": mounted_camera_to_world,
                            "semantic": capture_camera_to_world,
                            "instance": capture_camera_to_world,
                            "normal": capture_camera_to_world,
                        },
                        capture_pose_translation_error_m=float(
                            capture["capture_pose_translation_error_m"][frame_index]
                        ),
                        capture_pose_rotation_error_rad=float(
                            capture["capture_pose_rotation_error_rad"][frame_index]
                        ),
                        mast_joint_value_m=float(camera_to_base[2, 3])
                        - float(min(camera_config["heights_m"])),
                    ),
                    modality_refs=modality_refs,
                )
            )
    return tuple(records)


def _existing_event_targets(root: Path, scene_id: str, configuration_id: str) -> tuple[str, ...]:
    episode_root = root / "episodes" / scene_id / configuration_id
    if not episode_root.is_dir():
        return ()
    targets = []
    for path in sorted(episode_root.glob("*/events.json")):
        try:
            events = json.loads(path.read_text(encoding="utf-8"))
            if events:
                targets.append(str(events[0]["target_instance_id"]))
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return tuple(targets)


def _existing_event_types(root: Path, scene_id: str) -> dict[str, int]:
    counts = {item.value: 0 for item in InterventionType}
    for path in (root / "episodes" / scene_id).glob("*/*/events.json"):
        try:
            events = json.loads(path.read_text(encoding="utf-8"))
            if events:
                counts[str(events[0]["intervention_type"])] += 1
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return counts


def _choose_quota_intervention_type(
    weights: dict[str, float], counts: dict[str, int], available: set[InterventionType], seed: int
) -> InterventionType:
    if not available:
        raise SampleRejected("no_visible_target_for_any_intervention_type")
    total_after = sum(counts.values()) + 1
    rng = np.random.default_rng(seed)
    ranked = []
    for intervention_type in available:
        deficit = float(weights[intervention_type.value]) * total_after - counts[intervention_type.value]
        ranked.append((deficit + float(rng.uniform(0, 1e-9)), intervention_type))
    return max(ranked, key=lambda item: item[0])[1]


def _read_configuration_catalog(path: Path) -> tuple[ObjectState, ...]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Resuming configuration generation requires pyarrow") from error
    return tuple(ObjectState(**row) for row in pq.read_table(path).to_pylist())


def _intervention_visibility_table(
    catalog: tuple[ObjectState, ...],
    robot_views: dict[str, dict[str, np.ndarray]],
    config: dict[str, Any],
) -> dict[str, Any]:
    requirements = config["intervention"]["target_visibility"]
    minimum_pixels = int(requirements["minimum_pixels"])
    minimum_frames = int(requirements["minimum_frames"])
    minimum_robots = int(requirements["minimum_robots"])
    preferred_robots = int(requirements["preferred_robots"])
    table: dict[str, Any] = {}
    eligible: list[tuple[tuple[int, int, int], str]] = []
    for public_id, obj in enumerate(sorted(catalog, key=lambda item: item.instance_id), start=4):
        per_robot = {}
        total_frames = 0
        peak_pixels = 0
        participating = 0
        for robot_id, modalities in sorted(robot_views.items()):
            masks = np.asarray(modalities["instance"]) == public_id
            counts = masks.reshape(masks.shape[0], -1).sum(axis=1)
            qualifying = int(np.count_nonzero(counts >= minimum_pixels))
            per_robot[robot_id] = {
                "qualifying_frame_count": qualifying,
                "maximum_pixels": int(counts.max(initial=0)),
            }
            total_frames += qualifying
            peak_pixels = max(peak_pixels, int(counts.max(initial=0)))
            participating += int(qualifying > 0)
        accepted = total_frames >= minimum_frames and participating >= minimum_robots
        table[obj.instance_id] = {
            "public_instance_id": public_id,
            "category": obj.category,
            "qualifying_frame_count": total_frames,
            "participating_robot_count": participating,
            "preferred_multi_robot_visibility": participating >= preferred_robots,
            "maximum_pixels": peak_pixels,
            "per_robot": per_robot,
            "accepted": accepted,
        }
        if accepted:
            eligible.append(((int(participating >= preferred_robots), participating, total_frames), obj.instance_id))
    eligible.sort(reverse=True)
    return {
        "requirements": dict(requirements),
        "eligible_target_ids": [instance_id for _, instance_id in eligible],
        "objects": table,
    }


def _sparse_intervention_visibility_preflight(
    catalog: tuple[ObjectState, ...],
    robot_views: dict[str, dict[str, np.ndarray]],
    config: dict[str, Any],
    *,
    sampled_frame_count: int,
    full_frame_count: int,
) -> dict[str, Any]:
    """Cheaply reject routes with no visible schema-eligible intervention target."""
    requirements = config["intervention"]["target_visibility"]
    minimum_pixels = int(requirements["minimum_pixels"])
    minimum_robots = int(requirements["minimum_robots"])
    minimum_frames = max(
        1,
        int(np.ceil(
            int(requirements["minimum_frames"])
            * sampled_frame_count
            / max(1, full_frame_count)
        )),
    )
    eligible_types: dict[str, list[str]] = {}
    for intervention_type in InterventionType:
        for obj in eligible_intervention_targets(catalog, intervention_type):
            eligible_types.setdefault(obj.instance_id, []).append(
                intervention_type.value
            )
    table: dict[str, Any] = {}
    accepted_ids: list[str] = []
    for public_id, obj in enumerate(
        sorted(catalog, key=lambda item: item.instance_id), start=4
    ):
        if obj.instance_id not in eligible_types:
            continue
        per_robot: dict[str, Any] = {}
        total_frames = 0
        participating = 0
        peak_pixels = 0
        for robot_id, modalities in sorted(robot_views.items()):
            masks = np.asarray(modalities["instance"]) == public_id
            counts = masks.reshape(masks.shape[0], -1).sum(axis=1)
            qualifying = int(np.count_nonzero(counts >= minimum_pixels))
            maximum = int(counts.max(initial=0))
            per_robot[robot_id] = {
                "qualifying_frame_count": qualifying,
                "maximum_pixels": maximum,
            }
            total_frames += qualifying
            participating += int(qualifying > 0)
            peak_pixels = max(peak_pixels, maximum)
        accepted = (
            total_frames >= minimum_frames
            and participating >= minimum_robots
        )
        table[obj.instance_id] = {
            "public_instance_id": public_id,
            "category": obj.category,
            "eligible_intervention_types": eligible_types[obj.instance_id],
            "qualifying_frame_count": total_frames,
            "participating_robot_count": participating,
            "maximum_pixels": peak_pixels,
            "per_robot": per_robot,
            "accepted": accepted,
        }
        if accepted:
            accepted_ids.append(obj.instance_id)
    minimum_candidates = int(
        config["navigation"]["minimum_visible_intervention_candidates"]
    )
    return {
        "passed": len(accepted_ids) >= minimum_candidates,
        "sampled_frame_count": sampled_frame_count,
        "full_frame_count": full_frame_count,
        "requirements": {
            "minimum_pixels": minimum_pixels,
            "minimum_sparse_frames": minimum_frames,
            "minimum_robots": minimum_robots,
            "minimum_visible_intervention_candidates": minimum_candidates,
        },
        "eligible_target_ids": accepted_ids,
        "objects": table,
    }


def _post_render_intervention_effect(
    target_instance_id: str,
    before_catalog: tuple[ObjectState, ...],
    before_views: dict[str, dict[str, np.ndarray]],
    after_views: dict[str, dict[str, np.ndarray]],
    config: dict[str, Any],
) -> dict[str, Any]:
    ordered = sorted(before_catalog, key=lambda item: item.instance_id)
    public_id = 4 + next(index for index, obj in enumerate(ordered) if obj.instance_id == target_instance_id)
    effect = config["intervention"]["post_render_effect"]
    delta_threshold = float(effect["minimum_mean_rgb_delta"])
    changed_pixels = 0
    union_pixels = 0
    delta_sum = 0.0
    per_robot = {}
    for robot_id in sorted(before_views):
        before_mask = np.asarray(before_views[robot_id]["instance"]) == public_id
        after_mask = np.asarray(after_views[robot_id]["instance"]) == public_id
        before_rgb = np.asarray(before_views[robot_id]["rgb"])[..., :3].astype(np.float32)
        after_rgb = np.asarray(after_views[robot_id]["rgb"])[..., :3].astype(np.float32)
        rows = np.rint(np.linspace(0, before_rgb.shape[1] - 1, before_mask.shape[1])).astype(int)
        cols = np.rint(np.linspace(0, before_rgb.shape[2] - 1, before_mask.shape[2])).astype(int)
        rgb_delta = np.mean(np.abs(before_rgb[:, rows[:, None], cols[None, :]] - after_rgb[:, rows[:, None], cols[None, :]]), axis=-1)
        union = before_mask.squeeze() | after_mask.squeeze()
        silhouette = before_mask.squeeze() ^ after_mask.squeeze()
        changed = union & ((rgb_delta >= delta_threshold) | silhouette)
        robot_changed = int(changed.sum())
        robot_union = int(union.sum())
        changed_pixels += robot_changed
        union_pixels += robot_union
        delta_sum += float(rgb_delta[union].sum())
        per_robot[robot_id] = {"changed_pixels": robot_changed, "union_pixels": robot_union}
    mean_delta = delta_sum / max(1, union_pixels)
    checks = {
        "minimum_changed_pixels": changed_pixels >= int(effect["minimum_changed_pixels"]),
        "minimum_mean_rgb_delta": mean_delta >= delta_threshold,
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "target_public_instance_id": public_id,
        "changed_pixels": changed_pixels, "union_pixels": union_pixels,
        "mean_rgb_delta": mean_delta, "per_robot": per_robot,
    }
def _sibling_episode_diversity(
    root: Path,
    scene_id: str,
    configuration_id: str,
    trajectory_metrics: dict[str, Any],
    trajectories: tuple[Any, ...],
    intervention_type: str,
    target_instance_id: str,
) -> dict[str, Any]:
    """Compare an accepted episode with already finalized configuration siblings."""
    siblings_root = root / "episodes" / scene_id / configuration_id
    prior_episode_ids: list[str] = []
    prior_regions: set[str] = set()
    prior_path_families: set[str] = set()
    prior_regimes: set[str] = set()
    prior_targets: set[str] = set()
    prior_types: list[str] = []
    layout_distances: list[float] = []
    current_starts = {
        trajectory.robot_id: np.asarray(
            trajectory.base_to_world[0, :2, 3], dtype=np.float64
        )
        for trajectory in trajectories
    }
    if siblings_root.is_dir():
        for sibling in sorted(siblings_root.glob("episode_*")):
            metrics_path = sibling / "generation_metrics.json"
            events_path = sibling / "events.json"
            trajectories_path = sibling / "trajectories.npz"
            if not metrics_path.is_file() or not events_path.is_file():
                continue
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                events = json.loads(events_path.read_text(encoding="utf-8"))
                prior_episode_ids.append(sibling.name)
                sibling_trajectory = metrics["trajectory"]
                prior_regions.update(
                    map(str, sibling_trajectory.get("start_region_ids", []))
                )
                prior_regimes.add(
                    str(sibling_trajectory.get("observation_regime", "unknown"))
                )
                prior_path_families.update(
                    str(robot["path_family"])
                    for robot in sibling_trajectory.get("robots", {}).values()
                )
                if events:
                    prior_targets.add(str(events[0]["target_instance_id"]))
                    prior_types.append(str(events[0]["intervention_type"]))
                if trajectories_path.is_file():
                    with np.load(trajectories_path, allow_pickle=False) as arrays:
                        errors = [
                            np.linalg.norm(
                                current_starts[robot_id]
                                - np.asarray(
                                    arrays[f"{robot_id}_base_to_world"][
                                        0, :2, 3
                                    ],
                                    dtype=np.float64,
                                )
                            )
                            for robot_id in sorted(current_starts)
                        ]
                    layout_distances.append(
                        float(np.sqrt(np.mean(np.square(errors))))
                    )
            except (KeyError, OSError, ValueError, json.JSONDecodeError):
                continue
    current_regions = set(map(str, trajectory_metrics.get("start_region_ids", [])))
    current_families = {
        str(robot["path_family"])
        for robot in trajectory_metrics.get("robots", {}).values()
    }
    current_regime = str(
        trajectory_metrics.get("observation_regime", "unknown")
    )
    return {
        "prior_episode_ids": prior_episode_ids,
        "prior_episode_count": len(prior_episode_ids),
        "new_start_region_ids": sorted(current_regions - prior_regions),
        "start_region_reuse_fraction": (
            len(current_regions & prior_regions) / max(1, len(current_regions))
        ),
        "new_path_families": sorted(current_families - prior_path_families),
        "realized_regime_novel": current_regime not in prior_regimes,
        "intervention_target_novel": target_instance_id not in prior_targets,
        "intervention_type_novel": intervention_type not in prior_types,
        "minimum_prior_layout_rms_distance_m": (
            min(layout_distances) if layout_distances else None
        ),
    }




def generate_dataset(
    runtime: RuntimePaths,
    config: dict[str, Any],
    *,
    scene_id: str | None = None,
    allow_large: bool = False,
) -> tuple[Path, dict[str, Any]]:
    profile = str(config["profile"])
    if profile not in {"smoke", "integration"} and not allow_large:
        raise ConfigurationError(
            "Refusing large pilot/default generation without explicit --allow-large"
        )
    root = runtime.require_output()
    repository_root = Path(__file__).resolve().parents[2]
    commit = generator_git_commit(repository_root)
    source_fingerprint = generator_source_fingerprint(repository_root)
    writer = DatasetWriter(root)
    writer.initialize(
        {
            "schema_version": config["dataset"]["schema_version"],
            "dataset_semantics": "Dataset-v1.1",
            "profile": profile,
            "seed": int(config["seed"]),
            "generator_git_commit": commit,
            "generator_source_fingerprint": source_fingerprint,
            "source_of_truth": "world_state+simulator_snapshot+trajectory+event_log",
            "coordinate_conventions": {
                "world": "right-handed Z-up",
                "camera": "OpenCV x-right y-down z-forward",
                "poses": "local-to-world homogeneous matrices",
                "depth_linear": "metric camera-forward depth in meters",
            },
            "bev_conventions": {
                "occupancy": "observed geometry above floor; not traversability",
                "point_traversability": "floor-supported point navigability before robot footprint",
                "any_yaw_navigable": "at least one exact footprint yaw is collision-free",
                "yaw_freedom": "fraction of discretized footprint yaw bins collision-free",
                "traversability": "deprecated alias of any_yaw_navigable",
                "environment_resolution_mpp": float(config["bev"]["environment_meters_per_pixel"]),
                "world_resolution_mpp": float(config["bev"]["world_meters_per_pixel"]),
            },
            "intervention_taxonomy": sorted(config["intervention"]["type_weights"]),
            "robot_count": 3,
            "physical_frames": int(config["dataset"]["frames"]),
            "fps": float(config["dataset"]["fps"]),
            "camera_specification": config["camera"],
            "bev_specification": config["bev"],
            "modalities": {
                "robot_views": [
                    "rgb",
                    "depth_linear",
                    "semantic",
                    "instance",
                    "normal",
                ],
                "environment_bev": config["bev"]["modalities"],
                "world_bev": config["bev"]["world_modalities"],
            },
            "split_policy": {
                "unit": "base_scene_or_scene_family",
                "weights": config["dataset"]["splits"],
                "seed": int(
                    config["dataset"]["scene_family_split_seed"]
                ),
            },
            "intervention_policy": config["intervention"],
            "overlap_policy": {
                "pairwise": config["overlap"],
                "temporal_preflight": (
                    config["trajectory"]["overlap_preflight"]
                ),
            },
            "sampling_regime_configuration": {
                "weights": (
                    config["placement"]["observation_regime_weights"]
                ),
            },
        },
        resolved_config=config,
    )
    adapter = OmniGibsonAdapter(runtime, config)
    accepted_configurations = 0
    accepted_episodes = 0
    requested_regime_counts: Counter[str] = Counter()
    realized_regime_counts: Counter[str] = Counter()
    requested_regime_counts_by_split: dict[str, Counter[str]] = {}
    realized_regime_counts_by_split: dict[str, Counter[str]] = {}
    try:
        _write_status(root, status="running", stage="launch_simulator", profile=profile)
        adapter.start()
        writer.update_dataset_metadata(
            {
                "simulator_versions": (
                    adapter.runtime_report().get("versions", {})
                )
            }
        )
        scenes = adapter.discover_scenes()
        if scene_id is not None:
            if scene_id not in scenes:
                raise ConfigurationError(f"Requested scene is not installed: {scene_id}")
            scenes = [scene_id]
        scene_limit = config["dataset"].get("scene_limit")
        if scene_limit is not None:
            scenes = scenes[: int(scene_limit)]
        splits = assign_scene_family_splits(
            adapter.discover_scenes(),
            config["dataset"]["splits"],
            int(config["dataset"]["scene_family_split_seed"]),
        )
        requested_configurations = int(config["dataset"]["accepted_configurations_per_scene"])
        requested_episodes = int(config["dataset"]["accepted_episodes_per_configuration"])
        for scene_position, selected_scene in enumerate(scenes):
            _write_status(
                root,
                status="running",
                stage="load_scene",
                scene_id=selected_scene,
                scene_position=scene_position,
                accepted_configurations=accepted_configurations,
                accepted_episodes=accepted_episodes,
            )
            adapter.load_scene(
                selected_scene,
                robot_count=3,
                development_robot=(
                    config["robot"]["final_model"]
                    if config["robot"]["use_final_robot"]
                    else config["robot"]["development_model"]
                ),
            )
            base_snapshot = adapter.dump_snapshot()
            base_catalog = adapter.object_catalog_with_relations()
            writer.update_scene_taxonomy(selected_scene, base_catalog)
            scene_root = root / "scenes" / selected_scene
            if not (scene_root / "scene_meta.json").is_file():
                writer.write_scene(
                    selected_scene,
                    adapter.scene_record(splits[selected_scene]),
                    base_catalog,
                )
            completed_configurations = set(writer.completed_configuration_ids(selected_scene))
            accepted_catalogs: list[tuple[Any, ...]] = []
            accepted_hashes: set[str] = set()
            accepted_intervention_types = _existing_event_types(root, selected_scene)
            for configuration_id in completed_configurations:
                meta_path = (
                    root
                    / "configurations"
                    / selected_scene
                    / configuration_id
                    / "config_meta.json"
                )
                try:
                    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                    accepted_hashes.add(str(metadata["exact_state_hash"]))
                    accepted_catalogs.append(
                        _read_configuration_catalog(meta_path.with_name("object_catalog.parquet"))
                    )
                except (OSError, KeyError, json.JSONDecodeError):
                    pass
            expected_configuration_ids = {
                f"config_{index:03d}" for index in range(requested_configurations)
            }
            accepted_configurations += len(
                completed_configurations & expected_configuration_ids
            )
            for configuration_index in range(requested_configurations):
                configuration_id = f"config_{configuration_index:03d}"
                configuration_root = (
                    root / "configurations" / selected_scene / configuration_id
                )
                if configuration_id not in completed_configurations:
                    accepted = None
                    for attempt in range(int(config["generation"]["maximum_configuration_attempts"])):
                        seed = stable_seed(
                            int(config["seed"]), selected_scene,
                            configuration_id, "configuration", attempt,
                        )
                        _write_status(
                            root,
                            status="running",
                            stage="sample_configuration",
                            scene_id=selected_scene,
                            configuration_id=configuration_id,
                            attempt=attempt,
                            accepted_configurations=accepted_configurations,
                            accepted_episodes=accepted_episodes,
                        )
                        adapter.load_snapshot(base_snapshot)
                        try:
                            candidate = adapter.randomize_relation_preserving_configuration(seed)
                            if candidate["exact_state_hash"] in accepted_hashes:
                                raise SampleRejected("exact_duplicate_configuration")
                            if any(
                                near_duplicate_configuration(
                                    candidate["catalog"],
                                    catalog,
                                    translation_threshold_m=float(
                                        config["generation"]["near_duplicate_translation_m"]
                                    ),
                                    rotation_threshold_deg=float(
                                        config["generation"]["near_duplicate_rotation_deg"]
                                    ),
                                )
                                for catalog in accepted_catalogs
                            ):
                                raise SampleRejected("near_duplicate_configuration")
                            adapter.prepare_navigation_context(
                                str(candidate["exact_state_hash"]),
                                stable_seed(seed, "configuration-navigation-context"),
                                force=True,
                            )
                            navigation_metadata = adapter.navigation_context_metadata()
                            environment_arrays, _ = _render_environment_floors(adapter, config)
                            world_state = WorldState(
                                scene_id=selected_scene,
                                configuration_id=configuration_id,
                                physical_time_index=None,
                                objects=candidate["catalog"],
                                simulator_snapshot_ref="simulator_state.npy",
                            )
                            configuration = DynamicConfiguration(
                                configuration_id=configuration_id,
                                scene_id=selected_scene,
                                seed=seed,
                                exact_state_hash=candidate["exact_state_hash"],
                                world_state=world_state,
                                environment_bev_ref="bev/environment_base.npz",
                                simulator_snapshot_ref="simulator_state.npy",
                                accepted_attempt=attempt,
                                metadata={
                                    "baseline_exact_state_hash": candidate.get(
                                        "baseline_exact_state_hash"
                                    ),
                                    "changed_instance_ids": candidate.get(
                                        "changed_instance_ids", []
                                    ),
                                    "changed_object_count": int(
                                        candidate.get(
                                            "changed_object_count",
                                            len(
                                                candidate.get(
                                                    "changed_instance_ids", []
                                                )
                                            ),
                                        )
                                    ),
                                    "requested_changed_object_count": int(
                                        candidate.get(
                                            "requested_changed_object_count",
                                            len(
                                                candidate.get(
                                                    "changed_instance_ids", []
                                                )
                                            ),
                                        )
                                    ),
                                    "changes": candidate.get(
                                        "changes",
                                        [candidate.get("relation", {})],
                                    ),
                                    "stratification": candidate.get(
                                        "stratification", {}
                                    ),
                                    "configuration_checks": candidate.get(
                                        "checks", {}
                                    ),
                                    "maximum_snapshot_restore_error": (
                                        candidate.get(
                                            "maximum_snapshot_restore_error"
                                        )
                                    ),
                                    "randomization_metrics": {
                                        key: candidate[key]
                                        for key in (
                                            "translation_m",
                                            "rotation_deg",
                                            "free_traversable_candidates",
                                            "intervention_target_count",
                                        )
                                        if key in candidate
                                    },
                                    "navigation_context_ref": "navigation_context.json",
                                },
                            )
                            writer.write_configuration(
                                configuration,
                                candidate["catalog"],
                                snapshot=candidate["snapshot"],
                                environment_bev=environment_arrays,
                            )
                            dump_json(
                                configuration_root / "navigation_context.json",
                                navigation_metadata,
                            )
                            accepted = candidate
                            accepted_catalogs.append(candidate["catalog"])
                            accepted_hashes.add(candidate["exact_state_hash"])
                            accepted_configurations += 1
                            break
                        except SampleRejected as error:
                            writer.record_reject(
                                f"configuration:{selected_scene}/{configuration_id}",
                                error.reason,
                                {"attempt": attempt, **error.details},
                            )
                    if accepted is None:
                        raise SampleRejected(
                            "configuration_attempts_exhausted",
                            {"scene_id": selected_scene, "configuration_id": configuration_id},
                        )
                snapshot_path = configuration_root / "simulator_state.npy"
                configuration_snapshot = np.load(snapshot_path, allow_pickle=False)
                configuration_metadata = json.loads(
                    (configuration_root / "config_meta.json").read_text(
                        encoding="utf-8"
                    )
                )
                configuration_token = str(
                    configuration_metadata["exact_state_hash"]
                )
                adapter.load_snapshot(configuration_snapshot)
                adapter.prepare_navigation_context(
                    configuration_token,
                    stable_seed(
                        int(config["seed"]),
                        selected_scene,
                        configuration_id,
                        "configuration-navigation-context",
                    ),
                )
                navigation_metadata_path = (
                    configuration_root / "navigation_context.json"
                )
                if not navigation_metadata_path.is_file():
                    dump_json(
                        navigation_metadata_path,
                        adapter.navigation_context_metadata(),
                    )
                existing_episodes = set(
                    writer.completed_episode_ids(selected_scene, configuration_id)
                )
                accepted_episodes += len(existing_episodes)
                used_targets = set(
                    _existing_event_targets(root, selected_scene, configuration_id)
                )
                used_regions: set[str] = set()
                for existing_episode in existing_episodes:
                    metrics_path = root / "episodes" / selected_scene / configuration_id / existing_episode / "generation_metrics.json"
                    try:
                        prior_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                        used_regions.update(prior_metrics["trajectory"].get("start_region_ids", []))
                        prior_trajectory = prior_metrics["trajectory"]
                        prior_requested = str(prior_trajectory.get(
                            "requested_observation_regime", "unclassified"
                        ))
                        prior_realized = str(prior_trajectory.get(
                            "observation_regime", "unclassified"
                        ))
                        requested_regime_counts[prior_requested] += 1
                        realized_regime_counts[prior_realized] += 1
                        requested_regime_counts_by_split.setdefault(
                            splits[selected_scene], Counter()
                        )[prior_requested] += 1
                        realized_regime_counts_by_split.setdefault(
                            splits[selected_scene], Counter()
                        )[prior_realized] += 1
                    except (OSError, KeyError, json.JSONDecodeError):
                        pass
                for episode_index in range(requested_episodes):
                    episode_id = f"episode_{episode_index:03d}"
                    if episode_id in existing_episodes:
                        continue
                    episode_seed = stable_seed(
                        int(config["seed"]), selected_scene,
                        configuration_id, episode_id,
                    )
                    before = None
                    graph = None
                    heights = None
                    trajectories = None
                    w0_snapshot = None
                    w0_catalog = None
                    trajectory_metrics = None
                    temporal_overlap_metrics = None
                    visibility_table = None
                    split_name = splits[selected_scene]
                    requested_overlap_regime = _soft_requested_overlap_regime(
                        config["placement"]["observation_regime_weights"],
                        requested_regime_counts,
                        requested_regime_counts_by_split.setdefault(
                            split_name, Counter()
                        ),
                    )
                    for placement_attempt in range(int(config["placement"]["maximum_attempts"])):
                        _write_status(
                            root,
                            status="running",
                            stage="sample_episode_before",
                            scene_id=selected_scene,
                            configuration_id=configuration_id,
                            episode_id=episode_id,
                            attempt=placement_attempt,
                            accepted_configurations=accepted_configurations,
                            accepted_episodes=accepted_episodes,
                        )
                        adapter.load_snapshot(configuration_snapshot)
                        try:
                            heights, base_trajectory_candidates = (
                                adapter.sample_route_first_trajectory_sets(
                                    stable_seed(episode_seed, "route-first", placement_attempt),
                                    discouraged_region_ids=tuple(sorted(used_regions)),
                                )
                            )
                            trajectory_candidates = list(base_trajectory_candidates)
                            candidate_failures: list[dict[str, Any]] = []
                            candidate_rank = 0
                            while candidate_rank < len(trajectory_candidates):
                                candidate_trajectories, candidate_metrics = (
                                    trajectory_candidates[candidate_rank]
                                )
                                try:
                                    adapter.place_robots_at_trajectory_frame(
                                        candidate_trajectories, 0
                                    )
                                    candidate_graph, _ = _initial_overlap(adapter, config)
                                    candidate_catalog = (
                                        adapter.object_catalog_with_relations()
                                    )
                                    candidate_overlap = _temporal_overlap_preflight(
                                        adapter,
                                        config,
                                        candidate_trajectories,
                                        catalog=candidate_catalog,
                                        requested_regime=requested_overlap_regime,
                                    )
                                    requested_regime = str(
                                        candidate_overlap["requested_regime"]
                                    )
                                    realized_regime = str(
                                        candidate_overlap["realized_regime"]
                                    )
                                    candidate_metrics["requested_observation_regime"] = (
                                        requested_regime
                                    )
                                    candidate_metrics["observation_regime"] = realized_regime
                                    for candidate_trajectory in candidate_trajectories:
                                        candidate_trajectory.metadata[
                                            "requested_observation_regime"
                                        ] = requested_regime
                                        candidate_trajectory.metadata[
                                            "observation_regime"
                                        ] = realized_regime
                                    candidate_metrics["temporal_overlap"] = candidate_overlap
                                    candidate_metrics["overlap_regime_distribution"] = {
                                        "target_weights": config["placement"]["observation_regime_weights"],
                                        "global_requested_before_accept": dict(requested_regime_counts),
                                        "global_realized_before_accept": dict(realized_regime_counts),
                                        "split": split_name,
                                        "split_requested_before_accept": dict(
                                            requested_regime_counts_by_split[split_name]
                                        ),
                                        "split_realized_before_accept": dict(
                                            realized_regime_counts_by_split.setdefault(
                                                split_name, Counter()
                                            )
                                        ),
                                        "selection_policy": "soft_global_plus_split_deficit_preference",
                                    }
                                    candidate_metrics["temporal_preflight_candidate_rank"] = (
                                        candidate_rank
                                    )
                                    candidate_calibration = adapter.calibrated_floor_bounds(
                                        int(candidate_metrics["floor_index"]),
                                        float(config["bev"]["world_meters_per_pixel"]),
                                        float(config["bev"]["bounds_margin_m"]),
                                    )
                                    candidate_snapshot = adapter.dump_snapshot()
                                    candidate_before = adapter.playback_trajectories(
                                        candidate_trajectories,
                                        int(candidate_metrics["floor_index"]),
                                        candidate_calibration,
                                    )
                                    candidate_visibility = _intervention_visibility_table(
                                        candidate_catalog,
                                        candidate_before["robot_views"],
                                        config,
                                    )
                                    if not candidate_visibility["eligible_target_ids"]:
                                        raise SampleRejected(
                                            "no_visible_intervention_target",
                                            candidate_visibility["requirements"],
                                        )
                                    trajectories = candidate_trajectories
                                    trajectory_metrics = candidate_metrics
                                    graph = candidate_graph
                                    temporal_overlap_metrics = candidate_overlap
                                    w0_catalog = candidate_catalog
                                    world_calibration = candidate_calibration
                                    w0_snapshot = candidate_snapshot
                                    before = candidate_before
                                    visibility_table = candidate_visibility
                                    break
                                except SampleRejected as candidate_error:
                                    before = None
                                    candidate_failures.append({
                                        "candidate_rank": candidate_rank,
                                        "candidate_kind": candidate_metrics.get(
                                            "nested_trajectory_sets", {}
                                        ).get("candidate_kind", "base"),
                                        "reason": candidate_error.reason,
                                        "details": candidate_error.details,
                                        "trajectory_summary": {
                                            "observation_regime": candidate_metrics.get(
                                                "observation_regime"
                                            ),
                                            "joint_diversity": candidate_metrics.get(
                                                "joint_diversity"
                                            ),
                                            "robots": candidate_metrics.get("robots"),
                                            "nested_trajectory_sets": candidate_metrics.get(
                                                "nested_trajectory_sets"
                                            ),
                                        },
                                    })
                                    candidate_rank += 1
                            if before is None:
                                raise SampleRejected(
                                    "trajectory_set_candidates_exhausted",
                                    {"candidate_failures": candidate_failures},
                                )
                            break
                        except SampleRejected as error:
                            before = None
                            writer.record_reject(
                                f"episode-before:{selected_scene}/{configuration_id}/{episode_id}",
                                error.reason,
                                {"attempt": placement_attempt, **error.details},
                            )
                    if before is None:
                        raise SampleRejected(
                            "episode_before_attempts_exhausted",
                            {"scene_id": selected_scene, "configuration_id": configuration_id},
                        )
                    after = None
                    intervention = None
                    environment_after = None
                    after_catalog = None
                    after_snapshot = None
                    qa_results = None
                    post_render_effect = None
                    visible_ids = set(visibility_table["eligible_target_ids"])
                    available_intervention_types = {
                        intervention_type
                        for intervention_type in InterventionType
                        if any(
                            obj.instance_id in visible_ids
                            for obj in eligible_intervention_targets(w0_catalog, intervention_type)
                        )
                    }
                    fixed_intervention_type = _choose_quota_intervention_type(
                        config["intervention"]["type_weights"],
                        accepted_intervention_types,
                        available_intervention_types,
                        stable_seed(episode_seed, "intervention_type"),
                    )
                    event_attempt_count = min(
                        int(config["intervention"]["maximum_attempts"]),
                        int(config["intervention"]["post_render_effect"]["maximum_resample_attempts"]),
                    )
                    for event_attempt in range(event_attempt_count):
                        _write_status(
                            root,
                            status="running",
                            stage="sample_episode_intervention",
                            scene_id=selected_scene,
                            configuration_id=configuration_id,
                            episode_id=episode_id,
                            attempt=event_attempt,
                            accepted_configurations=accepted_configurations,
                            accepted_episodes=accepted_episodes,
                        )
                        adapter.load_snapshot(w0_snapshot)
                        try:
                            intervention = adapter.apply_atomic_intervention(
                                stable_seed(episode_seed, "intervention", event_attempt),
                                forced_type=fixed_intervention_type,
                                excluded_target_ids=tuple(sorted(used_targets)),
                                visible_target_ids=tuple(visibility_table["eligible_target_ids"]),
                            )
                            environment_after, _ = _render_environment_floors(adapter, config)
                            after = adapter.playback_trajectories(
                                trajectories,
                                int(trajectory_metrics["floor_index"]),
                                world_calibration,
                            )
                            post_render_effect = _post_render_intervention_effect(
                                intervention["event"].target_instance_id,
                                w0_catalog,
                                before["robot_views"], after["robot_views"], config,
                            )
                            if not post_render_effect["passed"]:
                                raise SampleRejected(
                                    "post_render_intervention_effect_failed",
                                    post_render_effect,
                                )
                            paired = check_paired_trajectories(
                                before["actual_trajectories"],
                                after["actual_trajectories"],
                                position_atol_m=float(
                                    config["trajectory"]["validation_position_tolerance_m"]
                                ),
                                matrix_atol=float(
                                    config["trajectory"]["validation_rotation_tolerance_rad"]
                                ),
                            )
                            bev_pair = check_bev_pair(
                                world_calibration,
                                world_calibration,
                                True,
                                True,
                            )
                            qa_results = (
                                QAResult(
                                    "initial_overlap",
                                    bool(not graph.near_duplicate_pairs),
                                    metrics={
                                        "minimum_overlap": float(min(graph.overlaps.values())),
                                        "maximum_overlap": float(max(graph.overlaps.values())),
                                    },
                                ),
                                QAResult(
                                    "temporal_overlap_connectivity",
                                    bool(temporal_overlap_metrics["passed"]),
                                    metrics={
                                        "connected_fraction": float(
                                            temporal_overlap_metrics["connected_fraction"]
                                        ),
                                        "maximum_isolated_run": max(
                                            temporal_overlap_metrics["maximum_consecutive_isolated_keyframes"].values()
                                        ),
                                        "union_edges": temporal_overlap_metrics["union_edges"],
                                        "requested_regime": temporal_overlap_metrics[
                                            "requested_regime"
                                        ],
                                        "realized_regime": temporal_overlap_metrics[
                                            "realized_regime"
                                        ],
                                    },
                                ),
                                QAResult(
                                    "before_rollout",
                                    all(before["checks"].values()),
                                    metrics=before["metrics"],
                                ),
                                QAResult(
                                    "after_rollout",
                                    all(after["checks"].values()),
                                    metrics=after["metrics"],
                                ),
                                paired,
                                bev_pair,
                                QAResult(
                                    "atomic_intervention",
                                    all(intervention["checks"].values()),
                                    metrics={
                                        "changed_object_count": len(
                                            intervention["changed_instance_ids"]
                                        )
                                    },
                                ),
                                QAResult(
                                    "post_render_intervention_effect",
                                    bool(post_render_effect["passed"]),
                                    metrics={
                                        "changed_pixels": int(post_render_effect["changed_pixels"]),
                                        "mean_rgb_delta": float(post_render_effect["mean_rgb_delta"]),
                                    },
                                ),
                            )
                            require_all(qa_results)
                            after_catalog = intervention["catalog"]
                            after_snapshot = intervention["snapshot"]
                            break
                        except SampleRejected as error:
                            after = None
                            intervention = None
                            writer.record_reject(
                                f"event:{selected_scene}/{configuration_id}/{episode_id}",
                                error.reason,
                                {"attempt": event_attempt, **error.details},
                            )
                    if after is None or intervention is None:
                        raise SampleRejected(
                            "intervention_attempts_exhausted",
                            {
                                "scene_id": selected_scene,
                                "configuration_id": configuration_id,
                                "episode_id": episode_id,
                            },
                        )
                    robot_states = _robot_states(config, heights, trajectories)
                    state_before = WorldState(
                        scene_id=selected_scene,
                        configuration_id=configuration_id,
                        physical_time_index=None,
                        objects=w0_catalog,
                        robots=robot_states,
                        simulator_snapshot_ref="simulator_before.npy",
                    )
                    state_after = WorldState(
                        scene_id=selected_scene,
                        configuration_id=configuration_id,
                        physical_time_index=None,
                        objects=after_catalog,
                        robots=robot_states,
                        simulator_snapshot_ref="simulator_after.npy",
                    )
                    sibling_diversity = _sibling_episode_diversity(
                        root,
                        selected_scene,
                        configuration_id,
                        trajectory_metrics,
                        trajectories,
                        fixed_intervention_type.value,
                        intervention["event"].target_instance_id,
                    )
                    with writer.begin_episode(
                        selected_scene, configuration_id, episode_id
                    ) as transaction:
                        before_world_ref = transaction.write_dense_group(
                            "bev/world_before", before["world_bev"]
                        )
                        after_world_ref = transaction.write_dense_group(
                            "bev/world_after", after["world_bev"]
                        )
                        environment_after_ref = transaction.write_dense_group(
                            "bev/environment_after", environment_after
                        )
                        before_view_refs = {
                            robot_id: transaction.write_dense_group(
                                f"robot_views/before/{robot_id}", modalities
                            )
                            for robot_id, modalities in before["robot_views"].items()
                        }
                        after_view_refs = {
                            robot_id: transaction.write_dense_group(
                                f"robot_views/after/{robot_id}", modalities
                            )
                            for robot_id, modalities in after["robot_views"].items()
                        }
                        np.save(
                            transaction.staging / "simulator_before.npy",
                            np.asarray(w0_snapshot),
                            allow_pickle=False,
                        )
                        np.save(
                            transaction.staging / "simulator_after.npy",
                            np.asarray(after_snapshot),
                            allow_pickle=False,
                        )
                        episode = WorldEpisode(
                            episode_id=episode_id,
                            scene_id=selected_scene,
                            configuration_id=configuration_id,
                            seed=episode_seed,
                            simulator_versions=adapter.runtime_report().get("versions", {}),
                            generator_git_commit=commit,
                            trajectories=trajectories,
                            intervention=intervention["event"],
                            state_before=state_before,
                            state_after=state_after,
                            environment_after_bev_ref=environment_after_ref,
                            world_before_bev_ref=before_world_ref,
                            world_after_bev_ref=after_world_ref,
                            observations_before=_observation_records(
                                config,
                                before["actual_trajectories"],
                                "before",
                                before_view_refs,
                                before["camera_capture_metadata"],
                            ),
                            observations_after=_observation_records(
                                config,
                                after["actual_trajectories"],
                                "after",
                                after_view_refs,
                                after["camera_capture_metadata"],
                            ),
                            qa=qa_results,
                        )
                        writer.write_episode_metadata(transaction, episode)
                        transaction.write_json(
                            "generation_metrics.json",
                            {
                                "overlap": graph,
                                "trajectory": trajectory_metrics,
                                "before": before["metrics"],
                                "after": after["metrics"],
                                "intervention_attempt": intervention["attempt"],
                                "fixed_intervention_type": fixed_intervention_type.value,
                                "intervention_visibility": visibility_table,
                                "post_render_intervention_effect": post_render_effect,
                                "sibling_episode_diversity": sibling_diversity,
                            },
                        )
                        inspection_root = transaction.staging / "inspection"
                        image_names = []
                        floor_id = (
                            f"floor_{int(trajectory_metrics['floor_index']):02d}"
                        )
                        environment_image_name = (
                            "environment_base_rooms_and_target.png"
                        )
                        with np.load(
                            configuration_root
                            / "bev"
                            / "environment_base.npz",
                            allow_pickle=False,
                        ) as environment_base:
                            save_environment_room_inspection(
                                inspection_root / environment_image_name,
                                environment_base,
                                w0_catalog,
                                intervention["event"],
                                floor_id=floor_id,
                            )
                        image_names.append(environment_image_name)

                        trajectory_image_name = "trajectory_inspection.png"
                        save_trajectory_inspection(
                            inspection_root / trajectory_image_name,
                            adapter.trajectory_traversability_inspection(
                                int(trajectory_metrics["floor_index"])
                            ),
                            trajectories,
                            temporal_overlap_metrics,
                        )
                        image_names.append(trajectory_image_name)

                        overlap_image_name = "overlap_keyframes.png"
                        save_overlap_graph_inspection(
                            inspection_root / overlap_image_name,
                            temporal_overlap_metrics,
                        )
                        image_names.append(overlap_image_name)

                        save_rgb(
                            inspection_root / "world_before_t000.png",
                            before["world_bev"]["rgb"][0],
                        )
                        save_rgb(
                            inspection_root / "world_after_t000.png",
                            after["world_bev"]["rgb"][0],
                        )
                        image_names.extend(
                            ["world_before_t000.png", "world_after_t000.png"]
                        )

                        target_crop_name = "intervention_target_crops.png"
                        if save_intervention_target_crops(
                            inspection_root / target_crop_name,
                            before["robot_views"],
                            after["robot_views"],
                            int(
                                post_render_effect[
                                    "target_public_instance_id"
                                ]
                            ),
                        ):
                            image_names.append(target_crop_name)

                        for robot_id in sorted(before["robot_views"]):
                            name = f"{robot_id}_before_t000.png"
                            save_rgb(
                                inspection_root / name,
                                before["robot_views"][robot_id]["rgb"][0],
                            )
                            image_names.append(name)
                        write_html_summary(
                            inspection_root,
                            f"{selected_scene}/{configuration_id}/{episode_id}",
                            {
                                "qa": qa_results,
                                "event": intervention["event"],
                                "trajectory": trajectory_metrics,
                                "sibling_episode_diversity": sibling_diversity,
                            },
                            image_names,
                        )
                        transaction.finalize()
                    used_targets.add(intervention["event"].target_instance_id)
                    accepted_intervention_types[fixed_intervention_type.value] += 1
                    requested_regime_counts[
                        str(trajectory_metrics["requested_observation_regime"])
                    ] += 1
                    realized_regime_counts[
                        str(trajectory_metrics["observation_regime"])
                    ] += 1
                    requested_regime_counts_by_split[split_name][
                        str(trajectory_metrics["requested_observation_regime"])
                    ] += 1
                    realized_regime_counts_by_split.setdefault(
                        split_name, Counter()
                    )[str(trajectory_metrics["observation_regime"])] += 1
                    used_regions.update(trajectory_metrics.get("start_region_ids", []))
                    for region_ids in trajectory_metrics.get("traversed_region_ids", {}).values():
                        used_regions.update(region_ids)
                    accepted_episodes += 1
                    _write_status(
                        root,
                        status="running",
                        stage="episode_finalized",
                        scene_id=selected_scene,
                        configuration_id=configuration_id,
                        episode_id=episode_id,
                        accepted_configurations=accepted_configurations,
                        accepted_episodes=accepted_episodes,
                    )
        result = {
            "status": "pass",
            "profile": profile,
            "scenes": scenes,
            "accepted_configurations": accepted_configurations,
            "accepted_episodes": accepted_episodes,
            "requested_overlap_regime_counts": dict(requested_regime_counts),
            "realized_overlap_regime_counts": dict(realized_regime_counts),
            "requested_overlap_regime_counts_by_split": {
                name: dict(values)
                for name, values in requested_regime_counts_by_split.items()
            },
            "realized_overlap_regime_counts_by_split": {
                name: dict(values)
                for name, values in realized_regime_counts_by_split.items()
            },
            "scope": "development profiles only; full generation not started",
        }
        dump_json(root / "generation_result.json", result)
        (root / "generation_failure.json").unlink(missing_ok=True)
        _write_status(root, **result, stage="complete")
        return root, result
    except BaseException as error:
        failure = {
            "status": "error",
            "profile": profile,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "accepted_configurations": accepted_configurations,
            "accepted_episodes": accepted_episodes,
        }
        dump_json(root / "generation_failure.json", failure)
        _write_status(root, **failure, stage="failed")
        raise
    finally:
        adapter.close()
