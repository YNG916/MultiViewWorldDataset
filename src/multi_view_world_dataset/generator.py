from __future__ import annotations

import json
import traceback
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from multi_view_world_dataset.assets import (
    ROBOT_ASSET_ID,
    ROBOT_ASSET_VERSION,
    robot_appearance_metadata,
    robot_asset_fingerprint,
)
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
    save_robot_appearance_summary,
)
from multi_view_world_dataset.sampling.configurations import (
    exact_state_hash,
    near_duplicate_configuration,
)
from multi_view_world_dataset.sampling.diversity import stable_seed, temporal_overlap_acceptance
from multi_view_world_dataset.sampling.interventions import eligible_intervention_targets
from multi_view_world_dataset.sampling.splits import assign_scene_family_splits, infer_scene_family
from multi_view_world_dataset.scene_eligibility import (
    load_scene_eligibility,
    reconcile_scene_eligibility,
    resolve_scene_eligibility_path,
    validate_scene_family_split_disjointness,
)
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


def _configuration_navigation_seed(configuration_metadata: Mapping[str, Any]) -> int:
    """Rebuild a persisted configuration's route bank with its acceptance seed."""
    return stable_seed(
        int(configuration_metadata["seed"]), "configuration-navigation-context"
    )


def _has_all_requested_episodes(completed_episode_ids: Sequence[str], requested_episodes: int) -> bool:
    """A committed configuration needs no simulator replay once every episode exists."""
    return all(
        f"episode_{index:03d}" in completed_episode_ids
        for index in range(requested_episodes)
    )


def _check_configuration_geometry(
    expected_objects: Sequence[ObjectState] | Sequence[Mapping[str, Any]],
    actual_objects: Sequence[ObjectState],
    *,
    aabb_tolerance_m: float,
) -> None:
    """Reject a snapshot whose collision geometry differs from its catalog."""
    def field(obj: ObjectState | Mapping[str, Any], name: str) -> Any:
        return obj[name] if isinstance(obj, Mapping) else getattr(obj, name)

    expected = {str(field(obj, "instance_id")): obj for obj in expected_objects}
    actual = {obj.instance_id: obj for obj in actual_objects}
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatches: list[dict[str, Any]] = []
    for instance_id in sorted(set(expected) & set(actual)):
        saved = expected[instance_id]
        current = actual[instance_id]
        aabb_error = max(
            float(np.max(np.abs(
                np.asarray(field(saved, name), dtype=np.float64)
                - np.asarray(getattr(current, name), dtype=np.float64)
            )))
            for name in ("bbox_min_world", "bbox_max_world")
        )
        pose_error = float(np.max(np.abs(
            np.asarray(field(saved, "object_to_world"), dtype=np.float64)
            - np.asarray(current.object_to_world, dtype=np.float64)
        )))
        if aabb_error > aabb_tolerance_m or pose_error > 1.0e-4:
            mismatches.append({
                "instance_id": instance_id,
                "category": current.category,
                "aabb_error_m": aabb_error,
                "pose_matrix_error": pose_error,
            })
    if missing or unexpected or mismatches:
        raise SampleRejected("configuration_snapshot_geometry_mismatch", {
            "missing_object_ids": missing[:10],
            "unexpected_object_ids": unexpected[:10],
            "mismatch_count": len(mismatches),
            "largest_mismatches": sorted(
                mismatches, key=lambda item: -item["aabb_error_m"]
            )[:10],
            "aabb_tolerance_m": aabb_tolerance_m,
        })


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
        adapter.refresh_collision_geometry_cache()
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
    keyframe_count_override: int | None = None,
    isolation_decision_mode: str = "sampled_keyframes",
    allow_dense_confirmation: bool = True,
) -> dict[str, Any]:
    """Validate sparse GT overlap and target visibility before dense capture."""
    preflight = config["trajectory"]["overlap_preflight"]
    frame_count = trajectories[0].frames
    requested_keyframe_count = int(
        keyframe_count_override
        if keyframe_count_override is not None
        else preflight["keyframe_count"]
    )
    keyframe_indices = np.unique(np.rint(np.linspace(
        0, frame_count - 1, requested_keyframe_count
    )).astype(int))
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
        regime_minimum_participating_keyframes=preflight.get(
            "regime_minimum_participating_keyframes",
            {"dense_shared": 1, "partial_chain": 2, "exploratory": 1},
        ),
        regime_maximum_isolation_fraction=preflight.get(
            "regime_maximum_isolation_fraction"
        ),
        isolation_decision_mode=isolation_decision_mode,
        episode_frame_count=frame_count,
    )
    metrics.update({
        "keyframe_indices": keyframe_indices.tolist(),
        "geometry_resolution": [width, height],
        "robot_ids": list(robot_ids),
        "keyframes": keyframes,
        "acceptance_source": (
            f"sparse_{len(keyframe_indices)}"
            if isolation_decision_mode == "sampled_keyframes"
            else f"dense_{len(keyframe_indices)}_confirmation"
        ),
        "gt_validation_accounting": {
            "sparse_7_validations": int(isolation_decision_mode == "sampled_keyframes"),
            "dense_13_confirmations": int(
                isolation_decision_mode == "normalized_duration"
            ),
        },
    })
    if requested_regime is not None:
        metrics["requested_regime"] = str(requested_regime)
        metrics["regime_target_match"] = bool(
            metrics["realized_regime"] == requested_regime
        )
    if (
        allow_dense_confirmation
        and bool(
            metrics.get("isolation_only_failure", False)
            or metrics.get("isolation_confirmation_recommended", False)
        )
    ):
        dense_count = int(preflight.get("dense_confirmation_keyframe_count", 13))
        try:
            dense_metrics = _temporal_overlap_preflight(
                adapter,
                config,
                trajectories,
                catalog=None,
                requested_regime=requested_regime,
                keyframe_count_override=dense_count,
                isolation_decision_mode="normalized_duration",
                allow_dense_confirmation=False,
            )
        except SampleRejected as dense_error:
            metrics["dense_confirmation"] = dense_error.details
            metrics["failure_reasons"] = list(metrics["failure_reasons"])
            metrics["failure_reasons"].append("dense_confirmation_failed")
            metrics["gt_validation_accounting"]["dense_13_confirmations"] = 1
            metrics["passed"] = False
            metrics["checks"]["no_severe_isolation"] = False
            metrics["isolation_only_failure"] = bool(
                metrics.get("passed_universal_hard_checks", False)
            )
        else:
            dense_metrics["sparse_validation"] = metrics
            dense_metrics["acceptance_source"] = (
                f"dense_{len(dense_metrics['keyframe_indices'])}_confirmation"
            )
            dense_metrics["gt_validation_accounting"] = {
                "sparse_7_validations": 1,
                "dense_13_confirmations": 1,
            }
            metrics = dense_metrics
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

def _realized_regime_deficit_preference(
    realized_regime: str,
    weights: dict[str, float],
    global_counts: Counter[str],
    split_counts: Counter[str],
) -> float:
    """Measure the running global + split deficit for one realized regime."""
    target_total = float(sum(weights.values()))
    if target_total <= 0.0 or realized_regime not in weights:
        raise ValueError("realized regime weights must be positive and complete")
    target_fraction = float(weights[realized_regime]) / target_total

    def deficit(counts: Counter[str]) -> float:
        total = int(sum(counts.values()))
        observed = counts[realized_regime] / total if total else 0.0
        return target_fraction - observed

    return float(deficit(global_counts) + deficit(split_counts))


def _gt_valid_candidate_soft_score(
    metrics: dict[str, Any],
    original_rank: int,
    candidate_count: int,
    realized_regime: str,
    weights: dict[str, float],
    global_counts: Counter[str],
    split_counts: Counter[str],
    *,
    lambda_regime: float,
) -> dict[str, float]:
    """Combine route quality and realized-regime deficit without a hard gate."""
    if lambda_regime < 0.0 or not np.isfinite(lambda_regime):
        raise ValueError("lambda_regime must be finite and non-negative")
    rank_denominator = max(1, candidate_count - 1)
    rank_quality = 1.0 - min(max(original_rank, 0), rank_denominator) / rank_denominator
    proxy_score = float(metrics.get("cheap_scene_visibility", {}).get("score", 0.0))
    bounded_proxy_quality = float(np.tanh(proxy_score / 4.0))
    quality_score = float(rank_quality + 0.10 * bounded_proxy_quality)
    deficit_preference = _realized_regime_deficit_preference(
        realized_regime, weights, global_counts, split_counts
    )
    final_score = quality_score + lambda_regime * deficit_preference
    return {
        "quality_score": quality_score,
        "rank_quality": float(rank_quality),
        "bounded_proxy_quality": bounded_proxy_quality,
        "realized_regime_deficit_preference": float(deficit_preference),
        "lambda_regime": float(lambda_regime),
        "final_score": float(final_score),
    }

def _adaptive_exact_validation(
    candidates: Sequence[Any],
    cumulative_batch_limits: Sequence[int],
    validate: Callable[[Any, int, int], Any],
) -> tuple[list[Any], dict[str, Any]]:
    """Validate each candidate at most once and expand only after an empty batch.

    Batch limits are cumulative ranks, e.g. ``(12, 24, 48)``.  Validation
    stops after the first batch containing at least one hard-valid candidate;
    selection among that batch's valid candidates remains a separate soft
    ranking step.
    """
    limits = tuple(int(value) for value in cumulative_batch_limits)
    if (
        not limits
        or any(value <= 0 for value in limits)
        or any(right <= left for left, right in zip(limits, limits[1:]))
    ):
        raise ValueError("exact-validation batch limits must be strictly increasing")
    valid: list[Any] = []
    records: list[dict[str, Any]] = []
    attempted_ranks: set[int] = set()
    previous_limit = 0
    accepted_batch: int | None = None
    for batch_limit in limits:
        upper = min(batch_limit, len(candidates))
        batch_valid: list[Any] = []
        for candidate_rank in range(previous_limit, upper):
            if candidate_rank in attempted_ranks:
                raise AssertionError("exact candidate was scheduled more than once")
            attempted_ranks.add(candidate_rank)
            candidate = candidates[candidate_rank]
            record = {
                "candidate_rank": candidate_rank,
                "validation_batch_limit": batch_limit,
                "gt_validation_attempted": True,
            }
            try:
                value = validate(candidate, candidate_rank, batch_limit)
            except SampleRejected as error:
                record.update({
                    "gt_valid": False,
                    "reject_reason": error.reason,
                    "reject_details": error.details,
                })
            else:
                record.update({"gt_valid": True, "reject_reason": None})
                batch_valid.append(value)
            records.append(record)
        valid.extend(batch_valid)
        if batch_valid:
            accepted_batch = batch_limit
            break
        previous_limit = upper
        if upper >= len(candidates):
            break
    return valid, {
        "configured_cumulative_batches": list(limits),
        "accepted_batch_limit": accepted_batch,
        "total_exact_candidates_tested": len(attempted_ranks),
        "gt_valid_candidate_count": len(valid),
        "candidate_records": records,
        "candidate_pool_exhausted": not valid,
    }



def _gt_rescue_candidates(
    adapter: Any,
    candidate_sets: Sequence[Any],
    candidate_failures: list[dict[str, Any]],
    seed: int,
) -> tuple[tuple[Any, ...], list[dict[str, Any]]]:
    """Build bounded GT-guided candidates only after the base pool is exhausted."""
    base = tuple(candidate_sets)
    generated: list[Any] = []
    failures: list[dict[str, Any]] = []
    builders = (
        (
            "complementary_hybrid",
            lambda: adapter.complementary_trajectory_hybrids(
                base, candidate_failures
            ),
        ),
        (
            "measured_overlap_bridge",
            lambda: adapter.measured_overlap_bridge_trajectories(
                base, candidate_failures, seed
            ),
        ),
    )
    for candidate_kind, build in builders:
        try:
            generated.extend(build())
        except SampleRejected as error:
            failures.append({
                "candidate_kind": candidate_kind,
                "reason": error.reason,
                "details": error.details,
            })
    return tuple(generated), failures


def _gt_validation_record_fields(overlap: Mapping[str, Any] | None) -> dict[str, Any]:
    """Flatten adaptive temporal-validation provenance into one candidate row."""
    details = dict(overlap or {})
    accounting = details.get("gt_validation_accounting", {})
    checks = details.get("checks", {})
    return {
        "sparse_7_validations": int(accounting.get("sparse_7_validations", 1)),
        "dense_13_confirmations": int(accounting.get("dense_13_confirmations", 0)),
        "temporal_acceptance_source": details.get("acceptance_source"),
        "passed_universal_hard_checks": details.get(
            "passed_universal_hard_checks"
        ),
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "structured_failure_reasons": list(details.get("failure_reasons", ())),
    }


def _candidate_accounting(
    *,
    base_candidates_generated: int,
    candidate_records: Sequence[Mapping[str, Any]],
    rescue_candidate_counts: Mapping[str, int],
    duplicate_candidates_removed: int = 0,
    accepted_candidate_source: str | None = None,
) -> dict[str, Any]:
    """Return internally checkable generated / exact-GT candidate totals."""
    base_exact = sum(
        record.get("validation_stage") == "adaptive_base"
        for record in candidate_records
    )
    rescue_exact = len(candidate_records) - base_exact
    sparse = sum(int(record.get("sparse_7_validations", 0)) for record in candidate_records)
    dense = sum(int(record.get("dense_13_confirmations", 0)) for record in candidate_records)
    return {
        "base_candidates_generated": int(base_candidates_generated),
        "base_candidates_exact_gt_validated": int(base_exact),
        "rescue_candidates_generated": int(sum(rescue_candidate_counts.values())),
        "rescue_candidates_exact_gt_validated": int(rescue_exact),
        "rescue_subtype_counts": dict(rescue_candidate_counts),
        "duplicate_candidates_removed": int(duplicate_candidates_removed),
        "sparse_7_validations": int(sparse),
        "dense_13_confirmations": int(dense),
        "exact_gt_candidate_record_count": len(candidate_records),
        "accepted_candidate_source": accepted_candidate_source,
        "totals_consistent": bool(
            base_exact + rescue_exact == len(candidate_records)
            and sparse == len(candidate_records)
        ),
    }


def _validate_gt_rescue_candidate(
    adapter: Any,
    config: dict[str, Any],
    candidate_trajectories: tuple[Any, ...],
    candidate_metrics: dict[str, Any],
    *,
    original_rank: int,
    candidate_count: int,
    requested_overlap_regime: str,
    regime_weights: dict[str, float],
    realized_regime_counts: Counter[str],
    split_realized_counts: Counter[str],
    lambda_regime: float,
    split_name: str,
    requested_regime_counts: Counter[str],
    split_requested_counts: Counter[str],
) -> dict[str, Any]:
    """Run the canonical exact-GT checks for a feedback rescue candidate."""
    adapter.place_robots_at_trajectory_frame(candidate_trajectories, 0)
    candidate_graph, _ = _initial_overlap(adapter, config)
    candidate_catalog = adapter.object_catalog_with_relations()
    candidate_overlap = _temporal_overlap_preflight(
        adapter,
        config,
        candidate_trajectories,
        catalog=candidate_catalog,
        requested_regime=requested_overlap_regime,
    )
    requested_regime = str(candidate_overlap["requested_regime"])
    realized_regime = str(candidate_overlap["realized_regime"])
    candidate_metrics["requested_observation_regime"] = requested_regime
    candidate_metrics["observation_regime"] = realized_regime
    for trajectory in candidate_trajectories:
        trajectory.metadata["requested_observation_regime"] = requested_regime
        trajectory.metadata["observation_regime"] = realized_regime
    candidate_metrics["temporal_overlap"] = candidate_overlap
    selection_score = _gt_valid_candidate_soft_score(
        candidate_metrics,
        original_rank,
        candidate_count,
        realized_regime,
        regime_weights,
        realized_regime_counts,
        split_realized_counts,
        lambda_regime=lambda_regime,
    )
    candidate_metrics["realized_regime_soft_selection"] = selection_score
    candidate_metrics["overlap_regime_distribution"] = {
        "target_weights": regime_weights,
        "global_requested_before_accept": dict(requested_regime_counts),
        "global_realized_before_accept": dict(realized_regime_counts),
        "split": split_name,
        "split_requested_before_accept": dict(split_requested_counts),
        "split_realized_before_accept": dict(split_realized_counts),
        "selection_policy": (
            "quality_plus_soft_realized_global_and_split_deficit"
        ),
    }
    candidate_metrics["temporal_preflight_candidate_rank"] = original_rank
    return {
        "original_rank": original_rank,
        "trajectories": candidate_trajectories,
        "metrics": candidate_metrics,
        "graph": candidate_graph,
        "catalog": candidate_catalog,
        "overlap": candidate_overlap,
        "selection_score": selection_score,
        "realized_regime": realized_regime,
    }


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
            mast_joint_value_m=(
                float(heights[robot_id]) - float(min(config["camera"]["heights_m"]))
                if config["robot"]["use_final_robot"] else None
            ),
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


def _available_intervention_types(
    catalog: tuple[ObjectState, ...],
    visible_target_ids: set[str],
    excluded_target_ids: set[str],
) -> set[InterventionType]:
    """Preselect only types with at least one candidate the adapter can use."""
    eligible_ids = visible_target_ids - excluded_target_ids
    available = set()
    for intervention_type in InterventionType:
        for obj in eligible_intervention_targets(catalog, intervention_type):
            if obj.instance_id not in eligible_ids:
                continue
            if intervention_type is InterventionType.RIGID_RELOCATION and not any(
                relation.get("predicate") in {"OnFloor", "OnTop", "Inside"}
                for relation in obj.relations
            ):
                continue
            available.add(intervention_type)
            break
    return available


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
    intervention_type: InterventionType | str,
) -> dict[str, Any]:
    ordered = sorted(before_catalog, key=lambda item: item.instance_id)
    public_id = 4 + next(index for index, obj in enumerate(ordered) if obj.instance_id == target_instance_id)
    effect = config["intervention"]["post_render_effect"]
    intervention_type_name = (
        intervention_type.value
        if isinstance(intervention_type, InterventionType)
        else str(intervention_type)
    )
    configured_threshold = effect["minimum_mean_rgb_delta"]
    delta_threshold = float(
        configured_threshold[intervention_type_name]
        if isinstance(configured_threshold, Mapping)
        else configured_threshold
    )
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
        "intervention_type": intervention_type_name,
        "minimum_mean_rgb_delta": delta_threshold,
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
    sampling_retry_epoch: int = 0,
) -> tuple[Path, dict[str, Any]]:
    profile = str(config["profile"])
    if profile not in {"smoke", "integration"} and not allow_large:
        raise ConfigurationError(
            "Refusing large pilot/default generation without explicit --allow-large"
        )
    if profile not in {"smoke", "integration"} and scene_id is None:
        raise ConfigurationError(
            "Multi-scene generation requires production-launch so every scene uses a fresh process"
        )
    if sampling_retry_epoch < 0:
        raise ConfigurationError("sampling_retry_epoch must be non-negative")
    root = runtime.require_output()
    repository_root = Path(__file__).resolve().parents[2]
    commit = generator_git_commit(repository_root)
    source_fingerprint = generator_source_fingerprint(repository_root)
    final_robot_fingerprint = robot_asset_fingerprint(repository_root)
    writer = DatasetWriter(root)
    writer.initialize(
        {
            "schema_version": config["dataset"]["schema_version"],
            "dataset_semantics": "Dataset-v1.1",
            "profile": profile,
            "seed": int(config["seed"]),
            "generator_git_commit": commit,
            "generator_source_fingerprint": source_fingerprint,
            "robot_asset": {
                "asset_id": ROBOT_ASSET_ID,
                "version": ROBOT_ASSET_VERSION,
                "fingerprint": final_robot_fingerprint,
            },
            "source_of_truth": "world_state+simulator_snapshot+trajectory+event_log",
            "coordinate_conventions": {
                "world": "right-handed Z-up",
                "camera": "OpenCV x-right y-down z-forward",
                "poses": "local-to-world homogeneous matrices",
                "depth_linear": "metric camera-forward depth in meters",
            },
            "image_channel_semantics": {
                "rgb": "uint8 RGB, channel-last, exactly 3 channels; renderer alpha removed",
                "normal": "float32 camera-space XYZ, channel-last, exactly 3 channels",
            },
            "robot_appearance_variants": robot_appearance_metadata(),
            "mast_joint_relation": "mast_joint_value_m = camera_height_m - 0.8",
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
        discovered_scenes = adapter.discover_scenes()
        eligibility_manifest_path = resolve_scene_eligibility_path(
            repository_root, config
        )
        eligibility_manifest = load_scene_eligibility(eligibility_manifest_path)
        eligibility = reconcile_scene_eligibility(
            discovered_scenes, eligibility_manifest
        )
        eligible_scenes = list(eligibility["eligible_scenes"])
        splits = assign_scene_family_splits(
            eligible_scenes,
            config["dataset"]["splits"],
            int(config["dataset"]["scene_family_split_seed"]),
        )
        validate_scene_family_split_disjointness(splits)
        writer.update_dataset_metadata({
            "scene_eligibility": eligibility,
            "scene_eligibility_manifest": str(eligibility_manifest_path),
            "scene_split_mapping": splits,
            "scene_family_mapping": {
                scene: infer_scene_family(scene) for scene in eligible_scenes
            },
        })
        scenes = eligible_scenes
        if scene_id is not None:
            if scene_id not in discovered_scenes:
                raise ConfigurationError(f"Requested scene is not installed: {scene_id}")
            if scene_id not in eligible_scenes:
                reason = eligibility_manifest.by_scene[scene_id].reason
                raise ConfigurationError(
                    f"Requested scene is ineligible: {scene_id}: {reason}"
                )
            scenes = [scene_id]
        scene_limit = config["dataset"].get("scene_limit")
        if scene_limit is not None:
            scenes = scenes[: int(scene_limit)]
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
            adapter.refresh_collision_geometry_cache()
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
            accepted_episodes += sum(
                len(writer.completed_episode_ids(selected_scene, configuration_id))
                for configuration_id in expected_configuration_ids
            )
            for configuration_index in range(requested_configurations):
                configuration_id = f"config_{configuration_index:03d}"
                if configuration_id in completed_configurations and _has_all_requested_episodes(
                    writer.completed_episode_ids(selected_scene, configuration_id), requested_episodes
                ):
                    for episode_index in range(requested_episodes):
                        metrics_path = (
                            root / "episodes" / selected_scene / configuration_id
                            / f"episode_{episode_index:03d}" / "generation_metrics.json"
                        )
                        try:
                            prior_trajectory = json.loads(
                                metrics_path.read_text(encoding="utf-8")
                            )["trajectory"]
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
                    _write_status(
                        root, status="running", stage="resume_skip_completed_configuration",
                        scene_id=selected_scene, configuration_id=configuration_id,
                        accepted_configurations=accepted_configurations, accepted_episodes=accepted_episodes,
                    )
                    continue
                configuration_root = (
                    root / "configurations" / selected_scene / configuration_id
                )
                if configuration_id not in completed_configurations:
                    accepted = None
                    for attempt in range(int(config["generation"]["maximum_configuration_attempts"])):
                        configuration_seed_parts: tuple[object, ...] = (
                            selected_scene,
                            configuration_id,
                            "configuration",
                            attempt,
                        )
                        if sampling_retry_epoch:
                            configuration_seed_parts += (
                                "sampling-retry-epoch",
                                sampling_retry_epoch,
                            )
                        seed = stable_seed(
                            int(config["seed"]), *configuration_seed_parts
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
                            initial_catalog = candidate["catalog"]
                            adapter.load_snapshot(candidate["snapshot"])
                            adapter.refresh_collision_geometry_cache()
                            _check_configuration_geometry(
                                initial_catalog,
                                adapter.object_catalog(),
                                aabb_tolerance_m=float(config["generation"]["configuration_geometry_aabb_tolerance_m"]),
                            )
                            navigation_context_started = time.perf_counter()
                            environment_arrays, _ = _render_environment_floors(adapter, config)
                            # BEV rendering initializes and flushes Fabric geometry.
                            # Canonicalize after that flush so fresh processes see
                            # the same collision hulls as the stored catalog.
                            adapter.load_snapshot(candidate["snapshot"])
                            adapter.refresh_collision_geometry_cache()
                            canonical_catalog = adapter.object_catalog_with_relations()
                            maximum_restore_error, discrete_equal = (
                                adapter._catalog_restore_metrics(initial_catalog, canonical_catalog)
                            )
                            if (
                                maximum_restore_error > float(config["generation"]["snapshot_restore_tolerance"])
                                or not discrete_equal
                            ):
                                raise SampleRejected("configuration_after_bev_state_mismatch", {
                                    "maximum_restore_error": maximum_restore_error,
                                    "discrete_equal": discrete_equal,
                                })
                            initial_by_id = {obj.instance_id: obj for obj in initial_catalog}
                            canonical_aabb_shift = max(
                                max(
                                    float(np.max(np.abs(
                                        np.asarray(getattr(obj, field))
                                        - np.asarray(getattr(initial_by_id[obj.instance_id], field))
                                    )))
                                    for field in ("bbox_min_world", "bbox_max_world")
                                )
                                for obj in canonical_catalog
                            )
                            candidate["catalog"] = canonical_catalog
                            candidate["exact_state_hash"] = exact_state_hash(
                                canonical_catalog,
                                decimals=int(config["generation"]["exact_hash_decimals"]),
                            )
                            candidate["canonical_aabb_shift_after_bev_m"] = canonical_aabb_shift
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
                                _configuration_navigation_seed({"seed": seed}),
                                force=True,
                            )
                            navigation_metadata = adapter.navigation_context_metadata()
                            adapter.load_snapshot(candidate["snapshot"])
                            adapter.refresh_collision_geometry_cache()
                            _check_configuration_geometry(
                                candidate["catalog"], adapter.object_catalog(),
                                aabb_tolerance_m=float(config["generation"]["configuration_geometry_aabb_tolerance_m"]),
                            )
                            navigation_context_build_s = (
                                time.perf_counter() - navigation_context_started
                            )
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
                                    "canonical_aabb_shift_after_bev_m": candidate.get(
                                        "canonical_aabb_shift_after_bev_m", 0.0
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
                                    "navigation_context_and_route_bank_build_s": navigation_context_build_s,
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
                if configuration_id in completed_configurations:
                    _write_status(
                        root, status="running", stage="resume_configuration",
                        scene_id=selected_scene, configuration_id=configuration_id,
                        accepted_configurations=accepted_configurations, accepted_episodes=accepted_episodes,
                    )
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
                adapter.refresh_collision_geometry_cache()
                try:
                    _check_configuration_geometry(
                        configuration_metadata["world_state"]["objects"],
                        adapter.object_catalog(),
                        aabb_tolerance_m=float(config["generation"]["configuration_geometry_aabb_tolerance_m"]),
                    )
                except SampleRejected as error:
                    raise SampleRejected(error.reason, {
                        "scene_id": selected_scene,
                        "configuration_id": configuration_id,
                        **error.details,
                    }) from error
                _write_status(
                    root, status="running", stage="build_navigation_context",
                    scene_id=selected_scene, configuration_id=configuration_id,
                    accepted_configurations=accepted_configurations, accepted_episodes=accepted_episodes,
                )
                adapter.prepare_navigation_context(
                    configuration_token,
                    _configuration_navigation_seed(configuration_metadata),
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
                    episode_seed_parts: tuple[object, ...] = (
                        selected_scene,
                        configuration_id,
                        episode_id,
                    )
                    if sampling_retry_epoch:
                        episode_seed_parts += (
                            "sampling-retry-epoch",
                            sampling_retry_epoch,
                        )
                    episode_seed = stable_seed(
                        int(config["seed"]), *episode_seed_parts
                    )
                    episode_started = time.perf_counter()
                    episode_timing: dict[str, float] = {}
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
                    placement_attempts_per_round = int(
                        config["placement"]["maximum_attempts"]
                    )
                    episode_sampling_rounds = int(
                        config["generation"]["maximum_episode_sampling_rounds"]
                    )
                    for placement_attempt in range(
                        placement_attempts_per_round * episode_sampling_rounds
                    ):
                        sampling_round = placement_attempt // placement_attempts_per_round
                        attempt_in_round = placement_attempt % placement_attempts_per_round
                        _write_status(
                            root,
                            status="running",
                            stage="sample_episode_before",
                            scene_id=selected_scene,
                            configuration_id=configuration_id,
                            episode_id=episode_id,
                            attempt=placement_attempt,
                            sampling_round=sampling_round,
                            attempt_in_round=attempt_in_round,
                            accepted_configurations=accepted_configurations,
                            accepted_episodes=accepted_episodes,
                            sampling_retry_epoch=sampling_retry_epoch,
                        )
                        adapter.load_snapshot(configuration_snapshot)
                        try:
                            route_search_started = time.perf_counter()
                            heights, base_trajectory_candidates = (
                                adapter.sample_route_first_trajectory_sets(
                                    stable_seed(episode_seed, "route-first", placement_attempt),
                                    discouraged_region_ids=tuple(sorted(used_regions)),
                                )
                            )
                            trajectory_candidates = list(base_trajectory_candidates)
                            episode_timing["route_bank_joint_search_s"] = (
                                episode_timing.get("route_bank_joint_search_s", 0.0)
                                + time.perf_counter() - route_search_started
                            )
                            candidate_failures: list[dict[str, Any]] = []
                            gt_valid_candidates: list[dict[str, Any]] = []
                            regime_weights = config["placement"]["observation_regime_weights"]
                            split_realized_counts = realized_regime_counts_by_split.setdefault(
                                split_name, Counter()
                            )
                            lambda_regime = float(
                                config["trajectory"]["overlap_preflight"][
                                    "realized_regime_soft_weight"
                                ]
                            )
                            exact_validation_started = time.perf_counter()
                            exact_batch_limits = tuple(map(
                                int, config["navigation"]["exact_validation_batches"]
                            ))
                            active_exact_batch_limit = exact_batch_limits[0]
                            exact_candidate_records: list[dict[str, Any]] = []
                            total_exact_candidates_tested = 0
                            for original_rank, (
                                candidate_trajectories,
                                candidate_metrics,
                            ) in enumerate(trajectory_candidates):
                                if original_rank >= active_exact_batch_limit:
                                    if gt_valid_candidates:
                                        break
                                    active_exact_batch_limit = next(
                                        limit for limit in exact_batch_limits
                                        if limit > original_rank
                                    )
                                total_exact_candidates_tested += 1
                                candidate_proxy_score = float(
                                    candidate_metrics.get(
                                        "cheap_scene_visibility", {}
                                    ).get("score", 0.0)
                                )
                                candidate_kind = str(
                                    candidate_metrics.get(
                                        "nested_trajectory_sets", {}
                                    ).get("candidate_kind", "base")
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
                                    selection_score = _gt_valid_candidate_soft_score(
                                        candidate_metrics,
                                        original_rank,
                                        len(trajectory_candidates),
                                        realized_regime,
                                        regime_weights,
                                        realized_regime_counts,
                                        split_realized_counts,
                                        lambda_regime=lambda_regime,
                                    )
                                    candidate_metrics["realized_regime_soft_selection"] = (
                                        selection_score
                                    )
                                    candidate_metrics["overlap_regime_distribution"] = {
                                        "target_weights": regime_weights,
                                        "global_requested_before_accept": dict(
                                            requested_regime_counts
                                        ),
                                        "global_realized_before_accept": dict(
                                            realized_regime_counts
                                        ),
                                        "split": split_name,
                                        "split_requested_before_accept": dict(
                                            requested_regime_counts_by_split[split_name]
                                        ),
                                        "split_realized_before_accept": dict(
                                            split_realized_counts
                                        ),
                                        "selection_policy": (
                                            "quality_plus_soft_realized_global_and_split_deficit"
                                        ),
                                    }
                                    candidate_metrics[
                                        "temporal_preflight_candidate_rank"
                                    ] = original_rank
                                    gt_valid_candidates.append({
                                        "original_rank": original_rank,
                                        "candidate_kind": candidate_kind,
                                        "validation_stage": "adaptive_base",
                                        "trajectories": candidate_trajectories,
                                        "metrics": candidate_metrics,
                                        "graph": candidate_graph,
                                        "catalog": candidate_catalog,
                                        "overlap": candidate_overlap,
                                        "selection_score": selection_score,
                                    })
                                    exact_candidate_records.append({
                                        "candidate_rank": original_rank,
                                        "candidate_kind": candidate_kind,
                                        "validation_stage": "adaptive_base",
                                        "cheap_proxy_score": candidate_proxy_score,
                                        "gt_validation_attempted": True,
                                        "gt_valid": True,
                                        "reject_reason": None,
                                        "validation_batch_limit": active_exact_batch_limit,
                                        "realized_regime": realized_regime,
                                    })
                                except SampleRejected as candidate_error:
                                    failure_requested = str(
                                        candidate_error.details.get(
                                            "requested_regime", requested_overlap_regime
                                        )
                                    )
                                    failure_realized = str(
                                        candidate_error.details.get(
                                            "realized_regime", "unclassified"
                                        )
                                    )
                                    candidate_metrics[
                                        "requested_observation_regime"
                                    ] = failure_requested
                                    candidate_metrics[
                                        "observation_regime"
                                    ] = failure_realized
                                    if (
                                        candidate_error.reason
                                        == "trajectory_temporal_overlap_failed"
                                    ):
                                        candidate_metrics["temporal_overlap"] = (
                                            candidate_error.details
                                        )
                                    for candidate_trajectory in candidate_trajectories:
                                        candidate_trajectory.metadata[
                                            "requested_observation_regime"
                                        ] = failure_requested
                                        candidate_trajectory.metadata[
                                            "observation_regime"
                                        ] = failure_realized
                                    candidate_failures.append({
                                        "candidate_rank": original_rank,
                                        "candidate_kind": candidate_kind,
                                        "cheap_proxy_score": candidate_proxy_score,
                                        "selection_stage": "gt_depth_preflight",
                                        "validation_stage": "adaptive_base",
                                        "reason": candidate_error.reason,
                                        "details": candidate_error.details,
                                    })
                                    exact_candidate_records.append({
                                        "candidate_rank": original_rank,
                                        "candidate_kind": candidate_kind,
                                        "validation_stage": "adaptive_base",
                                        "cheap_proxy_score": candidate_proxy_score,
                                        "gt_validation_attempted": True,
                                        "gt_valid": False,
                                        "reject_reason": candidate_error.reason,
                                        "validation_batch_limit": active_exact_batch_limit,
                                    })
                            rescue_candidate_counts: Counter[str] = Counter()
                            rescue_generation_failures: list[dict[str, Any]] = []
                            rescue_candidates: tuple[Any, ...] = ()
                            minimum_gt_valid_candidates = int(
                                config["trajectory"][
                                    "minimum_gt_valid_candidates_before_rescue"
                                ]
                            )
                            if (
                                len(gt_valid_candidates)
                                < minimum_gt_valid_candidates
                            ):
                                adapter.load_snapshot(configuration_snapshot)
                                (
                                    rescue_candidates,
                                    rescue_generation_failures,
                                ) = _gt_rescue_candidates(
                                    adapter,
                                    trajectory_candidates,
                                    candidate_failures,
                                    stable_seed(
                                        episode_seed,
                                        "measured-overlap-bridge",
                                        placement_attempt,
                                    ),
                                )
                                for rescue_offset, (
                                    candidate_trajectories,
                                    candidate_metrics,
                                ) in enumerate(rescue_candidates):
                                    original_rank = (
                                        len(trajectory_candidates)
                                        + rescue_offset
                                    )
                                    candidate_kind = str(
                                        candidate_metrics.get(
                                            "nested_trajectory_sets", {}
                                        ).get("candidate_kind", "gt_rescue")
                                    )
                                    rescue_candidate_counts[candidate_kind] += 1
                                    total_exact_candidates_tested += 1
                                    candidate_proxy_score = float(
                                        candidate_metrics.get(
                                            "cheap_scene_visibility", {}
                                        ).get("score", 0.0)
                                    )
                                    try:
                                        adapter.place_robots_at_trajectory_frame(
                                            candidate_trajectories, 0
                                        )
                                        candidate_graph, _ = _initial_overlap(
                                            adapter, config
                                        )
                                        candidate_catalog = (
                                            adapter.object_catalog_with_relations()
                                        )
                                        candidate_overlap = (
                                            _temporal_overlap_preflight(
                                                adapter,
                                                config,
                                                candidate_trajectories,
                                                catalog=candidate_catalog,
                                                requested_regime=(
                                                    requested_overlap_regime
                                                ),
                                            )
                                        )
                                        requested_regime = str(
                                            candidate_overlap[
                                                "requested_regime"
                                            ]
                                        )
                                        realized_regime = str(
                                            candidate_overlap["realized_regime"]
                                        )
                                        candidate_metrics[
                                            "requested_observation_regime"
                                        ] = requested_regime
                                        candidate_metrics[
                                            "observation_regime"
                                        ] = realized_regime
                                        for candidate_trajectory in (
                                            candidate_trajectories
                                        ):
                                            candidate_trajectory.metadata[
                                                "requested_observation_regime"
                                            ] = requested_regime
                                            candidate_trajectory.metadata[
                                                "observation_regime"
                                            ] = realized_regime
                                        candidate_metrics[
                                            "temporal_overlap"
                                        ] = candidate_overlap
                                        selection_score = (
                                            _gt_valid_candidate_soft_score(
                                                candidate_metrics,
                                                original_rank,
                                                (
                                                    len(trajectory_candidates)
                                                    + len(rescue_candidates)
                                                ),
                                                realized_regime,
                                                regime_weights,
                                                realized_regime_counts,
                                                split_realized_counts,
                                                lambda_regime=lambda_regime,
                                            )
                                        )
                                        candidate_metrics[
                                            "realized_regime_soft_selection"
                                        ] = selection_score
                                        candidate_metrics[
                                            "overlap_regime_distribution"
                                        ] = {
                                            "target_weights": regime_weights,
                                            "global_requested_before_accept": dict(
                                                requested_regime_counts
                                            ),
                                            "global_realized_before_accept": dict(
                                                realized_regime_counts
                                            ),
                                            "split": split_name,
                                            "split_requested_before_accept": dict(
                                                requested_regime_counts_by_split[
                                                    split_name
                                                ]
                                            ),
                                            "split_realized_before_accept": dict(
                                                split_realized_counts
                                            ),
                                            "selection_policy": (
                                                "quality_plus_soft_realized_"
                                                "global_and_split_deficit"
                                            ),
                                        }
                                        candidate_metrics[
                                            "temporal_preflight_candidate_rank"
                                        ] = original_rank
                                        gt_valid_candidates.append({
                                            "original_rank": original_rank,
                                            "candidate_kind": candidate_kind,
                                            "validation_stage": "gt_rescue",
                                            "trajectories": (
                                                candidate_trajectories
                                            ),
                                            "metrics": candidate_metrics,
                                            "graph": candidate_graph,
                                            "catalog": candidate_catalog,
                                            "overlap": candidate_overlap,
                                            "selection_score": selection_score,
                                        })
                                        exact_candidate_records.append({
                                            "candidate_rank": original_rank,
                                            "candidate_kind": candidate_kind,
                                            "validation_stage": "gt_rescue",
                                            "cheap_proxy_score": (
                                                candidate_proxy_score
                                            ),
                                            "gt_validation_attempted": True,
                                            "gt_valid": True,
                                            "reject_reason": None,
                                            "validation_batch_limit": (
                                                active_exact_batch_limit
                                            ),
                                            "realized_regime": realized_regime,
                                        })
                                    except SampleRejected as candidate_error:
                                        candidate_failures.append({
                                            "candidate_rank": original_rank,
                                            "candidate_kind": candidate_kind,
                                            "cheap_proxy_score": (
                                                candidate_proxy_score
                                            ),
                                            "selection_stage": (
                                                "gt_depth_preflight"
                                            ),
                                            "validation_stage": "gt_rescue",
                                            "reason": candidate_error.reason,
                                            "details": candidate_error.details,
                                        })
                                        exact_candidate_records.append({
                                            "candidate_rank": original_rank,
                                            "candidate_kind": candidate_kind,
                                            "validation_stage": "gt_rescue",
                                            "cheap_proxy_score": (
                                                candidate_proxy_score
                                            ),
                                            "gt_validation_attempted": True,
                                            "gt_valid": False,
                                            "reject_reason": (
                                                candidate_error.reason
                                            ),
                                            "validation_batch_limit": (
                                                active_exact_batch_limit
                                            ),
                                        })
                                all_rescue_candidates = list(rescue_candidates)
                                maximum_feedback_rounds = int(
                                    config["trajectory"][
                                        "maximum_gt_feedback_mutation_rounds"
                                    ]
                                )
                                for feedback_round in range(
                                    1, maximum_feedback_rounds
                                ):
                                    if (
                                        len(gt_valid_candidates)
                                        >= minimum_gt_valid_candidates
                                    ):
                                        break
                                    adapter.load_snapshot(configuration_snapshot)
                                    combined_candidates = tuple(
                                        trajectory_candidates
                                    ) + tuple(all_rescue_candidates)
                                    try:
                                        feedback_candidates = (
                                            adapter.measured_overlap_bridge_trajectories(
                                                combined_candidates,
                                                candidate_failures,
                                                stable_seed(
                                                    episode_seed,
                                                    "measured-overlap-feedback",
                                                    placement_attempt,
                                                    feedback_round,
                                                ),
                                            )
                                        )
                                    except SampleRejected as feedback_error:
                                        rescue_generation_failures.append({
                                            "candidate_kind": (
                                                "measured_overlap_route_mutation"
                                            ),
                                            "feedback_round": feedback_round + 1,
                                            "reason": feedback_error.reason,
                                            "details": feedback_error.details,
                                        })
                                        break
                                    if not feedback_candidates:
                                        break
                                    feedback_start_rank = len(combined_candidates)
                                    feedback_candidate_count = (
                                        feedback_start_rank
                                        + len(feedback_candidates)
                                    )
                                    for feedback_offset, (
                                        candidate_trajectories,
                                        candidate_metrics,
                                    ) in enumerate(feedback_candidates):
                                        original_rank = (
                                            feedback_start_rank + feedback_offset
                                        )
                                        candidate_kind = str(
                                            candidate_metrics.get(
                                                "nested_trajectory_sets", {}
                                            ).get(
                                                "candidate_kind",
                                                "measured_overlap_route_mutation",
                                            )
                                        )
                                        rescue_candidate_counts[candidate_kind] += 1
                                        total_exact_candidates_tested += 1
                                        candidate_proxy_score = float(
                                            candidate_metrics.get(
                                                "cheap_scene_visibility", {}
                                            ).get("score", 0.0)
                                        )
                                        validation_stage = (
                                            f"gt_feedback_mutation_round_"
                                            f"{feedback_round + 1}"
                                        )
                                        try:
                                            validated = (
                                                _validate_gt_rescue_candidate(
                                                    adapter,
                                                    config,
                                                    candidate_trajectories,
                                                    candidate_metrics,
                                                    original_rank=original_rank,
                                                    candidate_count=(
                                                        feedback_candidate_count
                                                    ),
                                                    requested_overlap_regime=(
                                                        requested_overlap_regime
                                                    ),
                                                    regime_weights=regime_weights,
                                                    realized_regime_counts=(
                                                        realized_regime_counts
                                                    ),
                                                    split_realized_counts=(
                                                        split_realized_counts
                                                    ),
                                                    lambda_regime=lambda_regime,
                                                    split_name=split_name,
                                                    requested_regime_counts=(
                                                        requested_regime_counts
                                                    ),
                                                    split_requested_counts=(
                                                        requested_regime_counts_by_split[
                                                            split_name
                                                        ]
                                                    ),
                                                )
                                            )
                                        except SampleRejected as candidate_error:
                                            candidate_failures.append({
                                                "candidate_rank": original_rank,
                                                "candidate_kind": candidate_kind,
                                                "cheap_proxy_score": (
                                                    candidate_proxy_score
                                                ),
                                                "selection_stage": (
                                                    "gt_depth_preflight"
                                                ),
                                                "validation_stage": (
                                                    validation_stage
                                                ),
                                                "feedback_round": (
                                                    feedback_round + 1
                                                ),
                                                "reason": candidate_error.reason,
                                                "details": candidate_error.details,
                                            })
                                            exact_candidate_records.append({
                                                "candidate_rank": original_rank,
                                                "candidate_kind": candidate_kind,
                                                "validation_stage": (
                                                    validation_stage
                                                ),
                                                "feedback_round": (
                                                    feedback_round + 1
                                                ),
                                                "cheap_proxy_score": (
                                                    candidate_proxy_score
                                                ),
                                                "gt_validation_attempted": True,
                                                "gt_valid": False,
                                                "reject_reason": (
                                                    candidate_error.reason
                                                ),
                                                "validation_batch_limit": (
                                                    active_exact_batch_limit
                                                ),
                                            })
                                        else:
                                            validated.update({
                                                "candidate_kind": candidate_kind,
                                                "validation_stage": (
                                                    validation_stage
                                                ),
                                            })
                                            gt_valid_candidates.append(validated)
                                            exact_candidate_records.append({
                                                "candidate_rank": original_rank,
                                                "candidate_kind": candidate_kind,
                                                "validation_stage": (
                                                    validation_stage
                                                ),
                                                "feedback_round": (
                                                    feedback_round + 1
                                                ),
                                                "cheap_proxy_score": (
                                                    candidate_proxy_score
                                                ),
                                                "gt_validation_attempted": True,
                                                "gt_valid": True,
                                                "reject_reason": None,
                                                "validation_batch_limit": (
                                                    active_exact_batch_limit
                                                ),
                                                "realized_regime": validated[
                                                    "realized_regime"
                                                ],
                                            })
                                    all_rescue_candidates.extend(
                                        feedback_candidates
                                    )
                                rescue_candidates = tuple(all_rescue_candidates)
                            failure_lookup = {
                                (
                                    int(item["candidate_rank"]),
                                    str(item.get("validation_stage", "")),
                                ): item.get("details", {})
                                for item in candidate_failures
                            }
                            valid_lookup = {
                                (
                                    int(item["original_rank"]),
                                    str(item.get("validation_stage", "")),
                                ): item.get("overlap", {})
                                for item in gt_valid_candidates
                            }
                            for record in exact_candidate_records:
                                lookup_key = (
                                    int(record["candidate_rank"]),
                                    str(record.get("validation_stage", "")),
                                )
                                overlap_details = (
                                    valid_lookup.get(lookup_key)
                                    if record.get("gt_valid", False)
                                    else failure_lookup.get(lookup_key)
                                )
                                record.update(
                                    _gt_validation_record_fields(overlap_details)
                                )
                            if total_exact_candidates_tested != len(
                                exact_candidate_records
                            ):
                                raise AssertionError(
                                    "exact GT candidate counter does not match records"
                                )
                            duplicate_candidates_removed = max(
                                (
                                    int(metrics.get(
                                        "rescue_candidate_accounting", {}
                                    ).get("duplicate_route_triplets_removed", 0))
                                    for _, metrics in rescue_candidates
                                ),
                                default=0,
                            )
                            candidate_accounting = _candidate_accounting(
                                base_candidates_generated=len(trajectory_candidates),
                                candidate_records=exact_candidate_records,
                                rescue_candidate_counts=rescue_candidate_counts,
                                duplicate_candidates_removed=(
                                    duplicate_candidates_removed
                                ),
                            )
                            episode_timing["exact_gt_validation_s"] = (
                                episode_timing.get("exact_gt_validation_s", 0.0)
                                + time.perf_counter() - exact_validation_started
                            )
                            if not gt_valid_candidates:
                                raise SampleRejected(
                                    "trajectory_set_gt_candidates_exhausted",
                                    {
                                        "candidate_failures": candidate_failures,
                                        "exact_gt_validation": {
                                            "configured_cumulative_batches": list(exact_batch_limits),
                                            "accepted_batch_limit": None,
                                            "total_exact_candidates_tested": total_exact_candidates_tested,
                                            "gt_valid_candidate_count": 0,
                                            "candidate_records": (
                                                exact_candidate_records
                                            ),
                                            "rescue_candidate_counts": dict(
                                                rescue_candidate_counts
                                            ),
                                            "rescue_generation_failures": (
                                                rescue_generation_failures
                                            ),
                                            "candidate_pool_exhausted": True,
                                            "candidate_accounting": (
                                                candidate_accounting
                                            ),
                                        },
                                    },
                                )
                            gt_valid_candidates.sort(
                                key=lambda item: (
                                    -float(item["selection_score"]["final_score"]),
                                    int(item["original_rank"]),
                                )
                            )
                            for selection_rank, candidate_record in enumerate(
                                gt_valid_candidates
                            ):
                                candidate_rank = int(candidate_record["original_rank"])
                                candidate_trajectories = candidate_record["trajectories"]
                                candidate_metrics = candidate_record["metrics"]
                                candidate_metrics["exact_gt_validation"] = {
                                    "configured_cumulative_batches": list(exact_batch_limits),
                                    "accepted_batch_limit": active_exact_batch_limit,
                                    "total_exact_candidates_tested": total_exact_candidates_tested,
                                    "gt_valid_candidate_count": len(gt_valid_candidates),
                                    "candidate_records": exact_candidate_records,
                                    "accepted_candidate_kind": candidate_record[
                                        "candidate_kind"
                                    ],
                                    "accepted_validation_stage": candidate_record[
                                        "validation_stage"
                                    ],
                                    "rescue_candidate_counts": dict(
                                        rescue_candidate_counts
                                    ),
                                    "rescue_generation_failures": (
                                        rescue_generation_failures
                                    ),
                                    "candidate_realized_regimes_considered": [
                                        record["realized_regime"]
                                        for record in exact_candidate_records
                                        if record["gt_valid"]
                                    ],
                                    "candidate_pool_exhausted": False,
                                    "candidate_accounting": {
                                        **candidate_accounting,
                                        "accepted_candidate_source": (
                                            f"{candidate_record['candidate_kind']}:"
                                            f"{candidate_record['validation_stage']}"
                                        ),
                                    },
                                }
                                candidate_metrics[
                                    "gt_valid_soft_selection_rank"
                                ] = selection_rank
                                candidate_graph = candidate_record["graph"]
                                candidate_catalog = candidate_record["catalog"]
                                candidate_overlap = candidate_record["overlap"]
                                try:
                                    adapter.load_snapshot(configuration_snapshot)
                                    adapter.place_robots_at_trajectory_frame(
                                        candidate_trajectories, 0
                                    )
                                    candidate_calibration = adapter.calibrated_floor_bounds(
                                        int(candidate_metrics["floor_index"]),
                                        float(config["bev"]["world_meters_per_pixel"]),
                                        float(config["bev"]["bounds_margin_m"]),
                                    )
                                    candidate_snapshot = adapter.dump_snapshot()
                                    before_rollout_started = time.perf_counter()
                                    candidate_before = adapter.playback_trajectories(
                                        candidate_trajectories,
                                        int(candidate_metrics["floor_index"]),
                                        candidate_calibration,
                                    )
                                    episode_timing["before_rollout_s"] = (
                                        episode_timing.get("before_rollout_s", 0.0)
                                        + time.perf_counter() - before_rollout_started
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
                                    trajectory_metrics["outer_motion_attempt_index"] = placement_attempt
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
                                        "soft_selection_rank": selection_rank,
                                        "selection_stage": "full_before_rollout",
                                        "candidate_kind": candidate_metrics.get(
                                            "nested_trajectory_sets", {}
                                        ).get("candidate_kind", "base"),
                                        "reason": candidate_error.reason,
                                        "details": candidate_error.details,
                                        "trajectory_summary": {
                                            "observation_regime": candidate_metrics.get(
                                                "observation_regime"
                                            ),
                                            "soft_selection": candidate_metrics.get(
                                                "realized_regime_soft_selection"
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
                                {
                                    "attempt": placement_attempt,
                                    "sampling_round": sampling_round,
                                    "attempt_in_round": attempt_in_round,
                                    **error.details,
                                },
                            )
                    if before is None:
                        raise SampleRejected(
                            "episode_before_attempts_exhausted",
                            {
                                "scene_id": selected_scene,
                                "configuration_id": configuration_id,
                                "episode_id": episode_id,
                                "sampling_rounds": episode_sampling_rounds,
                                "total_attempts": (
                                    placement_attempts_per_round
                                    * episode_sampling_rounds
                                ),
                            },
                        )
                    after = None
                    intervention = None
                    environment_after = None
                    after_catalog = None
                    after_snapshot = None
                    qa_results = None
                    post_render_effect = None
                    visible_ids = set(visibility_table["eligible_target_ids"])
                    available_intervention_types = _available_intervention_types(
                        w0_catalog,
                        visible_ids,
                        used_targets,
                    )
                    fixed_intervention_type = _choose_quota_intervention_type(
                        config["intervention"]["type_weights"],
                        accepted_intervention_types,
                        available_intervention_types,
                        stable_seed(episode_seed, "intervention_type"),
                    )
                    event_attempts_per_round = min(
                        int(config["intervention"]["maximum_attempts"]),
                        int(config["intervention"]["post_render_effect"]["maximum_resample_attempts"]),
                    )
                    event_attempt_count = min(
                        int(config["intervention"]["maximum_attempts"]),
                        event_attempts_per_round * episode_sampling_rounds,
                    )
                    for event_attempt in range(event_attempt_count):
                        sampling_round = event_attempt // event_attempts_per_round
                        attempt_in_round = event_attempt % event_attempts_per_round
                        _write_status(
                            root,
                            status="running",
                            stage="sample_episode_intervention",
                            scene_id=selected_scene,
                            configuration_id=configuration_id,
                            episode_id=episode_id,
                            attempt=event_attempt,
                            sampling_round=sampling_round,
                            attempt_in_round=attempt_in_round,
                            accepted_configurations=accepted_configurations,
                            accepted_episodes=accepted_episodes,
                            sampling_retry_epoch=sampling_retry_epoch,
                        )
                        adapter.load_snapshot(w0_snapshot)
                        try:
                            intervention_started = time.perf_counter()
                            intervention = adapter.apply_atomic_intervention(
                                stable_seed(episode_seed, "intervention", event_attempt),
                                forced_type=fixed_intervention_type,
                                excluded_target_ids=tuple(sorted(used_targets)),
                                visible_target_ids=tuple(visibility_table["eligible_target_ids"]),
                            )
                            environment_after, _ = _render_environment_floors(adapter, config)
                            episode_timing["intervention_and_environment_bev_s"] = (
                                episode_timing.get("intervention_and_environment_bev_s", 0.0)
                                + time.perf_counter() - intervention_started
                            )
                            after_rollout_started = time.perf_counter()
                            after = adapter.playback_trajectories(
                                trajectories,
                                int(trajectory_metrics["floor_index"]),
                                world_calibration,
                            )
                            episode_timing["after_rollout_s"] = (
                                episode_timing.get("after_rollout_s", 0.0)
                                + time.perf_counter() - after_rollout_started
                            )
                            post_render_effect = _post_render_intervention_effect(
                                intervention["event"].target_instance_id,
                                w0_catalog,
                                before["robot_views"], after["robot_views"], config,
                                intervention["event"].intervention_type,
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
                                {
                                    "attempt": event_attempt,
                                    "sampling_round": sampling_round,
                                    "attempt_in_round": attempt_in_round,
                                    **error.details,
                                },
                            )
                    if after is None or intervention is None:
                        raise SampleRejected(
                            "intervention_attempts_exhausted",
                            {
                                "scene_id": selected_scene,
                                "configuration_id": configuration_id,
                                "episode_id": episode_id,
                                "sampling_rounds": episode_sampling_rounds,
                                "total_attempts": event_attempt_count,
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
                    serialization_started = time.perf_counter()
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
                        appearance_name = "robot_appearance_summary.png"
                        save_robot_appearance_summary(
                            inspection_root / appearance_name,
                            before["world_bev"]["rgb"][0],
                            before["world_bev"]["instance"][0],
                        )
                        image_names.append(appearance_name)
                        write_html_summary(
                            inspection_root,
                            f"{selected_scene}/{configuration_id}/{episode_id}",
                            {
                                "qa": qa_results,
                                "event": intervention["event"],
                                "trajectory": trajectory_metrics,
                                "sibling_episode_diversity": sibling_diversity,
                                "robot_appearance_variants": robot_appearance_metadata(),
                            },
                            image_names,
                        )
                        episode_timing["serialization_and_inspection_s"] = (
                            time.perf_counter() - serialization_started
                        )
                        episode_timing["total_episode_s"] = (
                            time.perf_counter() - episode_started
                        )
                        generation_metrics = {
                            "overlap": graph,
                            "trajectory": trajectory_metrics,
                            "before": before["metrics"],
                            "after": after["metrics"],
                            "intervention_attempt": intervention["attempt"],
                            "fixed_intervention_type": fixed_intervention_type.value,
                            "intervention_visibility": visibility_table,
                            "post_render_intervention_effect": post_render_effect,
                            "sibling_episode_diversity": sibling_diversity,
                            "runtime_s": episode_timing,
                            "sampling_retry_epoch": sampling_retry_epoch,
                        }
                        transaction.write_json(
                            "generation_metrics.json", generation_metrics
                        )
                        episode_bytes = sum(
                            path.stat().st_size
                            for path in transaction.staging.rglob("*")
                            if path.is_file()
                        )
                        generation_metrics["storage"] = {
                            "episode_bytes_before_final_metadata_rewrite": episode_bytes
                        }
                        transaction.write_json(
                            "generation_metrics.json", generation_metrics
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
            "target_overlap_regime_distribution": dict(
                config["placement"]["observation_regime_weights"]
            ),
            "realized_overlap_regime_fractions": {
                name: realized_regime_counts[name] / max(1, sum(realized_regime_counts.values()))
                for name in sorted(config["placement"]["observation_regime_weights"])
            },
            "requested_overlap_regime_counts_by_split": {
                name: dict(values)
                for name, values in requested_regime_counts_by_split.items()
            },
            "realized_overlap_regime_counts_by_split": {
                name: dict(values)
                for name, values in realized_regime_counts_by_split.items()
            },
            "scope": "single-scene shard generation; full production is never started implicitly",
            "sampling_retry_epoch": sampling_retry_epoch,
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
            "sampling_retry_epoch": sampling_retry_epoch,
        }
        if isinstance(error, SampleRejected):
            failure["rejection_details"] = error.details
        dump_json(root / "generation_failure.json", failure)
        _write_status(root, **failure, stage="failed")
        raise
    finally:
        adapter.close()
