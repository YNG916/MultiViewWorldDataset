from __future__ import annotations

from collections import Counter
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from multi_view_world_dataset.errors import ConfigurationError
from multi_view_world_dataset.utils.serialization import dump_json

# Only datasets created before these keys were introduced use this compatibility
# map. New roots persist and validate the same values in resolved_config.yaml.
_LEGACY_COLLAPSE_THRESHOLDS = {
    "dominant_start_region_fraction_max": 0.90,
    "direct_path_fraction_max": 0.90,
    "parallel_episode_fraction_max": 0.80,
    "compact_start_episode_fraction_max": 0.90,
    "complete_triangle_keyframe_fraction_max": 0.90,
    "zero_visible_intervention_fraction_max": 0.00,
    "dominant_intervention_room_fraction_max": 0.90,
    "dominant_intervention_category_fraction_max": 0.90,
    "single_changed_object_configuration_fraction_max": 0.00,
    "temporal_union_connected_fraction_min": 0.10,
}


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "minimum": float(array.min()),
        "mean": float(array.mean()),
        "maximum": float(array.max()),
        "p10": float(np.quantile(array, 0.1)),
        "p50": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
    }


def _fraction(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    return {
        key: count / total
        for key, count in sorted(counter.items())
    } if total else {}


def _dominant_fraction(counter: Counter[str]) -> float:
    total = sum(counter.values())
    return max(counter.values(), default=0) / max(1, total)


def _topology(edge_count: int) -> str:
    return {
        0: "sparse",
        1: "single_edge",
        2: "chain",
        3: "triangle",
    }.get(edge_count, f"edges_{edge_count}")


def _resolved_config(root: Path) -> dict[str, Any]:
    path = root / "resolved_config.yaml"
    if not path.is_file():
        raise ConfigurationError(
            f"Dataset has no self-contained resolved_config.yaml: {root}"
        )
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ConfigurationError(f"Invalid resolved configuration: {path}")
    return config


def _warning(
    warnings: list[dict[str, Any]],
    name: str,
    value: float,
    maximum: float,
) -> None:
    if value > maximum:
        warnings.append({
            "name": name,
            "value": value,
            "configured_maximum": maximum,
        })


def summarize_generated_dataset(
    dataset_root: str | Path,
    *,
    output_path: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Aggregate finalized Dataset-v1.1 outputs without running the simulator."""
    root = Path(dataset_root).expanduser().resolve()
    if not (root / "dataset_meta.json").is_file():
        raise ConfigurationError(f"Not a generated dataset root: {root}")
    config = _resolved_config(root)
    diagnostics = config["sampling_diagnostics"]
    configured_thresholds = diagnostics.get("collapse_thresholds")
    thresholds = (
        configured_thresholds
        if configured_thresholds is not None
        else _LEGACY_COLLAPSE_THRESHOLDS
    )
    minimum_samples = int(
        diagnostics.get("minimum_samples_for_distribution_warnings", 10)
    )
    parallel_min = float(
        diagnostics.get("parallel_path_direction_similarity_min", 0.90)
    )

    episode_paths = sorted(
        path.parent
        for path in (root / "episodes").glob("*/*/episode_*/meta.json")
    )
    configuration_paths = sorted(
        path.parent
        for path in (root / "configurations").glob("*/*/config_meta.json")
    )

    start_regions: Counter[str] = Counter()
    episode_regions: Counter[str] = Counter()
    traversed_regions: Counter[str] = Counter()
    path_families: Counter[str] = Counter()
    requested_regimes: Counter[str] = Counter()
    realized_regimes: Counter[str] = Counter()
    overlap_topologies: Counter[str] = Counter()
    target_rooms: Counter[str] = Counter()
    target_categories: Counter[str] = Counter()
    intervention_types: Counter[str] = Counter()
    changed_categories: Counter[str] = Counter()
    changed_rooms: Counter[str] = Counter()
    rejection_stages: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    rejection_stage_reasons: Counter[str] = Counter()

    start_distances: list[float] = []
    spatial_coverages: list[float] = []
    path_lengths: list[float] = []
    displacements: list[float] = []
    tortuosities: list[float] = []
    cumulative_yaw_changes: list[float] = []
    heading_differences: list[float] = []
    path_direction_similarities: list[float] = []
    view_connectivity_proxies: list[float] = []
    minimum_separations: list[float] = []
    connected_fractions: list[float] = []
    overlap_values: list[float] = []
    visible_robot_counts: list[float] = []
    visible_frame_counts: list[float] = []
    visible_pixels: list[float] = []
    changed_pixels: list[float] = []
    rgb_deltas: list[float] = []
    configuration_changed_counts: list[float] = []
    camera_translation_errors: list[float] = []
    camera_rotation_errors: list[float] = []
    multimodal_alignment: list[float] = []
    storage_bytes: list[float] = []

    union_connected = 0
    parallel_episodes = 0
    compact_start_episodes = 0
    zero_visible_interventions = 0
    stable_target_mappings = 0
    observation_metadata_complete = 0
    bev_calibration_complete = 0

    taxonomy = _load_json(root / "taxonomy.json", {})
    instance_catalogs = taxonomy.get("instance_catalogs", {})
    stable_instances = {
        str(entry.get("object_state_id")): int(entry.get("public_instance_id"))
        for entries in instance_catalogs.values()
        for entry in entries
        if entry.get("object_state_id") is not None
        and entry.get("public_instance_id") is not None
    }

    for episode in episode_paths:
        metrics = _load_json(episode / "generation_metrics.json", {})
        events = _load_json(episode / "events.json", [])
        trajectory = metrics.get("trajectory", {})
        joint = trajectory.get("joint_diversity", {})

        regions = [str(value) for value in trajectory.get("start_region_ids", [])]
        start_regions.update(regions)
        for region in set(regions):
            episode_regions[region] += 1
        for values in trajectory.get("traversed_region_ids", {}).values():
            traversed_regions.update(str(value) for value in values)

        requested_regimes[str(
            trajectory.get("requested_observation_regime", "unknown")
        )] += 1
        realized_regimes[str(
            trajectory.get("observation_regime", "unknown")
        )] += 1
        spatial_coverages.append(
            float(joint.get("spatial_coverage_bbox_area_m2", 0.0))
        )
        heading_differences.append(
            float(joint.get("mean_pairwise_heading_difference_rad", 0.0))
        )
        similarity = float(
            joint.get("mean_path_direction_similarity", 0.0)
        )
        path_direction_similarities.append(similarity)
        view_connectivity_proxies.append(float(
            joint.get("temporal_camera_view_connectivity_proxy", 0.0)
        ))
        parallel_episodes += int(similarity >= parallel_min)
        minimum_separations.append(
            float(
                joint.get(
                    "minimum_inter_robot_distance_m",
                    trajectory.get("minimum_pairwise_distance_m", 0.0),
                )
            )
        )
        compact_start_episodes += int(
            float(joint.get("maximum_inter_robot_distance_m", float("inf")))
            <= float(diagnostics.get(
                "compact_start_max_pairwise_distance_m", 3.0
            ))
        )

        for robot in trajectory.get("robots", {}).values():
            path_families[str(robot.get("path_family", "unknown"))] += 1
            path_lengths.append(float(robot.get("arc_path_length_m", 0.0)))
            displacements.append(
                float(robot.get("start_end_displacement_m", 0.0))
            )
            tortuosities.append(float(robot.get("tortuosity", 0.0)))
            cumulative_yaw_changes.append(
                float(robot.get("cumulative_absolute_yaw_change_rad", 0.0))
            )

        trajectories_path = episode / "trajectories.npz"
        if trajectories_path.is_file():
            with np.load(trajectories_path, allow_pickle=False) as arrays:
                keys = sorted(
                    key for key in arrays.files
                    if key.endswith("_base_to_world")
                )
                starts = [
                    np.asarray(arrays[key][0, :2, 3], dtype=np.float64)
                    for key in keys
                ]
            start_distances.extend(
                float(np.linalg.norm(left - right))
                for left, right in combinations(starts, 2)
            )

        overlap = trajectory.get("temporal_overlap", {})
        connected_fractions.append(
            float(overlap.get("connected_fraction", 0.0))
        )
        union_connected += int(
            bool(overlap.get("checks", {}).get("union_graph_connected", False))
        )
        for keyframe in overlap.get("keyframes", []):
            overlap_topologies[_topology(len(keyframe.get("edges", [])))] += 1
            overlap_values.extend(
                float(value)
                for value in keyframe.get("overlaps", {}).values()
            )

        if events:
            event = events[0]
            intervention_types[str(
                event.get("intervention_type", "unknown")
            )] += 1
            target_id = str(event.get("target_instance_id", ""))
            before_state = event.get("before_object_state", {})
            target_rooms[str(before_state.get("room_id") or "unknown")] += 1
            target_categories[str(
                before_state.get("category") or "unknown"
            )] += 1
            stable_target_mappings += int(target_id in stable_instances)
            target = (
                metrics.get("intervention_visibility", {})
                .get("objects", {})
                .get(target_id, {})
            )
            participants = int(target.get("participating_robot_count", 0))
            visible_robot_counts.append(float(participants))
            visible_frame_counts.append(
                float(target.get("qualifying_frame_count", 0))
            )
            visible_pixels.append(float(target.get("maximum_pixels", 0)))
            zero_visible_interventions += int(participants == 0)
        effect = metrics.get("post_render_intervention_effect", {})
        changed_pixels.append(float(effect.get("changed_pixels", 0)))
        rgb_deltas.append(float(effect.get("mean_rgb_delta", 0.0)))

        before_qa = metrics.get("before", {})
        camera_translation_errors.append(
            float(before_qa.get("maximum_capture_translation_error_m", 0.0))
        )
        camera_rotation_errors.append(
            float(before_qa.get("maximum_capture_rotation_error_rad", 0.0))
        )
        multimodal_alignment.append(
            float(before_qa.get("minimum_multimodal_alignment_fraction", 0.0))
        )
        observations = _load_json(episode / "observations_before.json", [])
        if observations:
            required = {
                "camera_to_world", "world_to_camera",
                "geometry_pixel_intrinsics", "modality_camera_to_world",
            }
            observation_metadata_complete += int(all(
                required <= set(record.get("camera", {}))
                and (
                    "rgb_pixel_intrinsics" in record.get("camera", {})
                    or "pixel_intrinsics" in record.get("camera", {})
                )
                for record in observations
            ))
        bev_path = episode / "bev" / "world_before.npz"
        if bev_path.is_file():
            with np.load(bev_path, allow_pickle=False) as arrays:
                keys = set(arrays.files)
            required = {
                "calibration_world_bounds",
                "calibration_pixel_to_world",
                "calibration_world_to_pixel",
                "calibration_meters_per_pixel",
                "calibration_floor_z",
            }
            bev_calibration_complete += int(required <= keys)

        storage_bytes.append(float(sum(
            path.stat().st_size
            for path in episode.rglob("*")
            if path.is_file()
        )))

    for configuration_path in configuration_paths:
        configuration = _load_json(
            configuration_path / "config_meta.json", {}
        )
        metadata = configuration.get("metadata", {})
        changed_ids = metadata.get("changed_instance_ids", [])
        configuration_changed_counts.append(float(len(changed_ids)))
        world_objects = {
            str(obj.get("instance_id")): obj
            for obj in configuration.get("world_state", {}).get("objects", [])
        }
        for changed_id in changed_ids:
            obj = world_objects.get(str(changed_id), {})
            changed_categories[str(obj.get("category") or "unknown")] += 1
            changed_rooms[str(obj.get("room_id") or "unknown")] += 1

    rejects_path = root / "rejects.jsonl"
    if rejects_path.is_file():
        for line in rejects_path.read_text(encoding="utf-8").splitlines():
            try:
                reject = json.loads(line)
            except json.JSONDecodeError:
                continue
            scope = str(reject.get("scope", "unknown"))
            stage = scope.split(":", 1)[0]
            reason = str(reject.get("reason", "unknown"))
            rejection_stages[stage] += 1
            rejection_reasons[reason] += 1
            rejection_stage_reasons[f"{stage}|{reason}"] += 1

    episode_count = len(episode_paths)
    configuration_count = len(configuration_paths)
    total_keyframes = sum(overlap_topologies.values())
    report: dict[str, Any] = {
        "dataset_root": str(root),
        "dataset_metadata": _load_json(root / "dataset_meta.json", {}),
        "finalized_episode_count": episode_count,
        "finalized_configuration_count": configuration_count,
        "spatial": {
            "robot_start_region_counts": dict(start_regions),
            "robot_start_region_fractions": _fraction(start_regions),
            "episode_region_counts": dict(episode_regions),
            "episode_room_coverage_fractions": {
                key: value / max(1, episode_count)
                for key, value in sorted(episode_regions.items())
            },
            "traversed_region_counts": dict(traversed_regions),
            "pairwise_start_distance_m": _summary(start_distances),
            "spatial_coverage_bbox_area_m2": _summary(spatial_coverages),
            "compact_start_episode_fraction": (
                compact_start_episodes / max(1, episode_count)
            ),
        },
        "trajectory": {
            "path_family_counts": dict(path_families),
            "path_family_fractions": _fraction(path_families),
            "arc_path_length_m": _summary(path_lengths),
            "start_end_displacement_m": _summary(displacements),
            "tortuosity": _summary(tortuosities),
            "cumulative_absolute_yaw_change_rad": _summary(
                cumulative_yaw_changes
            ),
            "mean_pairwise_heading_difference_rad": _summary(
                heading_differences
            ),
            "mean_path_direction_similarity": _summary(
                path_direction_similarities
            ),
            "temporal_camera_view_connectivity_proxy": _summary(
                view_connectivity_proxies
            ),
            "minimum_inter_robot_distance_m": _summary(minimum_separations),
            "parallel_episode_fraction": (
                parallel_episodes / max(1, episode_count)
            ),
            "requested_regime_counts": dict(requested_regimes),
            "realized_regime_counts": dict(realized_regimes),
        },
        "overlap": {
            "topology_keyframe_counts": dict(overlap_topologies),
            "topology_keyframe_fractions": _fraction(overlap_topologies),
            "pairwise_omega": _summary(overlap_values),
            "temporal_connected_fraction": _summary(connected_fractions),
            "union_connected_episode_count": union_connected,
            "union_connected_episode_fraction": (
                union_connected / max(1, episode_count)
            ),
        },
        "intervention": {
            "type_counts": dict(intervention_types),
            "target_room_counts": dict(target_rooms),
            "target_category_counts": dict(target_categories),
            "visible_robot_count": _summary(visible_robot_counts),
            "visible_frame_count": _summary(visible_frame_counts),
            "maximum_visible_pixels": _summary(visible_pixels),
            "zero_visible_target_count": zero_visible_interventions,
            "changed_pixels": _summary(changed_pixels),
            "mean_rgb_delta": _summary(rgb_deltas),
        },
        "configuration": {
            "changed_object_count": _summary(configuration_changed_counts),
            "changed_category_counts": dict(changed_categories),
            "changed_room_counts": dict(changed_rooms),
        },
        "rejections": {
            "stage_counts": dict(rejection_stages),
            "reason_counts": dict(rejection_reasons),
            "stage_reason_counts": dict(rejection_stage_reasons),
        },
        "identity_and_calibration": {
            "stable_public_instance_catalog_count": len(stable_instances),
            "intervention_targets_with_stable_mapping": stable_target_mappings,
            "complete_observation_metadata_episode_count": (
                observation_metadata_complete
            ),
            "complete_world_bev_calibration_episode_count": (
                bev_calibration_complete
            ),
            "maximum_capture_translation_error_m": _summary(
                camera_translation_errors
            ),
            "maximum_capture_rotation_error_rad": _summary(
                camera_rotation_errors
            ),
            "minimum_multimodal_alignment_fraction": _summary(
                multimodal_alignment
            ),
        },
        "storage": {
            "bytes_per_episode": _summary(storage_bytes),
            "total_episode_bytes": int(sum(storage_bytes)),
        },
        "collapse_thresholds": {
            **thresholds,
            "minimum_samples_for_distribution_warnings": minimum_samples,
            "parallel_path_direction_similarity_min": parallel_min,
            "compact_start_max_pairwise_distance_m": diagnostics.get(
                "compact_start_max_pairwise_distance_m", 3.0
            ),
            "source": (
                "resolved_config" if configured_thresholds else "legacy_compatibility"
            ),
        },
    }

    warnings: list[dict[str, Any]] = []
    if episode_count >= minimum_samples:
        _warning(
            warnings, "dominant_start_region",
            _dominant_fraction(start_regions),
            float(thresholds["dominant_start_region_fraction_max"]),
        )
        _warning(
            warnings, "direct_path_collapse",
            path_families.get("direct", 0) / max(1, sum(path_families.values())),
            float(thresholds["direct_path_fraction_max"]),
        )
        _warning(
            warnings, "parallel_motion_collapse",
            parallel_episodes / max(1, episode_count),
            float(thresholds["parallel_episode_fraction_max"]),
        )
        _warning(
            warnings, "compact_start_triangle_collapse",
            compact_start_episodes / max(1, episode_count),
            float(thresholds["compact_start_episode_fraction_max"]),
        )
        union_fraction = union_connected / max(1, episode_count)
        union_minimum = float(
            thresholds["temporal_union_connected_fraction_min"]
        )
        if union_fraction < union_minimum:
            warnings.append({
                "name": "temporal_union_connectivity_collapse",
                "value": union_fraction,
                "configured_minimum": union_minimum,
            })
        _warning(
            warnings, "complete_triangle_overlap_collapse",
            overlap_topologies.get("triangle", 0) / max(1, total_keyframes),
            float(thresholds[
                "complete_triangle_keyframe_fraction_max"
            ]),
        )
        _warning(
            warnings, "zero_visible_interventions",
            zero_visible_interventions / max(1, episode_count),
            float(thresholds[
                "zero_visible_intervention_fraction_max"
            ]),
        )
        _warning(
            warnings, "dominant_intervention_room",
            _dominant_fraction(target_rooms),
            float(thresholds[
                "dominant_intervention_room_fraction_max"
            ]),
        )
        _warning(
            warnings, "dominant_intervention_category",
            _dominant_fraction(target_categories),
            float(thresholds[
                "dominant_intervention_category_fraction_max"
            ]),
        )
    if configuration_count >= minimum_samples:
        single = sum(value <= 1 for value in configuration_changed_counts)
        _warning(
            warnings, "single_changed_object_configuration_collapse",
            single / max(1, configuration_count),
            float(thresholds[
                "single_changed_object_configuration_fraction_max"
            ]),
        )
    report["collapse_warnings"] = warnings
    report["collapse_warning_evaluation_deferred"] = {
        "episodes": episode_count < minimum_samples,
        "configurations": configuration_count < minimum_samples,
    }

    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else root / "dataset_diagnostics.json"
    )
    dump_json(destination, report)
    return destination, report
