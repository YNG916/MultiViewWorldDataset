from __future__ import annotations

from copy import deepcopy
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from multi_view_world_dataset.errors import ConfigurationError


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml_config(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a profile with relative single-parent inheritance and validate frozen v1 settings."""
    config_path = Path(path).expanduser().resolve()
    seen = set() if _seen is None else _seen
    if config_path in seen:
        raise ConfigurationError(f"Cyclic config inheritance at {config_path}")
    seen.add(config_path)
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ConfigurationError(f"Configuration root must be a mapping: {config_path}")
    parent = raw.pop("extends", None)
    merged = raw
    if parent:
        merged = _deep_merge(load_yaml_config(config_path.parent / parent, seen), raw)
    validate_config(merged)
    return merged


def validate_config(config: dict[str, Any]) -> None:
    errors: list[str] = []
    dataset = config.get("dataset", {})
    camera = config.get("camera", {})
    bev = config.get("bev", {})
    intervention = config.get("intervention", {})
    generation = config.get("generation", {})
    trajectory = config.get("trajectory", {})
    navigation = config.get("navigation", {})
    eligibility_manifest = dataset.get("scene_eligibility_manifest")
    if not isinstance(eligibility_manifest, str) or not eligibility_manifest.strip():
        errors.append("dataset.scene_eligibility_manifest must be a non-empty path")
    if dataset.get("robots") != 3:
        errors.append("Dataset v1 requires exactly 3 robots")
    if config.get("profile") not in {"smoke", "integration"} and dataset.get("frames") != 60:
        errors.append("Dataset v1 non-development profiles require T=60")
    frozen_camera = {
        "rgb_width": 896,
        "rgb_height": 512,
        "geometry_width": 448,
        "geometry_height": 256,
        "hfov_deg": 70.0,
        "pitch_deg": -5.0,
        "roll_deg": 0.0,
        "near_m": 0.1,
        "far_m": 15.0,
    }
    for key, expected in frozen_camera.items():
        if camera.get(key) != expected:
            errors.append(f"Frozen camera setting {key} must be {expected!r}")
    if camera.get("heights_m") != [0.8, 1.0, 1.2, 1.4]:
        errors.append("Frozen camera heights must be [0.8, 1.0, 1.2, 1.4]")
    if bev.get("environment_meters_per_pixel") != 0.02:
        errors.append("B_env resolution must be 0.02 m/px")
    if bev.get("world_meters_per_pixel") != 0.04:
        errors.append("B_world resolution must be 0.04 m/px")
    weights = intervention.get("type_weights", {})
    if abs(sum(float(v) for v in weights.values()) - 1.0) > 1e-8:
        errors.append("Intervention type weights must sum to 1")
    if intervention.get("application_mode") != "pre_rollout":
        errors.append("Dataset v1 only supports pre_rollout interventions")
    configuration_sampling = config.get("configuration_sampling", {})
    minimum_changed = configuration_sampling.get("minimum_changed_objects")
    maximum_changed = configuration_sampling.get("maximum_changed_objects")
    if (
        not isinstance(minimum_changed, int)
        or not isinstance(maximum_changed, int)
        or minimum_changed < 2
        or maximum_changed > 6
        or minimum_changed > maximum_changed
    ):
        errors.append("Dataset-v1.1 DynamicConfiguration must randomize 2-6 objects")
    selected_scenes = config.get("production", {}).get("selected_scenes")
    if selected_scenes is not None and (
        not isinstance(selected_scenes, list)
        or not selected_scenes
        or len(set(map(str, selected_scenes))) != len(selected_scenes)
    ):
        errors.append("production.selected_scenes must be null or a non-empty unique list")
    for key in (
        "native_relation_high_level_attempts",
        "native_relation_low_level_attempts",
    ):
        value = generation.get(key)
        if not isinstance(value, int) or value < 1:
            errors.append(f"Generation setting {key} must be a positive integer")
    restore_tolerance = generation.get("snapshot_restore_tolerance")
    if not isinstance(restore_tolerance, (int, float)) or restore_tolerance <= 0:
        errors.append("Generation setting snapshot_restore_tolerance must be positive")
    family_weights = trajectory.get("path_family_weights", {})
    required_families = {"direct", "one_waypoint", "two_waypoint"}
    if set(family_weights) != required_families:
        errors.append("Trajectory path_family_weights must define direct, one_waypoint, and two_waypoint")
    elif (
        any(float(value) < 0.0 for value in family_weights.values())
        or abs(sum(float(value) for value in family_weights.values()) - 1.0) > 1.0e-8
    ):
        errors.append("Trajectory path family weights must be non-negative and sum to 1")
    minimum_waypoint = trajectory.get("minimum_waypoint_trajectories")
    if (
        not isinstance(minimum_waypoint, int)
        or not 0 <= minimum_waypoint <= 3
    ):
        errors.append(
            "Trajectory minimum_waypoint_trajectories must be an integer in [0,3]"
        )
    for key in (
        "initial_heading_tolerance_deg",
        "line_validation_spacing_m",
        "smoothing_validation_spacing_m",
        "candidate_pool_size",
        "joint_pool_rounds",
        "maximum_joint_valid_candidates",
        "trajectory_sets_per_placement",
        "sampling_maximum_attempts",
    ):
        value = trajectory.get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            errors.append(f"Trajectory setting {key} must be positive")
    if trajectory.get("initial_heading_policy") not in {"trajectory_tangent", "fixed_prior"}:
        errors.append("Trajectory initial_heading_policy must be trajectory_tangent or fixed_prior")
    trajectory_set_count = trajectory.get("trajectory_sets_per_placement")
    if not isinstance(trajectory_set_count, int) or not 8 <= trajectory_set_count <= 16:
        errors.append("trajectory_sets_per_placement must be an integer in [8,16]")
    hybrid_count = trajectory.get("maximum_complementary_hybrid_candidates")
    bridge_count = trajectory.get("maximum_measured_overlap_bridge_candidates")
    feedback_rounds = trajectory.get("maximum_gt_feedback_mutation_rounds")
    candidate_counts_valid = (
        isinstance(hybrid_count, int)
        and hybrid_count >= 0
        and isinstance(bridge_count, int)
        and bridge_count >= 0
    )
    if (
        not candidate_counts_valid
        or isinstance(trajectory_set_count, int)
        and trajectory_set_count + hybrid_count + bridge_count > 16
    ):
        errors.append(
            "base trajectory sets plus complementary and measured-overlap "
            "bridge candidates must total at most 16"
        )
    if not isinstance(feedback_rounds, int) or not 1 <= feedback_rounds <= 3:
        errors.append(
            "maximum_gt_feedback_mutation_rounds must be an integer in [1,3]"
        )
    for key in (
        "measured_overlap_bridge_pool_size",
        "measured_overlap_bridge_sampling_attempts",
        "measured_overlap_bridge_target_depth_m",
        "measured_overlap_bridge_guide_distance_m",
        "measured_overlap_bridge_guide_scale_m",
    ):
        value = trajectory.get(key)
        if not isinstance(value, (int, float)) or float(value) <= 0.0:
            errors.append(f"Trajectory setting {key} must be positive")
    for key in (
        "measured_overlap_bridge_heading_floor",
        "measured_overlap_bridge_guide_floor",
    ):
        value = trajectory.get(key)
        if (
            not isinstance(value, (int, float))
            or not 0.0 < float(value) <= 1.0
        ):
            errors.append(f"Trajectory setting {key} must lie in (0,1]")

    strengths = trajectory.get("smoothing_strengths", ())
    if not strengths or any(not 0.0 <= float(value) <= 1.0 for value in strengths):
        errors.append("Trajectory smoothing_strengths must lie in [0,1]")
    preflight = trajectory.get("overlap_preflight", {})
    regimes = config.get("placement", {}).get("observation_regime_weights", {})
    realized_regime_soft_weight = preflight.get("realized_regime_soft_weight")
    if (
        not isinstance(realized_regime_soft_weight, (int, float))
        or not isfinite(float(realized_regime_soft_weight))
        or float(realized_regime_soft_weight) < 0.0
    ):
        errors.append("Trajectory realized_regime_soft_weight must be finite and non-negative")
    required_regimes = {"dense_shared", "partial_chain", "exploratory"}
    coverage_saturations = trajectory.get("regime_coverage_saturation_m2", {})
    if set(coverage_saturations) != required_regimes or any(
        not isinstance(value, (int, float)) or float(value) <= 0.0
        for value in coverage_saturations.values()
    ):
        errors.append("Trajectory regime_coverage_saturation_m2 must define positive values for all regimes")
    heading_prior_weights = trajectory.get("regime_initial_heading_prior_weights", {})
    if set(heading_prior_weights) != required_regimes or any(
        not isinstance(value, (int, float))
        or not isfinite(float(value))
        or float(value) < 0.0
        for value in heading_prior_weights.values()
    ):
        errors.append(
            "Trajectory regime_initial_heading_prior_weights must define finite "
            "non-negative values for all regimes"
        )
    heading_floors = trajectory.get(
        "regime_initial_heading_probability_floor", {}
    )
    if set(heading_floors) != required_regimes or any(
        not isinstance(value, (int, float)) or not 0.0 < float(value) <= 1.0
        for value in heading_floors.values()
    ):
        errors.append(
            "Trajectory regime_initial_heading_probability_floor must define "
            "values in (0,1] for all regimes"
        )
    guide_scales = trajectory.get("regime_trajectory_guide_soft_scale_m", {})
    if set(guide_scales) != required_regimes or any(
        not isinstance(value, (int, float))
        or not isfinite(float(value))
        or float(value) <= 0.0
        for value in guide_scales.values()
    ):
        errors.append(
            "Trajectory regime_trajectory_guide_soft_scale_m must define finite "
            "positive values for all regimes"
        )
    guide_floors = trajectory.get("regime_trajectory_guide_probability_floor", {})
    if set(guide_floors) != required_regimes or any(
        not isinstance(value, (int, float)) or not 0.0 < float(value) <= 1.0
        for value in guide_floors.values()
    ):
        errors.append(
            "Trajectory regime_trajectory_guide_probability_floor must define "
            "values in (0,1] for all regimes"
        )
    view_weights = trajectory.get("regime_view_connectivity_weights", {})
    if set(view_weights) != required_regimes or any(
        not isinstance(value, (int, float))
        or not isfinite(float(value))
        or float(value) < 0.0
        for value in view_weights.values()
    ):
        errors.append(
            "Trajectory regime_view_connectivity_weights must define finite "
            "non-negative values for all regimes"
        )
    if set(regimes) != required_regimes or abs(sum(float(v) for v in regimes.values()) - 1.0) > 1e-8:
        errors.append("Placement observation_regime_weights must define the three regimes and sum to 1")
    connected_fractions = preflight.get("regime_connected_fraction_target", {})
    if set(connected_fractions) != required_regimes or any(
        not 0.0 <= float(value) <= 1.0 for value in connected_fractions.values()
    ):
        errors.append("Trajectory regime_connected_fraction_target must define [0,1] values for all regimes")
    shared_fractions = preflight.get("regime_shared_keyframe_fraction_target", {})
    if set(shared_fractions) != required_regimes or any(
        not 0.0 <= float(value) <= 1.0 for value in shared_fractions.values()
    ):
        errors.append(
            "Trajectory regime_shared_keyframe_fraction_target must define "
            "[0,1] values for all regimes"
        )
    isolation_limits = preflight.get(
        "regime_maximum_consecutive_isolated_keyframes", {}
    )
    if set(isolation_limits) != required_regimes or any(
        not isinstance(value, int) or value < 0 for value in isolation_limits.values()
    ):
        errors.append(
            "Trajectory regime_maximum_consecutive_isolated_keyframes must "
            "define non-negative integer values for all regimes"
        )
    dense_confirmation_count = preflight.get("dense_confirmation_keyframe_count")
    if (
        not isinstance(dense_confirmation_count, int)
        or dense_confirmation_count <= int(preflight.get("keyframe_count", 0))
    ):
        errors.append(
            "Trajectory dense_confirmation_keyframe_count must be an integer "
            "larger than overlap_preflight.keyframe_count"
        )
    participation_limits = preflight.get(
        "regime_minimum_participating_keyframes", {}
    )
    if set(participation_limits) != required_regimes or any(
        not isinstance(value, int) or value < 1
        for value in participation_limits.values()
    ):
        errors.append(
            "Trajectory regime_minimum_participating_keyframes must define "
            "positive integer values for all regimes"
        )
    isolation_fractions = preflight.get(
        "regime_maximum_isolation_fraction", {}
    )
    if set(isolation_fractions) != required_regimes or any(
        not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0
        for value in isolation_fractions.values()
    ):
        errors.append(
            "Trajectory regime_maximum_isolation_fraction must define [0,1] "
            "values for all regimes"
        )
    placement = config.get("placement", {})
    for key in ("floor_support_aabb_tolerance_m",):
        value = placement.get(key)
        if not isinstance(value, (int, float)) or float(value) < 0.0:
            errors.append(f"Placement setting {key} must be non-negative")

    obstacle_height = placement.get("dynamic_object_path_obstacle_height_m")
    if not isinstance(obstacle_height, (int, float)) or float(obstacle_height) <= 0.0:
        errors.append(
            "Placement dynamic_object_path_obstacle_height_m must be positive"
        )

    for key in (
        "footprint_yaw_bins",
        "footprint_physx_probe_count_per_category",
        "scene_worker_timeout_s",
        "scene_worker_max_attempts",
        "route_bank_target_size",
        "route_bank_minimum_size",
        "route_bank_max_raw_attempts",
        "route_candidate_attempts_per_raw",
        "route_bank_view_cohort_size",
        "top_triplets_for_exact_validation",
        "joint_route_search_budget",
        "se2_maximum_expansions",
        "cheap_visibility_keyframes",
        "cheap_visibility_ray_count",
        "start_blacklist_yaw_bins",
        "minimum_visible_intervention_candidates",
    ):
        value = navigation.get(key)
        if not isinstance(value, int) or value < 1:
            errors.append(f"Navigation setting {key} must be a positive integer")
    cohort_completion_probability = navigation.get(
        "route_bank_view_cohort_completion_probability"
    )
    if not isinstance(cohort_completion_probability, (int, float)) or not (
        0.0 <= float(cohort_completion_probability) <= 1.0
    ):
        errors.append(
            "navigation.route_bank_view_cohort_completion_probability must be in [0, 1]"
        )
    cohort_spatial_scale = navigation.get(
        "route_bank_view_cohort_spatial_scale_m"
    )
    if not isinstance(cohort_spatial_scale, (int, float)) or float(
        cohort_spatial_scale
    ) <= 0.0:
        errors.append(
            "navigation.route_bank_view_cohort_spatial_scale_m must be positive"
        )
    for key in (
        "route_bank_view_cohort_probability_floor",
        "route_bank_view_cohort_heading_probability_floor",
    ):
        value = navigation.get(key)
        if not isinstance(value, (int, float)) or not 0.0 < float(value) <= 1.0:
            errors.append(f"Navigation setting {key} must lie in (0, 1]")
    exact_batches = navigation.get("exact_validation_batches")
    if (
        not isinstance(exact_batches, list)
        or not exact_batches
        or any(not isinstance(value, int) or value < 1 for value in exact_batches)
        or any(
            right <= left
            for left, right in zip(exact_batches, exact_batches[1:])
        )
    ):
        errors.append(
            "navigation.exact_validation_batches must be strictly increasing positive integers"
        )
    if (
        isinstance(navigation.get("footprint_yaw_bins"), int)
        and navigation["footprint_yaw_bins"] < 4
    ):
        errors.append("navigation.footprint_yaw_bins must be at least 4")
    cheap_edge_threshold = navigation.get("cheap_visibility_edge_threshold")
    if not isinstance(cheap_edge_threshold, (int, float)) or not (
        0.0 <= float(cheap_edge_threshold) <= 1.0
    ):
        errors.append(
            "navigation.cheap_visibility_edge_threshold must be in [0, 1]"
        )
    visibility_priority_fraction = navigation.get(
        "joint_route_visibility_priority_fraction"
    )
    if not isinstance(visibility_priority_fraction, (int, float)) or not (
        0.0 <= float(visibility_priority_fraction) <= 1.0
    ):
        errors.append(
            "navigation.joint_route_visibility_priority_fraction must be in [0, 1]"
        )
    visibility_score_weight = navigation.get(
        "joint_route_visibility_score_weight"
    )
    if not isinstance(visibility_score_weight, (int, float)) or float(
        visibility_score_weight
    ) < 0.0:
        errors.append(
            "navigation.joint_route_visibility_score_weight must be non-negative"
        )
    if not isinstance(
        navigation.get("require_connected_start_regions"), bool
    ):
        errors.append(
            "navigation.require_connected_start_regions must be boolean"
        )
    cheap_isolated_fraction = navigation.get(
        "cheap_visibility_maximum_isolated_fraction"
    )
    if not isinstance(cheap_isolated_fraction, (int, float)) or not (
        0.0 <= float(cheap_isolated_fraction) < 1.0
    ):
        errors.append(
            "navigation.cheap_visibility_maximum_isolated_fraction "
            "must be in [0, 1)"
        )
    for key in (
        "footprint_safety_margin_m",
        "dynamic_obstacle_margin_m",
        "yaw_freedom_ranking_weight",
    ):
        value = navigation.get(key)
        if not isinstance(value, (int, float)) or float(value) < 0.0:
            errors.append(f"Navigation setting {key} must be non-negative")
    for key in (
        "footprint_sampling_spacing_fraction",
        "cheap_visibility_max_range_m",
        "start_blacklist_position_quantization_m",
        "se2_rotation_cost_cells",
    ):
        value = navigation.get(key)
        if not isinstance(value, (int, float)) or float(value) <= 0.0:
            errors.append(f"Navigation setting {key} must be positive")
    sparse_keyframes = navigation.get("sparse_physics_keyframes")
    yaw_fraction = navigation.get("route_seed_minimum_yaw_fraction")
    if (
        not isinstance(yaw_fraction, (int, float))
        or not 0.0 < float(yaw_fraction) <= 1.0
    ):
        errors.append("navigation.route_seed_minimum_yaw_fraction must lie in (0,1]")
    if (
        not isinstance(sparse_keyframes, list)
        or not sparse_keyframes
        or any(not isinstance(value, int) or value < 0 for value in sparse_keyframes)
    ):
        errors.append(
            "navigation.sparse_physics_keyframes must be non-negative integers"
        )

    configuration_sampling = config.get("configuration_sampling", {})
    minimum_changed = configuration_sampling.get("minimum_changed_objects")
    maximum_changed = configuration_sampling.get("maximum_changed_objects")
    fraction = configuration_sampling.get("movable_fraction")
    if not isinstance(fraction, (int, float)) or not 0.0 < fraction <= 1.0:
        errors.append("configuration_sampling.movable_fraction must lie in (0,1]")
    if not isinstance(minimum_changed, int) or not isinstance(maximum_changed, int) or not 1 <= minimum_changed <= maximum_changed:
        errors.append("configuration changed-object bounds must satisfy 1 <= minimum <= maximum")
    for key in (
        "keyframe_count", "geometry_width", "geometry_height",
    ):
        value = preflight.get(key)
        if not isinstance(value, int) or value < 1:
            errors.append(f"Trajectory overlap setting {key} must be a positive integer")
    sampling_diagnostics = config.get("sampling_diagnostics", {})
    warning_minimum = sampling_diagnostics.get(
        "minimum_samples_for_distribution_warnings"
    )
    if not isinstance(warning_minimum, int) or warning_minimum < 1:
        errors.append(
            "sampling_diagnostics minimum warning sample count must be positive"
        )
    collapse_thresholds = sampling_diagnostics.get(
        "collapse_thresholds", {}
    )
    required_collapse_thresholds = {
        "dominant_start_region_fraction_max",
        "direct_path_fraction_max",
        "parallel_episode_fraction_max",
        "compact_start_episode_fraction_max",
        "complete_triangle_keyframe_fraction_max",
        "zero_visible_intervention_fraction_max",
        "dominant_intervention_room_fraction_max",
        "dominant_intervention_category_fraction_max",
        "single_changed_object_configuration_fraction_max",
        "temporal_union_connected_fraction_min",
    }
    if set(collapse_thresholds) != required_collapse_thresholds or any(
        not isinstance(value, (int, float))
        or not 0.0 <= float(value) <= 1.0
        for value in collapse_thresholds.values()
    ):
        errors.append(
            "sampling_diagnostics collapse_thresholds must define all "
            "Dataset-v1.1 warning fractions in [0,1]"
        )
    parallel_similarity = sampling_diagnostics.get(
        "parallel_path_direction_similarity_min"
    )
    if (
        not isinstance(parallel_similarity, (int, float))
        or not -1.0 <= float(parallel_similarity) <= 1.0
    ):
        errors.append(
            "sampling_diagnostics parallel similarity must lie in [-1,1]"
        )
    compact_distance = sampling_diagnostics.get(
        "compact_start_max_pairwise_distance_m"
    )
    if (
        not isinstance(compact_distance, (int, float))
        or float(compact_distance) <= 0.0
    ):
        errors.append(
            "sampling_diagnostics compact start distance must be positive"
        )
    if errors:
        raise ConfigurationError("Invalid dataset configuration:\n- " + "\n- ".join(errors))
