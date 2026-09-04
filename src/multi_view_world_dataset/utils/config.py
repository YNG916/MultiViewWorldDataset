from __future__ import annotations

from copy import deepcopy
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
    for key in (
        "initial_heading_tolerance_deg",
        "line_validation_spacing_m",
        "smoothing_validation_spacing_m",
        "candidate_pool_size",
        "joint_pool_rounds",
        "sampling_maximum_attempts",
    ):
        value = trajectory.get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            errors.append(f"Trajectory setting {key} must be positive")
    strengths = trajectory.get("smoothing_strengths", ())
    if not strengths or any(not 0.0 < float(value) <= 1.0 for value in strengths):
        errors.append("Trajectory smoothing_strengths must lie in (0,1]")
    preflight = trajectory.get("overlap_preflight", {})
    connected_fraction = preflight.get("connected_fraction_min")
    if not isinstance(connected_fraction, (int, float)) or not 0.0 <= connected_fraction <= 1.0:
        errors.append("Trajectory overlap connected_fraction_min must lie in [0,1]")
    for key in (
        "keyframe_count", "geometry_width", "geometry_height",
        "maximum_consecutive_isolated_keyframes", "depth_sample_stride",
    ):
        value = preflight.get(key)
        if not isinstance(value, int) or value < 1:
            errors.append(f"Trajectory overlap setting {key} must be a positive integer")
    if errors:
        raise ConfigurationError("Invalid dataset configuration:\n- " + "\n- ".join(errors))

