from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import product

import numpy as np
from numpy.typing import ArrayLike, NDArray

from multi_view_world_dataset.cameras.transforms import compose_transforms, pose_from_xy_yaw
from multi_view_world_dataset.errors import SampleRejected
from multi_view_world_dataset.schema.records import Trajectory

FloatArray = NDArray[np.float64]
PlanSegment = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, float] | None]
PathValidator = Callable[[np.ndarray], bool]

_PATH_FAMILY_SEGMENTS = {
    "direct": 1,
    "one_waypoint": 2,
    "two_waypoint": 3,
}


def _wrap_angles(angles: ArrayLike) -> FloatArray:
    values = np.asarray(angles, dtype=np.float64)
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _polyline_length(points: ArrayLike) -> float:
    values = np.asarray(points, dtype=np.float64)
    if len(values) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(values, axis=0), axis=1).sum())


def densify_polyline(points: ArrayLike, maximum_spacing_m: float) -> FloatArray:
    """Linearly densify a polyline so every validation chord is short."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 2:
        raise ValueError("points must have shape [N,2] with N >= 2")
    if maximum_spacing_m <= 0:
        raise ValueError("maximum_spacing_m must be positive")
    dense = [values[0]]
    for left, right in zip(values[:-1], values[1:], strict=True):
        distance = float(np.linalg.norm(right - left))
        steps = max(1, int(np.ceil(distance / maximum_spacing_m)))
        dense.extend(left + (right - left) * alpha for alpha in np.linspace(0.0, 1.0, steps + 1)[1:])
    return np.asarray(dense, dtype=np.float64)


def collision_safe_planner_polyline(
    points: ArrayLike,
    is_path_traversable: PathValidator,
    *,
    validation_spacing_m: float,
) -> FloatArray | None:
    """Repair unsafe 8-connected diagonal steps with validated orthogonal corners."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 2:
        return None
    result = [values[0]]
    for right in values[1:]:
        left = result[-1]
        direct = densify_polyline(np.stack((left, right)), validation_spacing_m)
        if is_path_traversable(direct):
            result.append(right)
            continue
        repaired = False
        for corner in (
            np.asarray([left[0], right[1]], dtype=np.float64),
            np.asarray([right[0], left[1]], dtype=np.float64),
        ):
            detour = densify_polyline(
                np.stack((left, corner, right)), validation_spacing_m
            )
            if is_path_traversable(detour):
                result.extend((corner, right))
                repaired = True
                break
        if not repaired:
            return None
    repaired_path = np.asarray(result, dtype=np.float64)
    dense = densify_polyline(repaired_path, validation_spacing_m)
    return repaired_path if is_path_traversable(dense) else None


def shortcut_polyline(
    points: ArrayLike,
    is_path_traversable: PathValidator,
    *,
    validation_spacing_m: float,
) -> FloatArray:
    """Greedily remove planner vertices only when the full LOS chord is safe."""
    values = np.asarray(points, dtype=np.float64)
    if len(values) < 3:
        return values.copy()
    result = [values[0]]
    source = 0
    while source < len(values) - 1:
        target = len(values) - 1
        while target > source + 1:
            line = densify_polyline(values[[source, target]], validation_spacing_m)
            if is_path_traversable(line):
                break
            target -= 1
        result.append(values[target])
        source = target
    return np.asarray(result, dtype=np.float64)


def _catmull_rom_curve(
    points: FloatArray,
    *,
    strength: float,
    validation_spacing_m: float,
) -> FloatArray:
    """Evaluate a C1 Catmull-Rom curve, blending toward its safe chords."""
    if len(points) == 2:
        return densify_polyline(points, validation_spacing_m)
    output: list[np.ndarray] = []
    for index in range(len(points) - 1):
        p0 = points[max(0, index - 1)]
        p1 = points[index]
        p2 = points[index + 1]
        p3 = points[min(len(points) - 1, index + 2)]
        chord = float(np.linalg.norm(p2 - p1))
        sample_count = max(3, int(np.ceil(chord / validation_spacing_m)) + 1)
        parameters = np.linspace(0.0, 1.0, sample_count, endpoint=index == len(points) - 2)
        for t in parameters:
            t2 = t * t
            t3 = t2 * t
            curved = 0.5 * (
                2.0 * p1
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            linear = p1 + (p2 - p1) * t
            output.append(linear + strength * (curved - linear))
    curve = np.asarray(output, dtype=np.float64)
    curve[0] = points[0]
    curve[-1] = points[-1]
    return densify_polyline(curve, validation_spacing_m)


def smooth_collision_safe_path(
    points: ArrayLike,
    is_path_traversable: PathValidator,
    *,
    smoothing_strengths: Sequence[float],
    validation_spacing_m: float,
) -> tuple[FloatArray, float] | None:
    """Smooth a shortcut path and back off until the dense curve is safe."""
    values = np.asarray(points, dtype=np.float64)
    if len(values) == 2:
        dense = densify_polyline(values, validation_spacing_m)
        return (dense, 0.0) if is_path_traversable(dense) else None
    for strength in smoothing_strengths:
        if not 0.0 < float(strength) <= 1.0:
            raise ValueError("smoothing strengths must lie in (0,1]")
        curve = _catmull_rom_curve(
            values,
            strength=float(strength),
            validation_spacing_m=validation_spacing_m,
        )
        if is_path_traversable(curve):
            return curve, float(strength)
    return None


def resample_path_by_arc_length(points: ArrayLike, frames: int) -> FloatArray:
    """Resample a spatial curve at uniform arc-length coordinates."""
    values = np.asarray(points, dtype=np.float64)
    if frames < 2 or len(values) < 2:
        raise ValueError("arc-length resampling requires frames >= 2 and at least two points")
    segment_lengths = np.linalg.norm(np.diff(values, axis=0), axis=1)
    keep = np.concatenate(([True], segment_lengths > 1.0e-9))
    values = values[keep]
    if len(values) < 2:
        raise ValueError("path has zero arc length")
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(values, axis=0), axis=1))))
    targets = np.linspace(0.0, cumulative[-1], frames)
    return np.column_stack(
        [np.interp(targets, cumulative, values[:, dimension]) for dimension in range(2)]
    )


def _tangent_yaws(points: FloatArray) -> FloatArray:
    tangent = np.gradient(points, axis=0)
    norms = np.linalg.norm(tangent, axis=1)
    for index in np.flatnonzero(norms <= 1.0e-10):
        if index:
            tangent[index] = tangent[index - 1]
        elif len(tangent) > 1:
            tangent[index] = tangent[index + 1]
    return np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))


def trajectory_from_spatial_path(
    robot_id: str,
    path_xy: ArrayLike,
    floor_z: float,
    camera_to_robot_base: ArrayLike,
    *,
    frames: int,
    fps: float,
    path_family: str = "unspecified",
    control_waypoints_xy: ArrayLike | None = None,
    planner_path_xy: ArrayLike | None = None,
    smoothed_path_xy: ArrayLike | None = None,
    metadata: Mapping[str, object] | None = None,
) -> Trajectory:
    sampled_xy = resample_path_by_arc_length(path_xy, frames)
    yaws = _tangent_yaws(sampled_xy)
    bases = np.stack(
        [pose_from_xy_yaw(float(x), float(y), floor_z, float(yaw)) for (x, y), yaw in zip(sampled_xy, yaws, strict=True)]
    )
    camera_relative = np.asarray(camera_to_robot_base, dtype=np.float64)
    cameras = np.stack([compose_transforms(base, camera_relative) for base in bases])
    empty = np.empty((0, 2), dtype=np.float64)
    return Trajectory(
        robot_id=robot_id,
        fps=fps,
        base_to_world=bases,
        camera_to_world=cameras,
        path_family=path_family,
        control_waypoints_xy=empty if control_waypoints_xy is None else control_waypoints_xy,
        planner_path_xy=empty if planner_path_xy is None else planner_path_xy,
        smoothed_path_xy=np.asarray(path_xy if smoothed_path_xy is None else smoothed_path_xy),
        metadata=dict(metadata or {}),
    )


def trajectories_equal(
    before: Trajectory,
    after: Trajectory,
    *,
    position_atol_m: float = 1e-5,
    matrix_atol: float = 1e-6,
) -> bool:
    if before.robot_id != after.robot_id or before.base_to_world.shape != after.base_to_world.shape:
        return False
    base_translation_equal = np.allclose(
        before.base_to_world[:, :3, 3], after.base_to_world[:, :3, 3], atol=position_atol_m, rtol=0
    )
    return bool(
        base_translation_equal
        and np.allclose(before.base_to_world[:, :3, :3], after.base_to_world[:, :3, :3], atol=matrix_atol, rtol=0)
        and np.allclose(before.camera_to_world, after.camera_to_world, atol=matrix_atol, rtol=0)
    )


def trajectory_kinematic_metrics(trajectory: Trajectory) -> dict[str, float]:
    positions = trajectory.base_to_world[:, :3, 3]
    planar_steps = np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)
    sampled_length = float(planar_steps.sum())
    arc_length = float(trajectory.metadata.get("smoothed_arc_length_m", sampled_length))
    displacement = float(np.linalg.norm(positions[-1, :2] - positions[0, :2]))
    velocities = np.diff(positions, axis=0) * trajectory.fps
    accelerations = np.diff(velocities, axis=0) * trajectory.fps
    yaws = np.unwrap(np.arctan2(trajectory.base_to_world[:, 1, 0], trajectory.base_to_world[:, 0, 0]))
    yaw_steps = np.diff(yaws)
    curvature = np.abs(yaw_steps) / np.maximum(planar_steps, 1.0e-9)
    return {
        "arc_path_length_m": arc_length,
        "path_length_m": arc_length,
        "sampled_chord_path_length_m": sampled_length,
        "start_end_displacement_m": displacement,
        "tortuosity": arc_length / max(displacement, 1.0e-9),
        "net_yaw_change_rad": float(yaws[-1] - yaws[0]),
        "cumulative_absolute_yaw_change_rad": float(np.abs(yaw_steps).sum()),
        "maximum_curvature_radpm": float(curvature.max(initial=0.0)),
        "maximum_linear_speed_mps": float(np.linalg.norm(velocities[:, :2], axis=1).max(initial=0.0)),
        "maximum_angular_speed_radps": float(np.abs(yaw_steps * trajectory.fps).max(initial=0.0)),
        "maximum_acceleration_mps2": float(np.linalg.norm(accelerations[:, :2], axis=1).max(initial=0.0)),
    }


def _normalised_family_distribution(weights: Mapping[str, float]) -> tuple[tuple[str, ...], FloatArray]:
    unknown = set(weights) - set(_PATH_FAMILY_SEGMENTS)
    missing = set(_PATH_FAMILY_SEGMENTS) - set(weights)
    if unknown or missing:
        raise ValueError(f"path family weights mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")
    families = tuple(_PATH_FAMILY_SEGMENTS)
    probabilities = np.asarray([float(weights[name]) for name in families], dtype=np.float64)
    if np.any(probabilities < 0.0) or not np.isfinite(probabilities).all() or probabilities.sum() <= 0.0:
        raise ValueError("path family weights must be finite, non-negative, and have positive sum")
    return families, probabilities / probabilities.sum()


def _sample_route_controls(
    start_xy: FloatArray,
    start_yaw: float,
    candidates: FloatArray,
    family: str,
    minimum_length_m: float,
    maximum_length_m: float,
    initial_heading_tolerance_rad: float,
    rng: np.random.Generator,
) -> FloatArray | None:
    segment_count = _PATH_FAMILY_SEGMENTS[family]
    controls = [start_xy]
    current = start_xy
    for segment_index in range(segment_count):
        distances = np.linalg.norm(candidates - current, axis=1)
        minimum_step = 0.75 * minimum_length_m / segment_count
        maximum_step = maximum_length_m / segment_count
        eligible = (distances >= max(0.10, minimum_step)) & (distances <= maximum_step)
        if segment_index == 0:
            bearings = np.arctan2(candidates[:, 1] - current[1], candidates[:, 0] - current[0])
            eligible &= np.abs(_wrap_angles(bearings - start_yaw)) <= initial_heading_tolerance_rad
        indices = np.flatnonzero(eligible)
        if not len(indices):
            return None
        current = candidates[int(indices[int(rng.integers(len(indices)))])]
        controls.append(current)
    return np.asarray(controls, dtype=np.float64)


def _plan_route(
    controls: FloatArray,
    plan_segment: PlanSegment,
    is_path_traversable: PathValidator,
    *,
    line_validation_spacing_m: float,
) -> tuple[FloatArray, float] | None:
    shortcut_segments: list[FloatArray] = []
    geodesic_length = 0.0
    for start, goal in zip(controls[:-1], controls[1:], strict=True):
        result = plan_segment(start.copy(), goal.copy())
        if result is None:
            return None
        planner_points, segment_geodesic = result
        planner_points = np.asarray(planner_points, dtype=np.float64)
        if planner_points.ndim != 2 or planner_points.shape[1] != 2 or not len(planner_points):
            return None
        if np.linalg.norm(planner_points[0] - start) > 1.0e-7:
            planner_points = np.vstack((start, planner_points))
        if np.linalg.norm(planner_points[-1] - goal) > 1.0e-7:
            planner_points = np.vstack((planner_points, goal))
        safe_planner_points = collision_safe_planner_polyline(
            planner_points,
            is_path_traversable,
            validation_spacing_m=line_validation_spacing_m,
        )
        if safe_planner_points is None:
            return None
        shortcut = shortcut_polyline(
            safe_planner_points,
            is_path_traversable,
            validation_spacing_m=line_validation_spacing_m,
        )
        shortcut_segments.append(shortcut if not shortcut_segments else shortcut[1:])
        geodesic_length += float(segment_geodesic)
    return np.concatenate(shortcut_segments, axis=0), geodesic_length


def _sample_robot_pool(
    robot_id: str,
    start: FloatArray,
    camera_mount: FloatArray,
    candidates: FloatArray,
    floor_z: float,
    rng: np.random.Generator,
    *,
    frames: int,
    fps: float,
    path_length_range_m: tuple[float, float],
    maximum_linear_speed_mps: float,
    maximum_angular_speed_radps: float,
    maximum_acceleration_mps2: float,
    plan_segment: PlanSegment,
    is_path_traversable: PathValidator,
    path_family_weights: Mapping[str, float],
    initial_heading_tolerance_rad: float,
    line_validation_spacing_m: float,
    smoothing_validation_spacing_m: float,
    smoothing_strengths: Sequence[float],
    candidate_pool_size: int,
    maximum_attempts: int,
) -> list[Trajectory]:
    families, probabilities = _normalised_family_distribution(path_family_weights)
    start_xy = start[:2, 3]
    start_yaw = float(np.arctan2(start[1, 0], start[0, 0]))
    minimum_length, maximum_length = path_length_range_m
    pool: list[Trajectory] = []
    attempts_by_family = {name: 0 for name in families}
    rejection_counts = {
        "no_control_candidates": 0,
        "planner_or_strict_validation": 0,
        "smoothing_collision": 0,
        "arc_length": 0,
        "initial_heading": 0,
        "final_traversability": 0,
        "linear_speed": 0,
        "angular_speed": 0,
        "acceleration": 0,
    }
    for _ in range(maximum_attempts):
        family = str(rng.choice(families, p=probabilities))
        attempts_by_family[family] += 1
        controls = _sample_route_controls(
            start_xy,
            start_yaw,
            candidates,
            family,
            minimum_length,
            maximum_length,
            initial_heading_tolerance_rad,
            rng,
        )
        if controls is None:
            rejection_counts["no_control_candidates"] += 1
            continue
        planned = _plan_route(
            controls,
            plan_segment,
            is_path_traversable,
            line_validation_spacing_m=line_validation_spacing_m,
        )
        if planned is None:
            rejection_counts["planner_or_strict_validation"] += 1
            continue
        shortcut_path, geodesic_length = planned
        smoothed = smooth_collision_safe_path(
            shortcut_path,
            is_path_traversable,
            smoothing_strengths=smoothing_strengths,
            validation_spacing_m=smoothing_validation_spacing_m,
        )
        if smoothed is None:
            rejection_counts["smoothing_collision"] += 1
            continue
        smooth_path, smoothing_strength = smoothed
        smooth_arc_length = _polyline_length(smooth_path)
        if not minimum_length <= smooth_arc_length <= maximum_length:
            rejection_counts["arc_length"] += 1
            continue
        candidate = trajectory_from_spatial_path(
            robot_id,
            smooth_path,
            floor_z,
            camera_mount,
            frames=frames,
            fps=fps,
            path_family=family,
            control_waypoints_xy=controls,
            planner_path_xy=shortcut_path,
            smoothed_path_xy=smooth_path,
            metadata={
                "planner_geodesic_length_m": geodesic_length,
                "smoothed_arc_length_m": smooth_arc_length,
                "smoothing_strength": smoothing_strength,
            },
        )
        first_yaw = float(np.arctan2(candidate.base_to_world[0, 1, 0], candidate.base_to_world[0, 0, 0]))
        heading_error = abs(float(_wrap_angles(first_yaw - start_yaw)))
        metrics = trajectory_kinematic_metrics(candidate)
        dense_sampled = densify_polyline(
            candidate.base_to_world[:, :2, 3], smoothing_validation_spacing_m
        )
        checks = {
            "initial_heading": heading_error <= initial_heading_tolerance_rad,
            "final_traversability": is_path_traversable(dense_sampled),
            "linear_speed": metrics["maximum_linear_speed_mps"] <= maximum_linear_speed_mps + 1.0e-9,
            "angular_speed": metrics["maximum_angular_speed_radps"] <= maximum_angular_speed_radps + 1.0e-9,
            "acceleration": metrics["maximum_acceleration_mps2"] <= maximum_acceleration_mps2 + 1.0e-9,
        }
        failures = [name for name, passed in checks.items() if not passed]
        if failures:
            for name in failures:
                rejection_counts[name] += 1
            continue
        candidate.metadata["initial_heading_error_rad"] = heading_error
        pool.append(candidate)
        if len(pool) >= candidate_pool_size:
            return pool
    if not pool:
        raise SampleRejected(
            "trajectory_no_geodesic_path",
            {
                "robot_id": robot_id,
                "candidate_count": len(candidates),
                "attempts_by_family": attempts_by_family,
                "rejection_counts": rejection_counts,
                "path_length_range_m": list(path_length_range_m),
            },
        )
    return pool


def sample_geodesic_trajectory_set(
    starts: Mapping[str, np.ndarray],
    camera_to_robot_bases: Mapping[str, np.ndarray],
    traversable_xy: np.ndarray,
    floor_z: float,
    rng: np.random.Generator,
    *,
    frames: int,
    fps: float,
    path_length_range_m: tuple[float, float],
    minimum_pairwise_distance_m: float,
    maximum_linear_speed_mps: float,
    maximum_angular_speed_radps: float,
    maximum_acceleration_mps2: float,
    plan_segment: PlanSegment,
    is_path_traversable: PathValidator,
    path_family_weights: Mapping[str, float],
    minimum_waypoint_trajectories: int,
    initial_heading_tolerance_rad: float,
    line_validation_spacing_m: float,
    smoothing_validation_spacing_m: float,
    smoothing_strengths: Sequence[float],
    candidate_pool_size: int,
    maximum_attempts: int,
    joint_pool_rounds: int,
) -> tuple[Trajectory, ...]:
    """Sample independent robot paths, then jointly enforce temporal separation."""
    robot_ids = tuple(sorted(starts))
    if robot_ids != tuple(sorted(camera_to_robot_bases)):
        raise ValueError("Robot starts and camera mounts must have identical IDs")
    candidates = np.asarray(traversable_xy, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] != 2:
        raise ValueError("traversable_xy must have shape [K,2]")
    if (
        frames < 2
        or candidate_pool_size < 1
        or maximum_attempts < 1
        or joint_pool_rounds < 1
        or minimum_waypoint_trajectories < 0
        or minimum_waypoint_trajectories > len(robot_ids)
    ):
        raise ValueError("invalid trajectory sampling count or waypoint minimum")

    round_diagnostics: list[dict[str, object]] = []
    for round_index in range(joint_pool_rounds):
        try:
            pools = {
                robot_id: _sample_robot_pool(
                    robot_id,
                    np.asarray(starts[robot_id], dtype=np.float64),
                    np.asarray(camera_to_robot_bases[robot_id], dtype=np.float64),
                    candidates,
                    floor_z,
                    rng,
                    frames=frames,
                    fps=fps,
                    path_length_range_m=path_length_range_m,
                    maximum_linear_speed_mps=maximum_linear_speed_mps,
                    maximum_angular_speed_radps=maximum_angular_speed_radps,
                    maximum_acceleration_mps2=maximum_acceleration_mps2,
                    plan_segment=plan_segment,
                    is_path_traversable=is_path_traversable,
                    path_family_weights=path_family_weights,
                    initial_heading_tolerance_rad=initial_heading_tolerance_rad,
                    line_validation_spacing_m=line_validation_spacing_m,
                    smoothing_validation_spacing_m=smoothing_validation_spacing_m,
                    smoothing_strengths=smoothing_strengths,
                    candidate_pool_size=candidate_pool_size,
                    maximum_attempts=maximum_attempts,
                )
                for robot_id in robot_ids
            }
        except SampleRejected as error:
            round_diagnostics.append(
                {
                    "round_index": round_index,
                    "failure_reason": error.reason,
                    "failure_details": error.details,
                }
            )
            continue

        combinations = list(
            product(*(range(len(pools[robot_id])) for robot_id in robot_ids))
        )
        best: tuple[tuple[float, float, float], tuple[Trajectory, ...]] | None = None
        for combination_index in rng.permutation(len(combinations)):
            selection = combinations[int(combination_index)]
            trajectories = tuple(
                pools[robot_id][selection[index]]
                for index, robot_id in enumerate(robot_ids)
            )
            waypoint_count = sum(
                trajectory.path_family != "direct"
                for trajectory in trajectories
            )
            if waypoint_count < minimum_waypoint_trajectories:
                continue
            positions = np.stack(
                [trajectory.base_to_world[:, :2, 3] for trajectory in trajectories]
            )
            pairwise_distances = np.stack(
                [
                    np.linalg.norm(positions[left] - positions[right], axis=1)
                    for left in range(len(trajectories))
                    for right in range(left + 1, len(trajectories))
                ]
            )
            if np.any(pairwise_distances < minimum_pairwise_distance_m):
                continue
            yaws = np.stack(
                [
                    np.arctan2(
                        trajectory.base_to_world[:, 1, 0],
                        trajectory.base_to_world[:, 0, 0],
                    )
                    for trajectory in trajectories
                ]
            )
            maximum_heading_spread = max(
                float(
                    np.max(
                        np.abs(_wrap_angles(yaws[left] - yaws[right]))
                    )
                )
                for left in range(len(trajectories))
                for right in range(left + 1, len(trajectories))
            )
            score = (
                float(np.max(pairwise_distances)),
                maximum_heading_spread,
                float(np.mean(pairwise_distances)),
            )
            if best is None or score < best[0]:
                best = score, trajectories
        if best is not None:
            score, trajectories = best
            for trajectory in trajectories:
                trajectory.metadata["joint_pool_round"] = round_index
                trajectory.metadata["joint_compactness_score"] = list(score)
                trajectory.metadata["joint_waypoint_trajectory_count"] = sum(
                    item.path_family != "direct" for item in trajectories
                )
            return trajectories
        round_diagnostics.append(
            {
                "round_index": round_index,
                "valid_path_counts": {
                    robot_id: len(pool) for robot_id, pool in pools.items()
                },
                "joint_combination_count": len(combinations),
            }
        )

    raise SampleRejected(
        "trajectory_joint_separation_failed",
        {
            "joint_pool_rounds": joint_pool_rounds,
            "round_diagnostics": round_diagnostics,
            "minimum_pairwise_distance_m": minimum_pairwise_distance_m,
            "minimum_waypoint_trajectories": minimum_waypoint_trajectories,
        },
    )
