from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import product

import numpy as np
from numpy.typing import ArrayLike, NDArray

from multi_view_world_dataset.cameras.transforms import compose_transforms, pose_from_xy_yaw
from multi_view_world_dataset.errors import SampleRejected
from multi_view_world_dataset.sampling.diversity import (
    formation_degenerate,
    joint_trajectory_metrics,
    regime_trajectory_soft_score,
)
from multi_view_world_dataset.schema.records import Trajectory

FloatArray = NDArray[np.float64]
PlanSegment = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, float] | None]
PathValidator = Callable[[np.ndarray], bool]

_PATH_FAMILY_SEGMENTS = {
    "direct": 1,
    "one_waypoint": 2,
    "two_waypoint": 3,
}


def lane_preserving_guides(
    starts_xy: ArrayLike,
    heading_rad: float,
    distance_m: float,
) -> FloatArray:
    """Translate every start by one scene-view vector without collapsing lanes."""
    starts = np.asarray(starts_xy, dtype=np.float64)
    if starts.ndim != 2 or starts.shape[1] != 2 or not len(starts):
        raise ValueError("starts_xy must have shape [N,2] with N >= 1")
    if not np.isfinite(starts).all() or not np.isfinite(heading_rad):
        raise ValueError("lane guide inputs must be finite")
    if not np.isfinite(distance_m) or distance_m <= 0.0:
        raise ValueError("distance_m must be finite and positive")
    displacement = float(distance_m) * np.asarray(
        [np.cos(float(heading_rad)), np.sin(float(heading_rad))],
        dtype=np.float64,
    )
    return starts + displacement


def _wrap_angles(angles: ArrayLike) -> FloatArray:
    values = np.asarray(angles, dtype=np.float64)
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _angular_density_balanced_weights(
    bearings: FloatArray,
    indices: NDArray[np.int64],
    weights: FloatArray,
    *,
    bin_count: int = 36,
) -> FloatArray:
    """Remove traversable-pixel count as a hidden directional prior."""
    if bin_count < 1:
        raise ValueError("bin_count must be positive")
    selected = _wrap_angles(np.asarray(bearings, dtype=np.float64)[indices])
    bins = np.floor((selected + np.pi) * bin_count / (2.0 * np.pi)).astype(int)
    bins = np.clip(bins, 0, bin_count - 1)
    counts = np.bincount(bins, minlength=bin_count)
    balanced = np.asarray(weights, dtype=np.float64) / counts[bins]
    if not np.isfinite(balanced).all() or np.any(balanced <= 0.0):
        raise ValueError("candidate weights must be finite and positive")
    return balanced


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
        if not 0.0 <= float(strength) <= 1.0:
            raise ValueError("smoothing strengths must lie in [0,1]")
        curve = (
            densify_polyline(values, validation_spacing_m)
            if float(strength) == 0.0
            else _catmull_rom_curve(
                values,
                strength=float(strength),
                validation_spacing_m=validation_spacing_m,
            )
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
    # Symmetric sinusoidal acceleration/deceleration ramps preserve most of the
    # clip at cruise speed without the unrealistic instantaneous start/stop of
    # uniform frame spacing.
    intervals = frames - 1
    phase = (np.arange(intervals, dtype=np.float64) + 0.5) / intervals
    ramp_fraction = 0.12
    weights = np.ones(intervals, dtype=np.float64)
    leading = phase < ramp_fraction
    trailing = phase > 1.0 - ramp_fraction
    weights[leading] = np.sin(0.5 * np.pi * phase[leading] / ramp_fraction)
    weights[trailing] = np.sin(0.5 * np.pi * (1.0 - phase[trailing]) / ramp_fraction)
    progress = np.concatenate(([0.0], np.cumsum(weights)))
    progress /= progress[-1]
    targets = progress * cumulative[-1]
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
        metadata={"velocity_profile": "symmetric_sinusoidal_ramp", **dict(metadata or {})},
    )


def minimum_separation_event(
    trajectories: Sequence[Trajectory],
) -> dict[str, object]:
    """Describe the closest robot pair and frame with JSON-safe values."""
    if len(trajectories) < 2:
        raise ValueError("minimum separation requires at least two trajectories")
    frame_counts = {len(trajectory.base_to_world) for trajectory in trajectories}
    if len(frame_counts) != 1:
        raise ValueError("joint trajectories must have the same frame count")
    positions = np.stack(
        [trajectory.base_to_world[:, :2, 3] for trajectory in trajectories]
    )
    pairs = [
        (left, right)
        for left in range(len(trajectories))
        for right in range(left + 1, len(trajectories))
    ]
    distances = np.stack(
        [np.linalg.norm(positions[left] - positions[right], axis=1) for left, right in pairs]
    )
    pair_index, frame_index = np.unravel_index(int(np.argmin(distances)), distances.shape)
    left, right = pairs[int(pair_index)]
    left_id = trajectories[left].robot_id
    right_id = trajectories[right].robot_id
    return {
        "distance_m": float(distances[pair_index, frame_index]),
        "frame_index": int(frame_index),
        "robot_pair": [left_id, right_id],
        "positions_xy": {
            left_id: positions[left, frame_index].tolist(),
            right_id: positions[right, frame_index].tolist(),
        },
        "start_xy": {
            trajectory.robot_id: positions[index, 0].tolist()
            for index, trajectory in enumerate(trajectories)
        },
        "end_xy": {
            trajectory.robot_id: positions[index, -1].tolist()
            for index, trajectory in enumerate(trajectories)
        },
        "path_families": {
            trajectory.robot_id: trajectory.path_family for trajectory in trajectories
        },
    }


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
    maximum_control_turn_rad: float,
    rng: np.random.Generator,
    *,
    soft_initial_heading: bool = False,
    initial_heading_probability_floor: float = 0.20,
    guide_xy: FloatArray | None = None,
    guide_soft_scale_m: float = 1.5,
    guide_probability_floor: float = 0.10,
    prefer_guide: bool = False,
) -> FloatArray | None:
    segment_count = _PATH_FAMILY_SEGMENTS[family]
    controls = [start_xy]
    current = start_xy
    previous_bearing = start_yaw
    for segment_index in range(segment_count):
        distances = np.linalg.norm(candidates - current, axis=1)
        minimum_step = 0.75 * minimum_length_m / segment_count
        maximum_step = maximum_length_m / segment_count
        eligible = (distances >= max(0.10, minimum_step)) & (distances <= maximum_step)
        bearings = np.arctan2(
            candidates[:, 1] - current[1], candidates[:, 0] - current[0]
        )
        if segment_index == 0 and not soft_initial_heading:
            # A far waypoint bearing is only a proxy for the native geodesic
            # initial tangent. Prefer it when topology permits, but do not make
            # a curved corridor impossible before planning. The actual smoothed
            # first tangent is checked against the same hard tolerance below.
            heading_eligible = (
                np.abs(_wrap_angles(bearings - start_yaw))
                <= initial_heading_tolerance_rad
            )
            if np.any(eligible & heading_eligible):
                eligible &= heading_eligible
        elif segment_index > 0:
            # Control points are a spatial prior, not permission for an
            # instantaneous U-turn. Consecutive route bearings constrain the
            # subsequently smoothed curve to physically trackable turns.
            eligible &= np.abs(_wrap_angles(bearings - previous_bearing)) <= maximum_control_turn_rad
        indices = np.flatnonzero(eligible)
        if not len(indices):
            return None
        if prefer_guide and segment_index == 0 and guide_xy is not None:
            guide_indices = indices
            if family == "direct":
                guide_indices = indices[distances[indices] >= minimum_length_m]
            if not len(guide_indices):
                return None
            guide = np.asarray(guide_xy, dtype=np.float64)
            if guide.shape != (2,) or not np.isfinite(guide).all():
                raise ValueError("guide_xy must be a finite XY point")
            selected_index = int(guide_indices[np.argmin(
                np.linalg.norm(candidates[guide_indices] - guide, axis=1)
            )])
            previous_bearing = float(bearings[selected_index])
            current = candidates[selected_index]
            controls.append(current)
            continue
        selection_weights = None
        if segment_index == 0 and soft_initial_heading:
            if not 0.0 < initial_heading_probability_floor <= 1.0:
                raise ValueError("initial heading probability floor must lie in (0,1]")
            heading_errors = np.abs(_wrap_angles(bearings[indices] - start_yaw))
            scale = max(float(initial_heading_tolerance_rad), 1.0e-6)
            selection_weights = initial_heading_probability_floor + (
                1.0 - initial_heading_probability_floor
            ) * np.exp(-0.5 * (heading_errors / scale) ** 2)
        if guide_xy is not None:
            if guide_soft_scale_m <= 0.0:
                raise ValueError("guide soft scale must be positive")
            if not 0.0 < guide_probability_floor <= 1.0:
                raise ValueError("guide probability floor must lie in (0,1]")
            guide = np.asarray(guide_xy, dtype=np.float64)
            if guide.shape != (2,) or not np.isfinite(guide).all():
                raise ValueError("guide_xy must be a finite XY point")
            guide_distances = np.linalg.norm(candidates[indices] - guide, axis=1)
            guide_weights = guide_probability_floor + (
                1.0 - guide_probability_floor
            ) * np.exp(-0.5 * (guide_distances / guide_soft_scale_m) ** 2)
            selection_weights = (
                guide_weights if selection_weights is None
                else selection_weights * guide_weights
            )
        if selection_weights is not None:
            selection_weights = _angular_density_balanced_weights(
                bearings, indices, selection_weights
            )
        if selection_weights is not None:
            selected_index = int(rng.choice(
                indices, p=selection_weights / selection_weights.sum()
            ))
        else:
            selected_index = int(indices[int(rng.integers(len(indices)))])
        previous_bearing = float(bearings[selected_index])
        current = candidates[selected_index]
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
    derive_initial_heading_from_tangent: bool,
    initial_heading_probability_floor: float,
    maximum_control_turn_rad: float,
    line_validation_spacing_m: float,
    smoothing_validation_spacing_m: float,
    smoothing_strengths: Sequence[float],
    candidate_pool_size: int,
    maximum_attempts: int,
    guide_xy: FloatArray | None = None,
    guide_soft_scale_m: float = 1.5,
    guide_probability_floor: float = 0.10,
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
    for attempt_index in range(maximum_attempts):
        prefer_guide = bool(
            attempt_index == 0
            and guide_xy is not None
            and float(path_family_weights.get("direct", 0.0)) > 0.0
        )
        family = "direct" if prefer_guide else str(rng.choice(families, p=probabilities))
        attempts_by_family[family] += 1
        controls = _sample_route_controls(
            start_xy,
            start_yaw,
            candidates,
            family,
            minimum_length,
            maximum_length,
            initial_heading_tolerance_rad,
            maximum_control_turn_rad,
            rng,
            soft_initial_heading=derive_initial_heading_from_tangent,
            initial_heading_probability_floor=initial_heading_probability_floor,
            guide_xy=guide_xy,
            guide_soft_scale_m=guide_soft_scale_m,
            guide_probability_floor=guide_probability_floor,
            prefer_guide=prefer_guide,
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
                "validated_guide_seed": prefer_guide,
            },
        )
        first_yaw = float(np.arctan2(candidate.base_to_world[0, 1, 0], candidate.base_to_world[0, 0, 0]))
        prior_heading_error = abs(float(_wrap_angles(first_yaw - start_yaw)))
        heading_error = prior_heading_error
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
        candidate.metadata["initial_heading_policy"] = (
            "trajectory_tangent" if derive_initial_heading_from_tangent else "fixed_prior"
        )
        candidate.metadata["initial_heading_prior_error_rad"] = prior_heading_error
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

def sample_geodesic_robot_trajectory_pool(
    robot_id: str,
    start: FloatArray,
    camera_mount: FloatArray,
    candidates: FloatArray,
    floor_z: float,
    rng: np.random.Generator,
    **kwargs: object,
) -> tuple[Trajectory, ...]:
    """Expose the validated one-robot pool for bounded joint rescue sampling.

    This is intentionally the same planner / smoothing / kinematic path used
    by :func:`sample_geodesic_trajectory_set`; it does not introduce a second
    trajectory implementation. Joint separation and view connectivity remain
    the caller's responsibility because they depend on the fixed peer paths.
    """
    return tuple(
        _sample_robot_pool(
            robot_id,
            np.asarray(start, dtype=np.float64),
            np.asarray(camera_mount, dtype=np.float64),
            np.asarray(candidates, dtype=np.float64),
            float(floor_z),
            rng,
            **kwargs,
        )
    )


def sample_geodesic_trajectory_set(
    starts: Mapping[str, np.ndarray],
    camera_to_robot_bases: Mapping[str, np.ndarray],
    traversable_xy: np.ndarray | Mapping[str, np.ndarray],
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
    maximum_control_turn_rad: float = np.deg2rad(55.0),
    observation_regime: str = "partial_chain",
    formation_degeneracy_limits: Mapping[str, float] | None = None,
    regime_coverage_saturation_m2: Mapping[str, float] | None = None,
    regime_initial_heading_prior_weights: Mapping[str, float] | None = None,
    regime_view_connectivity_weights: Mapping[str, float] | None = None,
    camera_hfov_deg: float = 70.0,
    derive_initial_heading_from_tangent: bool = False,
    initial_heading_probability_floor: float = 0.20,
    trajectory_guides_xy: Mapping[str, np.ndarray] | None = None,
    guide_soft_scale_m: float = 1.5,
    guide_probability_floor: float = 0.10,
    maximum_joint_valid_candidates: int | None = None,
) -> tuple[Trajectory, ...]:
    """Sample independent robot paths, then jointly enforce temporal separation."""
    robot_ids = tuple(sorted(starts))
    if robot_ids != tuple(sorted(camera_to_robot_bases)):
        raise ValueError("Robot starts and camera mounts must have identical IDs")
    if isinstance(traversable_xy, Mapping):
        if tuple(sorted(traversable_xy)) != robot_ids:
            raise ValueError("Per-robot traversability must have identical robot IDs")
        candidates_by_robot = {
            robot_id: np.asarray(traversable_xy[robot_id], dtype=np.float64)
            for robot_id in robot_ids
        }
    else:
        shared_candidates = np.asarray(traversable_xy, dtype=np.float64)
        candidates_by_robot = {
            robot_id: shared_candidates for robot_id in robot_ids
        }
    if trajectory_guides_xy is not None:
        unknown_guides = set(trajectory_guides_xy) - set(robot_ids)
        if unknown_guides:
            raise ValueError(
                f"trajectory guides contain unknown robots: {sorted(unknown_guides)}"
            )
        guides_by_robot = {
            robot_id: np.asarray(trajectory_guides_xy[robot_id], dtype=np.float64)
            for robot_id in robot_ids if robot_id in trajectory_guides_xy
        }
    else:
        guides_by_robot = {}
    if any(
        candidates.ndim != 2 or candidates.shape[1] != 2 or not len(candidates)
        for candidates in candidates_by_robot.values()
    ):
        raise ValueError("traversable_xy must provide non-empty [K,2] arrays")
    initial_positions = np.stack([
        np.asarray(starts[robot_id], dtype=np.float64)[:2, 3]
        for robot_id in robot_ids
    ])
    initial_pairwise = np.asarray([
        np.linalg.norm(initial_positions[left] - initial_positions[right])
        for left in range(len(robot_ids))
        for right in range(left + 1, len(robot_ids))
    ])
    initial_minimum_distance = float(initial_pairwise.min(initial=np.inf))
    if initial_minimum_distance < minimum_pairwise_distance_m:
        raise SampleRejected(
            "trajectory_start_separation_failed",
            {
                "minimum_start_distance_m": initial_minimum_distance,
                "required_minimum_distance_m": minimum_pairwise_distance_m,
            },
        )
    if (
        frames < 2
        or candidate_pool_size < 1
        or maximum_attempts < 1
        or joint_pool_rounds < 1
        or minimum_waypoint_trajectories < 0
        or minimum_waypoint_trajectories > len(robot_ids)
        or not 0.0 < maximum_control_turn_rad <= np.pi
        or (
            maximum_joint_valid_candidates is not None
            and maximum_joint_valid_candidates < 1
        )
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
                    candidates_by_robot[robot_id],
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
                    maximum_control_turn_rad=maximum_control_turn_rad,
                    line_validation_spacing_m=line_validation_spacing_m,
                    smoothing_validation_spacing_m=smoothing_validation_spacing_m,
                    derive_initial_heading_from_tangent=derive_initial_heading_from_tangent,
                    initial_heading_probability_floor=initial_heading_probability_floor,
                    smoothing_strengths=smoothing_strengths,
                    candidate_pool_size=candidate_pool_size,
                    maximum_attempts=maximum_attempts,
                    guide_xy=guides_by_robot.get(robot_id),
                    guide_soft_scale_m=guide_soft_scale_m,
                    guide_probability_floor=guide_probability_floor,
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
        valid: list[tuple[tuple[Trajectory, ...], dict[str, object]]] = []
        waypoint_rejections = 0
        separation_rejections = 0
        evaluated_combination_count = 0
        maximum_candidate_minimum_distance = 0.0
        maximum_candidate_separation_event: dict[str, object] | None = None
        for combination_index in rng.permutation(len(combinations)):
            evaluated_combination_count += 1
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
                waypoint_rejections += 1
                continue
            separation_event = minimum_separation_event(trajectories)
            candidate_minimum_distance = float(separation_event["distance_m"])
            if (
                maximum_candidate_separation_event is None
                or candidate_minimum_distance > maximum_candidate_minimum_distance
            ):
                maximum_candidate_minimum_distance = candidate_minimum_distance
                maximum_candidate_separation_event = {
                    **separation_event,
                    "pool_selection": {
                        robot_id: int(selection[index])
                        for index, robot_id in enumerate(robot_ids)
                    },
                }
            if candidate_minimum_distance < minimum_pairwise_distance_m:
                separation_rejections += 1
                continue
            metrics = joint_trajectory_metrics(
                trajectories, camera_hfov_deg=camera_hfov_deg
            )
            metrics["formation_degenerate"] = bool(
                formation_degeneracy_limits
                and formation_degenerate(metrics, formation_degeneracy_limits)
            )
            valid.append((trajectories, metrics))
            if (
                maximum_joint_valid_candidates is not None
                and len(valid) >= maximum_joint_valid_candidates
            ):
                break
        if valid:
            # Regime-aware stochastic soft selection; never minimize compactness
            # or heading spread as a hidden objective.
            saturation_by_regime = regime_coverage_saturation_m2 or {
                observation_regime: max(
                    float(item[1]["spatial_coverage_bbox_area_m2"]) for item in valid
                )
            }
            utilities = []
            for _, metrics in valid:
                utilities.append(regime_trajectory_soft_score(
                    metrics, observation_regime, saturation_by_regime,
                    jitter=float(rng.uniform(0.0, 0.10)),
                    heading_prior_weights=regime_initial_heading_prior_weights,
                    view_connectivity_weights=regime_view_connectivity_weights,
                ))
            trajectories, joint_metrics = valid[int(np.argmax(utilities))]
            for trajectory in trajectories:
                trajectory.metadata["joint_pool_round"] = round_index
                trajectory.metadata["observation_regime"] = observation_regime
                trajectory.metadata["joint_diversity_metrics"] = dict(joint_metrics)
                trajectory.metadata["joint_waypoint_trajectory_count"] = sum(
                    item.path_family != "direct" for item in trajectories
                )
                trajectory.metadata["joint_combination_evaluated_count"] = (
                    evaluated_combination_count
                )
                trajectory.metadata["joint_valid_candidate_count"] = len(valid)
                trajectory.metadata["joint_valid_candidate_limit_reached"] = bool(
                    maximum_joint_valid_candidates is not None
                    and len(valid) >= maximum_joint_valid_candidates
                )
            return trajectories
        round_diagnostics.append(
            {
                "round_index": round_index,
                "valid_path_counts": {
                    robot_id: len(pool) for robot_id, pool in pools.items()
                },
                "joint_combination_count": len(combinations),
                "waypoint_rejection_count": waypoint_rejections,
                "separation_rejection_count": separation_rejections,
                "maximum_candidate_minimum_distance_m": maximum_candidate_minimum_distance,
                "best_separation_event": maximum_candidate_separation_event,
            }
        )

    raise SampleRejected(
        "trajectory_joint_separation_failed",
        {
            "joint_pool_rounds": joint_pool_rounds,
            "round_diagnostics": round_diagnostics,
            "minimum_pairwise_distance_m": minimum_pairwise_distance_m,
            "minimum_waypoint_trajectories": minimum_waypoint_trajectories,
            "minimum_start_distance_m": initial_minimum_distance,
        },
    )
