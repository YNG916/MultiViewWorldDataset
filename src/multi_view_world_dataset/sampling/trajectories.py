from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from multi_view_world_dataset.cameras.transforms import compose_transforms, pose_from_xy_yaw
from multi_view_world_dataset.errors import SampleRejected
from multi_view_world_dataset.schema.records import Trajectory

FloatArray = NDArray[np.float64]


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def generate_smooth_trajectory(
    robot_id: str,
    start_xy_yaw: tuple[float, float, float],
    end_xy_yaw: tuple[float, float, float],
    floor_z: float,
    camera_to_robot_base: ArrayLike,
    *,
    frames: int,
    fps: float,
) -> Trajectory:
    if frames < 2:
        raise ValueError("Trajectory requires at least two physical frames")
    alpha = np.linspace(0.0, 1.0, frames)
    smooth = 3 * alpha**2 - 2 * alpha**3
    x0, y0, yaw0 = start_xy_yaw
    x1, y1, yaw1 = end_xy_yaw
    yaw_delta = _wrap_angle(yaw1 - yaw0)
    bases = np.stack(
        [pose_from_xy_yaw(x0 + s * (x1 - x0), y0 + s * (y1 - y0), floor_z, yaw0 + s * yaw_delta) for s in smooth]
    )
    camera_relative = np.asarray(camera_to_robot_base, dtype=np.float64)
    cameras = np.stack([compose_transforms(base, camera_relative) for base in bases])
    return Trajectory(robot_id=robot_id, fps=fps, base_to_world=bases, camera_to_world=cameras)


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
    velocities = np.diff(positions, axis=0) * trajectory.fps
    accelerations = np.diff(velocities, axis=0) * trajectory.fps
    yaws = np.unwrap(np.arctan2(trajectory.base_to_world[:, 1, 0], trajectory.base_to_world[:, 0, 0]))
    angular_speeds = np.diff(yaws) * trajectory.fps
    return {
        "path_length_m": float(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1).sum()),
        "maximum_linear_speed_mps": float(np.linalg.norm(velocities[:, :2], axis=1).max(initial=0.0)),
        "maximum_angular_speed_radps": float(np.abs(angular_speeds).max(initial=0.0)),
        "maximum_acceleration_mps2": float(np.linalg.norm(accelerations[:, :2], axis=1).max(initial=0.0)),
    }


def sample_smooth_trajectory_set(
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
    is_path_traversable: Callable[[np.ndarray], bool],
    maximum_attempts: int,
) -> tuple[Trajectory, ...]:
    robot_ids = tuple(sorted(starts))
    if robot_ids != tuple(sorted(camera_to_robot_bases)):
        raise ValueError("Robot starts and camera mounts must have identical IDs")
    candidates = np.asarray(traversable_xy, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] != 2:
        raise ValueError("traversable_xy must have shape [K,2]")
    minimum_length, maximum_length = path_length_range_m
    alpha = np.linspace(0.0, 1.0, frames)
    smooth = 3 * alpha**2 - 2 * alpha**3
    maximum_step = float(np.diff(smooth).max(initial=0.0))
    maximum_second_difference = float(np.abs(np.diff(smooth, n=2)).max(initial=0.0))
    speed_per_meter = maximum_step * fps
    acceleration_per_meter = maximum_second_difference * fps**2
    speed_limited_length = (
        float("inf") if speed_per_meter == 0 else maximum_linear_speed_mps / speed_per_meter
    )
    acceleration_limited_length = (
        float("inf")
        if acceleration_per_meter == 0
        else maximum_acceleration_mps2 / acceleration_per_meter
    )
    maximum_kinematic_length = min(
        maximum_length,
        speed_limited_length,
        acceleration_limited_length,
    )
    trajectory_pools: dict[str, list[Trajectory]] = {}
    for robot_id in robot_ids:
        start = np.asarray(starts[robot_id], dtype=np.float64)
        start_xy = start[:2, 3]
        start_yaw = float(np.arctan2(start[1, 0], start[0, 0]))
        distances = np.linalg.norm(candidates - start_xy, axis=1)
        eligible = np.flatnonzero(
            (distances >= minimum_length)
            & (distances <= maximum_kinematic_length + 1e-12)
        )
        if not len(eligible):
            raise SampleRejected(
                "trajectory_constraints_infeasible",
                {
                    "robot_id": robot_id,
                    "candidate_count": len(candidates),
                    "frames": frames,
                    "path_length_range_m": list(path_length_range_m),
                    "maximum_kinematic_length_m": maximum_kinematic_length,
                },
            )
        pool: list[Trajectory] = []
        for index in rng.permutation(eligible):
            endpoint = candidates[int(index)]
            candidate = generate_smooth_trajectory(
                robot_id,
                (float(start_xy[0]), float(start_xy[1]), start_yaw),
                (float(endpoint[0]), float(endpoint[1]), start_yaw),
                floor_z,
                camera_to_robot_bases[robot_id],
                frames=frames,
                fps=fps,
            )
            metrics = trajectory_kinematic_metrics(candidate)
            if (
                is_path_traversable(candidate.base_to_world[:, :2, 3])
                and metrics["maximum_linear_speed_mps"] <= maximum_linear_speed_mps
                and metrics["maximum_angular_speed_radps"] <= maximum_angular_speed_radps
                and metrics["maximum_acceleration_mps2"] <= maximum_acceleration_mps2
            ):
                pool.append(candidate)
                if len(pool) >= maximum_attempts:
                    break
        if not pool:
            raise SampleRejected(
                "trajectory_no_traversable_path",
                {
                    "robot_id": robot_id,
                    "eligible_endpoint_count": len(eligible),
                    "frames": frames,
                },
            )
        trajectory_pools[robot_id] = pool
    combination_attempts = maximum_attempts * 10
    for _ in range(combination_attempts):
        trajectories = tuple(
            trajectory_pools[robot_id][
                int(rng.integers(len(trajectory_pools[robot_id])))
            ]
            for robot_id in robot_ids
        )
        positions = np.stack(
            [trajectory.base_to_world[:, :2, 3] for trajectory in trajectories]
        )
        if all(
            np.all(
                np.linalg.norm(positions[left] - positions[right], axis=1)
                >= minimum_pairwise_distance_m
            )
            for left in range(len(trajectories))
            for right in range(left + 1, len(trajectories))
        ):
            return trajectories
    raise SampleRejected(
        "trajectory_sampling_failed",
        {
            "candidate_count": len(candidates),
            "frames": frames,
            "path_length_range_m": list(path_length_range_m),
            "maximum_attempts": maximum_attempts,
            "combination_attempts": combination_attempts,
            "valid_path_counts": {
                robot_id: len(pool) for robot_id, pool in trajectory_pools.items()
            },
        },
    )

