from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from multi_view_world_dataset.cameras.transforms import compose_transforms, pose_from_xy_yaw
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

