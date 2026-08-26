from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from multi_view_world_dataset.errors import GeometryError

FloatArray = NDArray[np.float64]

# A point in OpenCV camera coordinates maps into USD camera coordinates as (x, -y, -z).
T_USD_CAMERA_OPENCV_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])


def validate_transform(transform: ArrayLike, *, atol: float = 1e-7) -> FloatArray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise GeometryError("Transform must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=atol):
        raise GeometryError("Transform bottom row must be [0,0,0,1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol):
        raise GeometryError("Transform rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=atol):
        raise GeometryError("Transform rotation is not right-handed")
    return matrix


def invert_transform(transform: ArrayLike) -> FloatArray:
    matrix = validate_transform(transform)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -(matrix[:3, :3].T @ matrix[:3, 3])
    return inverse


def compose_transforms(*transforms: ArrayLike) -> FloatArray:
    if not transforms:
        return np.eye(4, dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    for transform in transforms:
        result = result @ validate_transform(transform)
    return validate_transform(result)


def transform_points(transform: ArrayLike, points: ArrayLike) -> FloatArray:
    matrix = validate_transform(transform)
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.shape[-1] != 3 or not np.isfinite(xyz).all():
        raise GeometryError("Points must be finite with shape [...,3]")
    return xyz @ matrix[:3, :3].T + matrix[:3, 3]


def pose_from_xy_yaw(x: float, y: float, z: float, yaw_rad: float) -> FloatArray:
    cosine, sine = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array(
        [[cosine, -sine, 0, x], [sine, cosine, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], dtype=np.float64
    )


def rotation_angle(transform_a: ArrayLike, transform_b: ArrayLike) -> float:
    relative = invert_transform(transform_a) @ validate_transform(transform_b)
    cosine = np.clip((np.trace(relative[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def opencv_camera_to_world(usd_camera_to_world: ArrayLike, native_to_dataset_world: ArrayLike | None = None) -> FloatArray:
    native_to_dataset = np.eye(4) if native_to_dataset_world is None else validate_transform(native_to_dataset_world)
    return compose_transforms(native_to_dataset, usd_camera_to_world, T_USD_CAMERA_OPENCV_CAMERA)

