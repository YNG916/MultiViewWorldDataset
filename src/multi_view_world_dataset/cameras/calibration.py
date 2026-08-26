from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from multi_view_world_dataset.cameras.transforms import invert_transform, transform_points, validate_transform
from multi_view_world_dataset.errors import GeometryError

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PinholeCalibration:
    width: int
    height: int
    hfov_deg: float
    near_m: float
    far_m: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise GeometryError("Image dimensions must be positive")
        if not (0 < self.hfov_deg < 180):
            raise GeometryError("HFOV must be between 0 and 180 degrees")
        if not (0 < self.near_m < self.far_m):
            raise GeometryError("Invalid clipping range")

    @property
    def pixel_intrinsics(self) -> FloatArray:
        focal = (self.width / 2.0) / np.tan(np.deg2rad(self.hfov_deg) / 2.0)
        # Square pixels; vertical FOV follows from the image aspect ratio.
        return np.array(
            [[focal, 0.0, self.width / 2.0], [0.0, focal, self.height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def normalized_intrinsics(self) -> FloatArray:
        matrix = self.pixel_intrinsics.copy()
        matrix[0] /= self.width
        matrix[1] /= self.height
        return matrix

    @property
    def focal_length_over_aperture(self) -> float:
        return 1.0 / (2.0 * np.tan(np.deg2rad(self.hfov_deg) / 2.0))


def backproject_depth(
    depth_linear: ArrayLike,
    pixel_intrinsics: ArrayLike,
    camera_to_world: ArrayLike,
    *,
    stride: int = 1,
) -> FloatArray:
    depth = np.asarray(depth_linear, dtype=np.float64)
    intrinsics = np.asarray(pixel_intrinsics, dtype=np.float64)
    if depth.ndim != 2 or intrinsics.shape != (3, 3) or stride < 1:
        raise GeometryError("Depth must be HxW, intrinsics 3x3, and stride positive")
    validate_transform(camera_to_world)
    rows, columns = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    sampled = depth[::stride, ::stride]
    valid = np.isfinite(sampled) & (sampled > 0)
    z = sampled[valid]
    x = (columns[valid] + 0.5 - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (rows[valid] + 0.5 - intrinsics[1, 2]) * z / intrinsics[1, 1]
    return transform_points(camera_to_world, np.column_stack((x, y, z)))


def project_world_points(
    world_points: ArrayLike,
    pixel_intrinsics: ArrayLike,
    camera_to_world: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    points_camera = transform_points(invert_transform(camera_to_world), world_points)
    intrinsics = np.asarray(pixel_intrinsics, dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise GeometryError("Intrinsics must be 3x3")
    z = points_camera[:, 2]
    pixels = np.full((len(points_camera), 2), np.nan, dtype=np.float64)
    in_front = z > 0
    pixels[in_front, 0] = intrinsics[0, 0] * points_camera[in_front, 0] / z[in_front] + intrinsics[0, 2] - 0.5
    pixels[in_front, 1] = intrinsics[1, 1] * points_camera[in_front, 1] / z[in_front] + intrinsics[1, 2] - 0.5
    return pixels, z

