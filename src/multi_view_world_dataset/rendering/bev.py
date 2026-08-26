from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from multi_view_world_dataset.errors import GeometryError

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BEVCalibration:
    floor_id: str
    floor_z: float
    meters_per_pixel: float
    world_bounds: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        xmin, ymin, xmax, ymax = self.world_bounds
        if self.meters_per_pixel <= 0 or xmax <= xmin or ymax <= ymin:
            raise GeometryError("Invalid BEV bounds or resolution")
        width_f = (xmax - xmin) / self.meters_per_pixel
        height_f = (ymax - ymin) / self.meters_per_pixel
        if not np.isclose(width_f, round(width_f), atol=1e-7) or not np.isclose(height_f, round(height_f), atol=1e-7):
            raise GeometryError("BEV bounds must span an integer number of pixels")

    @classmethod
    def from_bounds(
        cls,
        floor_id: str,
        floor_z: float,
        meters_per_pixel: float,
        raw_bounds: tuple[float, float, float, float],
        margin_m: float = 0.0,
    ) -> "BEVCalibration":
        xmin, ymin, xmax, ymax = raw_bounds
        resolution = meters_per_pixel
        xmin = np.floor((xmin - margin_m) / resolution) * resolution
        ymin = np.floor((ymin - margin_m) / resolution) * resolution
        xmax = np.ceil((xmax + margin_m) / resolution) * resolution
        ymax = np.ceil((ymax + margin_m) / resolution) * resolution
        return cls(floor_id, floor_z, resolution, (float(xmin), float(ymin), float(xmax), float(ymax)))

    @property
    def width(self) -> int:
        return int(round((self.world_bounds[2] - self.world_bounds[0]) / self.meters_per_pixel))

    @property
    def height(self) -> int:
        return int(round((self.world_bounds[3] - self.world_bounds[1]) / self.meters_per_pixel))

    @property
    def pixel_to_world_transform(self) -> FloatArray:
        xmin, _, _, ymax = self.world_bounds
        m = self.meters_per_pixel
        return np.array([[m, 0, 0, xmin + 0.5 * m], [0, -m, 0, ymax - 0.5 * m], [0, 0, 1, self.floor_z], [0, 0, 0, 1]])

    def pixel_to_world(self, pixels_uv: ArrayLike) -> FloatArray:
        pixels = np.asarray(pixels_uv, dtype=np.float64)
        if pixels.shape[-1] != 2:
            raise GeometryError("BEV pixels must have shape [...,2]")
        xmin, _, _, ymax = self.world_bounds
        world = np.empty(pixels.shape[:-1] + (3,), dtype=np.float64)
        world[..., 0] = xmin + (pixels[..., 0] + 0.5) * self.meters_per_pixel
        world[..., 1] = ymax - (pixels[..., 1] + 0.5) * self.meters_per_pixel
        world[..., 2] = self.floor_z
        return world

    def world_to_pixel(self, world_xyz: ArrayLike) -> FloatArray:
        world = np.asarray(world_xyz, dtype=np.float64)
        if world.shape[-1] != 3:
            raise GeometryError("World points must have shape [...,3]")
        xmin, _, _, ymax = self.world_bounds
        pixels = np.empty(world.shape[:-1] + (2,), dtype=np.float64)
        pixels[..., 0] = (world[..., 0] - xmin) / self.meters_per_pixel - 0.5
        pixels[..., 1] = (ymax - world[..., 1]) / self.meters_per_pixel - 0.5
        return pixels

