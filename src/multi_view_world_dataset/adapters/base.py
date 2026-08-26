from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from multi_view_world_dataset.rendering.bev import BEVCalibration
from multi_view_world_dataset.schema.records import BaseSceneRecord, ObjectState


@dataclass(frozen=True)
class BEVRender:
    calibration: BEVCalibration
    modalities: dict[str, np.ndarray]
    projection_token: str
    contains_robots: bool


class BaseSimulatorAdapter(ABC):
    """Only boundary where a backend may expose native objects internally."""

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def discover_scenes(self) -> list[str]: ...

    @abstractmethod
    def load_scene(self, scene_id: str, *, robot_count: int = 0, development_robot: str = "turtlebot") -> None: ...

    @abstractmethod
    def scene_record(self, split: str) -> BaseSceneRecord: ...

    @abstractmethod
    def object_catalog(self) -> tuple[ObjectState, ...]: ...

    @abstractmethod
    def dump_snapshot(self) -> np.ndarray: ...

    @abstractmethod
    def load_snapshot(self, snapshot: np.ndarray) -> None: ...

    @abstractmethod
    def render_floor_bev(
        self, floor_index: int, calibration: BEVCalibration, *, include_robots: bool, modalities: tuple[str, ...]
    ) -> BEVRender: ...

    @abstractmethod
    def runtime_report(self) -> dict[str, Any]: ...

    def __enter__(self) -> "BaseSimulatorAdapter":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

