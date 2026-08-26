from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _matrix4(value: Any, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4, 4) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    return array


class InterventionType(str, Enum):
    RIGID_RELOCATION = "rigid_relocation"
    ARTICULATION = "articulation"
    STATE_CHANGE = "state_change"


class ApplicationMode(str, Enum):
    PRE_ROLLOUT = "pre_rollout"
    TIMED = "timed"


@dataclass(frozen=True)
class BaseSceneRecord:
    scene_id: str
    scene_family: str
    simulator_scene_model: str
    floor_ids: tuple[str, ...]
    floor_heights_m: tuple[float, ...]
    split: str
    object_catalog_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectState:
    instance_id: str
    asset_uid: str
    category: str
    native_path: str
    structural: bool
    movable: bool
    articulated: bool
    available_states: tuple[str, ...]
    object_to_world: FloatArray
    bbox_min_world: tuple[float, float, float]
    bbox_max_world: tuple[float, float, float]
    scale: tuple[float, float, float]
    joint_names: tuple[str, ...] = ()
    joint_limits: tuple[tuple[float, float], ...] = ()
    joint_values: tuple[float, ...] = ()
    room_id: str | None = None
    floor_id: str | None = None
    relations: tuple[dict[str, Any], ...] = ()
    semantic_states: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_to_world", _matrix4(self.object_to_world, "object_to_world"))
        if len(self.joint_names) != len(self.joint_values):
            raise ValueError("joint_names and joint_values lengths differ")


@dataclass(frozen=True)
class RobotState:
    robot_id: str
    model: str
    base_to_world: FloatArray
    camera_height_m: float
    mast_joint_value_m: float | None = None
    joint_values: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_to_world", _matrix4(self.base_to_world, "base_to_world"))


@dataclass(frozen=True)
class CameraState:
    camera_id: str
    robot_id: str
    width: int
    height: int
    pixel_intrinsics: FloatArray
    normalized_intrinsics: FloatArray
    camera_to_world: FloatArray
    world_to_camera: FloatArray
    robot_base_to_world: FloatArray
    camera_to_robot_base: FloatArray
    near_m: float
    far_m: float
    camera_height_m: float
    projection: str = "perspective"

    def __post_init__(self) -> None:
        for name in (
            "camera_to_world",
            "world_to_camera",
            "robot_base_to_world",
            "camera_to_robot_base",
        ):
            object.__setattr__(self, name, _matrix4(getattr(self, name), name))
        for name in ("pixel_intrinsics", "normalized_intrinsics"):
            matrix = np.asarray(getattr(self, name), dtype=np.float64)
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                raise ValueError(f"{name} must be a finite 3x3 matrix")
            object.__setattr__(self, name, matrix)
        if not (0 < self.near_m < self.far_m):
            raise ValueError("Camera clipping range must satisfy 0 < near < far")


@dataclass(frozen=True)
class WorldState:
    scene_id: str
    configuration_id: str
    physical_time_index: int | None
    objects: tuple[ObjectState, ...]
    robots: tuple[RobotState, ...] = ()
    changed_object_tracks: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    simulator_snapshot_ref: str | None = None


@dataclass(frozen=True)
class DynamicConfiguration:
    configuration_id: str
    scene_id: str
    seed: int
    exact_state_hash: str
    world_state: WorldState
    environment_bev_ref: str
    simulator_snapshot_ref: str
    accepted_attempt: int


@dataclass(frozen=True)
class Trajectory:
    robot_id: str
    fps: float
    base_to_world: FloatArray
    camera_to_world: FloatArray

    def __post_init__(self) -> None:
        base = np.asarray(self.base_to_world, dtype=np.float64)
        camera = np.asarray(self.camera_to_world, dtype=np.float64)
        if base.ndim != 3 or base.shape[1:] != (4, 4):
            raise ValueError("base_to_world trajectory must have shape [T,4,4]")
        if camera.shape != base.shape:
            raise ValueError("camera_to_world trajectory must match base_to_world shape")
        if not np.isfinite(base).all() or not np.isfinite(camera).all():
            raise ValueError("Trajectory contains non-finite poses")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        object.__setattr__(self, "base_to_world", base)
        object.__setattr__(self, "camera_to_world", camera)

    @property
    def frames(self) -> int:
        return int(self.base_to_world.shape[0])


@dataclass(frozen=True)
class InterventionEvent:
    event_id: str
    intervention_type: InterventionType
    target_instance_id: str
    application_mode: ApplicationMode
    time_index: int | None
    parameters: dict[str, Any]
    before_object_state: ObjectState | None = None
    after_object_state: ObjectState | None = None

    def __post_init__(self) -> None:
        if self.application_mode is ApplicationMode.PRE_ROLLOUT and self.time_index is not None:
            raise ValueError("pre_rollout event must have time_index=None")
        if self.application_mode is ApplicationMode.TIMED and (self.time_index is None or self.time_index < 0):
            raise ValueError("timed event requires a non-negative physical time_index")


@dataclass(frozen=True)
class Observation:
    robot_id: str
    physical_time_index: int
    camera: CameraState
    modality_refs: dict[str, str]


@dataclass(frozen=True)
class QAResult:
    check: str
    passed: bool
    reason: str | None = None
    metrics: dict[str, float | int | str | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class WorldEpisode:
    episode_id: str
    scene_id: str
    configuration_id: str
    seed: int
    simulator_versions: dict[str, str | None]
    generator_git_commit: str | None
    trajectories: tuple[Trajectory, ...]
    intervention: InterventionEvent
    state_before: WorldState
    state_after: WorldState
    environment_after_bev_ref: str
    world_before_bev_ref: str
    world_after_bev_ref: str
    observations_before: tuple[Observation, ...]
    observations_after: tuple[Observation, ...]
    qa: tuple[QAResult, ...]

    def __post_init__(self) -> None:
        if len(self.trajectories) != 3:
            raise ValueError("Dataset v1 episode requires exactly 3 trajectories")
        frame_counts = {trajectory.frames for trajectory in self.trajectories}
        if len(frame_counts) != 1:
            raise ValueError("All robot trajectories must have the same physical frame count")

