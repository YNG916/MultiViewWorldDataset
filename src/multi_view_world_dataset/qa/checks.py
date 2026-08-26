from __future__ import annotations

import numpy as np

from multi_view_world_dataset.cameras.overlap import OverlapGraph
from multi_view_world_dataset.cameras.transforms import compose_transforms, invert_transform
from multi_view_world_dataset.errors import GeometryError, SampleRejected
from multi_view_world_dataset.schema.records import CameraState, ObjectState, QAResult, Trajectory


def check_camera_calibration(camera: CameraState, atol: float = 1e-6) -> QAResult:
    identity = compose_transforms(camera.camera_to_world, camera.world_to_camera)
    base_composed = compose_transforms(camera.robot_base_to_world, camera.camera_to_robot_base)
    passed = np.allclose(identity, np.eye(4), atol=atol) and np.allclose(base_composed, camera.camera_to_world, atol=atol)
    return QAResult("camera_calibration", bool(passed), None if passed else "transform_inverse_or_composition_failed")


def check_depth(depth: np.ndarray, near_m: float, far_m: float) -> QAResult:
    values = np.asarray(depth)
    finite = np.isfinite(values)
    valid = finite & (values >= near_m) & (values <= far_m)
    ratio = float(valid.sum() / max(1, finite.sum()))
    passed = bool(finite.any() and ratio > 0.99)
    return QAResult("finite_valid_depth", passed, None if passed else "invalid_depth", {"valid_ratio": ratio})


def check_instance_ids(*catalogs: tuple[ObjectState, ...]) -> QAResult:
    if not catalogs:
        return QAResult("stable_instance_ids", False, "no_catalogs")
    baseline = {(obj.asset_uid, obj.category, obj.native_path): obj.instance_id for obj in catalogs[0]}
    unique = len(set(baseline.values())) == len(baseline)
    stable = all(
        {(obj.asset_uid, obj.category, obj.native_path): obj.instance_id for obj in catalog} == baseline for catalog in catalogs[1:]
    )
    passed = unique and stable
    return QAResult("stable_instance_ids", passed, None if passed else "duplicate_or_unstable_instance_id")


def check_overlap_graph(graph: OverlapGraph) -> QAResult:
    passed = graph.connected and not graph.near_duplicate_pairs
    reason = None if passed else ("overlap_graph_disconnected" if not graph.connected else "near_duplicate_views")
    metrics = {f"omega_{left}_{right}": value for (left, right), value in graph.overlaps.items()}
    return QAResult("overlap_graph", passed, reason, metrics)


def check_paired_trajectories(
    before: tuple[Trajectory, ...],
    after: tuple[Trajectory, ...],
    *,
    position_atol_m: float = 1e-5,
    matrix_atol: float = 1e-6,
) -> QAResult:
    before_by_id = {trajectory.robot_id: trajectory for trajectory in before}
    after_by_id = {trajectory.robot_id: trajectory for trajectory in after}
    passed = before_by_id.keys() == after_by_id.keys()
    max_position_error = 0.0
    max_matrix_error = 0.0
    if passed:
        for robot_id, left in before_by_id.items():
            right = after_by_id[robot_id]
            if left.base_to_world.shape != right.base_to_world.shape:
                passed = False
                break
            max_position_error = max(
                max_position_error,
                float(np.max(np.abs(left.base_to_world[:, :3, 3] - right.base_to_world[:, :3, 3]))),
            )
            max_matrix_error = max(max_matrix_error, float(np.max(np.abs(left.camera_to_world - right.camera_to_world))))
        passed = passed and max_position_error <= position_atol_m and max_matrix_error <= matrix_atol
    return QAResult(
        "paired_trajectory_equality",
        bool(passed),
        None if passed else "before_after_trajectory_mismatch",
        {"max_position_error_m": max_position_error, "max_matrix_error": max_matrix_error},
    )


def check_bev_pair(before_calibration: object, after_calibration: object, before_has_robots: bool, after_has_robots: bool) -> QAResult:
    same = before_calibration == after_calibration
    passed = same and before_has_robots and after_has_robots
    return QAResult("paired_world_bev", passed, None if passed else "bev_calibration_or_robot_presence_failed")


def require_all(results: tuple[QAResult, ...] | list[QAResult]) -> None:
    failures = [result for result in results if not result.passed]
    if failures:
        reason = ",".join(f"{result.check}:{result.reason}" for result in failures)
        if any(result.check == "camera_calibration" for result in failures):
            raise GeometryError(reason)
        raise SampleRejected(reason)

