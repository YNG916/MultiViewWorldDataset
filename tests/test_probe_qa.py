from __future__ import annotations

import numpy as np

from multi_view_world_dataset.adapters.base import BEVRender
from multi_view_world_dataset.adapters.omnigibson import OmniGibsonAdapter
from multi_view_world_dataset.pipeline import _bev_content_metrics, _bev_geometry_metrics
from multi_view_world_dataset.rendering.bev import BEVCalibration


def _render(*, world: bool, robot_pixels: bool = True) -> BEVRender:
    calibration = BEVCalibration("floor_00", 0.0, 1.0, (0.0, 0.0, 6.0, 2.0))
    instance = np.full((2, 6, 1), 2, dtype=np.int64)
    instance_id = np.zeros((2, 6, 1), dtype=np.int64)
    id_mapping: dict[int, str] = {0: "BACKGROUND"}
    if world and robot_pixels:
        instance_id[..., 0] = np.repeat([3, 4, 5], 4).reshape(2, 6)
        id_mapping.update(
            {
                3: "/World/robot_00/base/visuals",
                4: "/World/robot_01/base/visuals",
                5: "/World/robot_02/base/visuals",
            }
        )
    metadata = {
        "horizontal_aperture": 60.0,
        "vertical_aperture": 20.0,
        "segmentation_info": {
            "seg_instance": {2: "chair_0"},
            "seg_instance_id": id_mapping,
        },
    }
    return BEVRender(
        calibration,
        {
            "rgb": np.zeros((2, 6, 4), dtype=np.uint8),
            "instance": instance,
            "instance_id": instance_id,
        },
        "orthographic",
        world,
        metadata,
    )


def test_bev_geometry_requires_exact_shape_and_ten_x_aperture():
    render = _render(world=False)
    passed, metrics = _bev_geometry_metrics(render)
    assert passed
    assert metrics["expected_aperture"] == [60.0, 20.0]

    bad = BEVRender(
        render.calibration,
        {"rgb": np.zeros((1, 6, 4), dtype=np.uint8)},
        "orthographic",
        False,
        {**render.metadata, "horizontal_aperture": 4.0},
    )
    assert not _bev_geometry_metrics(bad)[0]


def test_bev_content_requires_furniture_robot_mask_and_each_world_robot():
    environment = _render(world=False)
    world = _render(world=True)
    passed, metrics = _bev_content_metrics(environment, world, ("robot_00", "robot_01", "robot_02"))
    assert passed
    assert metrics["visible_nonstructural_pixels"] == 12
    assert metrics["environment_robot_pixels"] == {"robot_00": 0, "robot_01": 0, "robot_02": 0}

    missing_robot = _render(world=True, robot_pixels=False)
    assert not _bev_content_metrics(environment, missing_robot, ("robot_00", "robot_01", "robot_02"))[0]


def test_adapter_pose_matrix_returns_rigid_matrix_after_bounded_float_drift():
    class FakeTransformUtils:
        @staticmethod
        def pose2mat(pose):
            matrix = np.eye(4, dtype=np.float32)
            matrix[0, 0] += 1.0e-6
            matrix[:3, 3] = pose[0]
            return matrix

    class FakeObject:
        @staticmethod
        def get_position_orientation():
            return np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0, 1.0])

    adapter = object.__new__(OmniGibsonAdapter)
    adapter._transform_utils = FakeTransformUtils()
    adapter._runtime_findings = {}
    matrix = adapter._pose_matrix(FakeObject())
    assert matrix.shape == (4, 4)
    assert np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3))
    assert np.allclose(matrix[:3, 3], [1.0, 2.0, 3.0])
    assert adapter._runtime_findings["maximum_pose_rotation_projection_error"] > 0
