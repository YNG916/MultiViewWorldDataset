import numpy as np
import pytest

from multi_view_world_dataset.cameras.calibration import PinholeCalibration, backproject_depth, project_world_points
from multi_view_world_dataset.cameras.transforms import (
    T_USD_CAMERA_OPENCV_CAMERA,
    compose_transforms,
    invert_transform,
    opencv_camera_to_world,
    pose_from_xy_yaw,
    transform_points,
    validate_transform,
)
from multi_view_world_dataset.errors import GeometryError
from multi_view_world_dataset.rendering.bev import BEVCalibration


def test_transform_inverse_and_composition():
    transform = pose_from_xy_yaw(1.2, -3.4, 0.7, 0.63)
    inverse = invert_transform(transform)
    np.testing.assert_allclose(transform @ inverse, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(compose_transforms(transform, inverse), np.eye(4), atol=1e-12)


def test_world_camera_world_roundtrip():
    camera_to_world = pose_from_xy_yaw(2.0, 1.0, 0.8, -0.4)
    points_camera = np.array([[0.1, -0.2, 1.0], [1.0, 2.0, 4.0], [-0.4, 0.2, 2.5]])
    points_world = transform_points(camera_to_world, points_camera)
    recovered = transform_points(invert_transform(camera_to_world), points_world)
    np.testing.assert_allclose(recovered, points_camera, atol=1e-12)


def test_usd_to_opencv_camera_axes():
    camera_to_world = opencv_camera_to_world(np.eye(4))
    np.testing.assert_array_equal(camera_to_world, T_USD_CAMERA_OPENCV_CAMERA)
    # OpenCV +Z forward becomes USD/world -Z for an identity USD camera.
    np.testing.assert_array_equal(transform_points(camera_to_world, [[0, 0, 1]]), [[0, 0, -1]])


def test_calibrated_depth_backprojection_and_projection():
    calibration = PinholeCalibration(8, 6, 70.0, 0.1, 15.0)
    depth = np.full((6, 8), 2.0)
    camera_to_world = pose_from_xy_yaw(1.0, 2.0, 0.5, 0.3)
    points_world = backproject_depth(depth, calibration.pixel_intrinsics, camera_to_world)
    pixels, recovered_depth = project_world_points(points_world, calibration.pixel_intrinsics, camera_to_world)
    columns, rows = np.meshgrid(np.arange(8), np.arange(6))
    np.testing.assert_allclose(pixels, np.column_stack((columns.ravel(), rows.ravel())), atol=1e-10)
    np.testing.assert_allclose(recovered_depth, 2.0, atol=1e-10)


def test_bev_pixel_world_pixel_roundtrip():
    calibration = BEVCalibration.from_bounds("floor_00", 0.2, 0.02, (-1.01, -2.02, 3.04, 4.08), 0.5)
    pixels = np.array([[0.0, 0.0], [10.25, 20.75], [calibration.width - 1, calibration.height - 1]])
    world = calibration.pixel_to_world(pixels)
    recovered = calibration.world_to_pixel(world)
    np.testing.assert_allclose(recovered, pixels, atol=1e-10)


def test_bad_transform_fails_loudly():
    invalid = np.eye(4)
    invalid[0, 0] = 2
    with pytest.raises(GeometryError):
        validate_transform(invalid)
