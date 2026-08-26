import numpy as np

from multi_view_world_dataset.cameras.calibration import PinholeCalibration
from multi_view_world_dataset.cameras.overlap import build_overlap_graph, pairwise_visible_surface_overlap


def test_identical_planar_depth_has_full_overlap():
    calibration = PinholeCalibration(64, 32, 70, 0.1, 15)
    depth = np.full((32, 64), 3.0)
    overlap = pairwise_visible_surface_overlap(
        depth, calibration.pixel_intrinsics, np.eye(4),
        depth, calibration.pixel_intrinsics, np.eye(4),
        stride=4, tolerance_m=1e-5,
    )
    assert overlap == 1.0


def test_connected_graph_does_not_require_all_pairs():
    ids = ("a", "b", "c")
    depths = {name: np.full((16, 16), 2.0) for name in ids}
    calibration = PinholeCalibration(16, 16, 70, 0.1, 15)
    intrinsics = {name: calibration.pixel_intrinsics for name in ids}
    poses = {name: np.eye(4) for name in ids}
    graph = build_overlap_graph(
        ids, depths, intrinsics, poses, edge_threshold=0.5, near_duplicate_threshold=1.1, stride=2
    )
    assert graph.connected
    assert len(graph.edges) == 3
