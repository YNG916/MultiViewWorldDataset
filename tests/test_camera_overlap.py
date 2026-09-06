import numpy as np

from multi_view_world_dataset.adapters.omnigibson import robot_multimodal_alignment_metrics
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


def _synthetic_robot_frames(view_offset: int = 0):
    masks = []
    for index in range(3):
        mask = np.zeros((36, 48), dtype=np.float32)
        row = 4 + 8 * index
        column = 5 + 11 * index
        mask[row : row + 7, column : column + 9] = 1.0
        masks.append(mask)
    frames = {}
    for index in range(3):
        modality_mask = masks[(index + view_offset) % 3]
        rgb = np.repeat(np.repeat(modality_mask, 2, axis=0), 2, axis=1)
        rgb = np.repeat(rgb[..., None], 3, axis=-1) * 255.0
        normal = np.repeat(modality_mask[..., None], 3, axis=-1)
        frame = {
            "rgb": rgb,
            "depth_linear": 2.0 + masks[index],
            "semantic": modality_mask.astype(np.int32),
            "instance": (modality_mask * (index + 2)).astype(np.int32),
            "normal": normal,
        }
        frames[f"robot_{index:02d}"] = {
            name: [value.copy() for _ in range(3)]
            for name, value in frame.items()
        }
    return frames


def test_multimodal_alignment_identifies_each_robots_own_view():
    metrics = robot_multimodal_alignment_metrics(
        _synthetic_robot_frames(), keyframe_count=3
    )
    assert all(
        value["own_view_best_fraction"] == 1.0
        for value in metrics["modalities"].values()
    )


def test_multimodal_alignment_rejects_cross_robot_view_rotation():
    metrics = robot_multimodal_alignment_metrics(
        _synthetic_robot_frames(view_offset=1), keyframe_count=3
    )
    assert all(
        value["own_view_best_fraction"] == 0.0
        for value in metrics["modalities"].values()
    )
