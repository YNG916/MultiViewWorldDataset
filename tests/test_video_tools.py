from pathlib import Path

import numpy as np
import pytest

from npz2video_bev import load_rgb_frames


def test_bev_loader_accepts_rgba_and_normalizes_float_rgb(tmp_path: Path):
    path = tmp_path / "world_before.npz"
    rgba = np.zeros((2, 4, 5, 4), dtype=np.float32)
    rgba[..., 0] = 1.0
    occupancy = np.zeros((2, 4, 5), dtype=np.uint8)
    occupancy[:, 1, 1] = 1
    np.savez_compressed(path, rgb=rgba, occupancy=occupancy)

    frames = load_rgb_frames(path)

    assert frames.shape == (2, 4, 5, 3)
    assert frames.dtype == np.uint8
    assert np.all(frames[..., 0] == 255)


def test_bev_loader_rejects_stale_perspective_occupancy(tmp_path: Path):
    path = tmp_path / "world_after.npz"
    rgb = np.zeros((3, 4, 5, 3), dtype=np.uint8)
    occupancy = np.zeros((3, 4, 5), dtype=np.uint8)
    occupancy[1] = 1
    np.savez_compressed(path, rgb=rgb, occupancy=occupancy)

    with pytest.raises(ValueError, match="stale robot-perspective capture bug"):
        load_rgb_frames(path)


def test_robot_loader_mode_does_not_require_occupancy(tmp_path: Path):
    path = tmp_path / "robot_00.npz"
    rgb = np.zeros((2, 4, 5, 3), dtype=np.uint8)
    np.savez_compressed(path, rgb=rgb)

    frames = load_rgb_frames(path, validate_bev=False)

    assert np.array_equal(frames, rgb)
