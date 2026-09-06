from pathlib import Path

import numpy as np

from visualize_dataset import (
    Source,
    _categorical,
    convert_source,
    discover_episodes,
    discover_sources,
    scalar_bounds,
    visualize_frame,
)


def test_visualizers_handle_each_modality_deterministically():
    labels = np.array([[0, 7], [9, 7]], dtype=np.int64)
    first = _categorical(labels)
    second = _categorical(labels)
    assert np.array_equal(first, second)
    assert np.array_equal(first[0, 0], [0, 0, 0])
    assert np.array_equal(first[0, 1], first[1, 1])

    depth = np.array([[1.0, 2.0], [np.inf, np.nan]], dtype=np.float32)
    output = visualize_frame("depth_linear", depth, scalar_bounds(depth))
    assert output.shape == (2, 2, 3)
    assert output.dtype == np.uint8
    assert np.array_equal(output[1, 0], [0, 0, 0])

    normals = np.array([[[-1.0, 0.0, 1.0, 1.0]]], dtype=np.float32)
    assert np.array_equal(visualize_frame("normal", normals)[0, 0], [0, 127, 255])


def test_episode_and_source_discovery_includes_environment_and_all_views(tmp_path: Path):
    dataset = tmp_path / "dataset"
    episode = dataset / "episodes" / "Rs_int" / "config_000" / "episode_000"
    config_bev = dataset / "configurations" / "Rs_int" / "config_000" / "bev"
    for path in (
        episode / "bev" / "world_before.npz",
        episode / "bev" / "world_after.npz",
        episode / "bev" / "environment_after.npz",
        episode / "robot_views" / "before" / "robot_00.npz",
        episode / "robot_views" / "after" / "robot_00.npz",
        config_bev / "environment_base.npz",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, rgb=np.zeros((2, 3, 3), dtype=np.uint8))

    assert discover_episodes(dataset) == [episode]
    assert [source.name for source in discover_sources(episode)] == [
        "environment_base",
        "environment_after",
        "world_before",
        "robot_before_robot_00",
        "world_after",
        "robot_after_robot_00",
    ]


def test_convert_source_writes_every_temporal_modality(tmp_path: Path):
    npz = tmp_path / "robot_00.npz"
    rgb = np.zeros((2, 8, 10, 3), dtype=np.uint8)
    depth = np.array([np.linspace(0.2, 3.0, 80).reshape(8, 10)] * 2, dtype=np.float32)
    semantic = np.zeros((2, 8, 10), dtype=np.int32)
    semantic[:, 2:5, 3:7] = 42
    np.savez_compressed(npz, rgb=rgb, depth_linear=depth, semantic=semantic)

    output = tmp_path / "out"
    report = convert_source(
        Source("robot_before_robot_00", npz, temporal=True),
        output,
        output_kind="video",
        fps=10.0,
        codec="MJPG",
        video_extension=".avi",
        contact_frames=2,
        thumbnail_width=40,
        frame_stride=1,
        maximum_occupancy_fraction=0.98,
        allow_suspicious_bev=False,
    )

    assert {item["key"] for item in report["items"]} == {"rgb", "depth_linear", "semantic"}
    for modality in ("rgb", "depth_linear", "semantic"):
        assert (output / "robot_before_robot_00" / f"{modality}.avi").is_file()
        assert (output / "robot_before_robot_00" / f"{modality}_contact.png").is_file()
