import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

import multi_view_world_dataset.rendering.labels as label_module
from multi_view_world_dataset.adapters.omnigibson import OmniGibsonAdapter
from multi_view_world_dataset.errors import ConfigurationError, SampleRejected, SimulatorUnavailableError
from multi_view_world_dataset.generator import (
    _gt_valid_candidate_soft_score,
    _robot_states,
    _sparse_intervention_visibility_preflight,
    _temporal_overlap_preflight,
)
from multi_view_world_dataset.rendering.labels import (
    SemanticIDCollisionError,
    remap_public_labels,
    stable_semantic_id,
    validate_semantic_id_uniqueness,
)
from multi_view_world_dataset.rendering.inspection_v11 import save_robot_appearance_summary
from multi_view_world_dataset.sampling.diversity import (
    complementary_hybrid_trajectory_sets,
    joint_trajectory_metrics,
    regime_trajectory_soft_score,
    stable_seed,
    temporal_overlap_acceptance,
)
from multi_view_world_dataset.sampling.trajectories import trajectory_from_spatial_path
from multi_view_world_dataset.schema.records import ObjectState, RobotState
from multi_view_world_dataset.storage.writer import DatasetWriter


def _object(instance_id: str, path: str, category: str = "chair") -> ObjectState:
    return ObjectState(
        instance_id=instance_id, asset_uid=instance_id, category=category,
        native_path=path, structural=False, movable=True, articulated=False,
        available_states=(), object_to_world=np.eye(4),
        bbox_min_world=(0, 0, 0), bbox_max_world=(1, 1, 1), scale=(1, 1, 1),
    )


def test_sparse_intervention_visibility_preflight_uses_public_instance_ids():
    catalog = (
        _object("chair_a", "/World/chair_a"),
        _object("chair_b", "/World/chair_b"),
    )
    views = {
        "robot_00": {"instance": np.full((3, 8, 8), 4, dtype=np.int32)},
        "robot_01": {"instance": np.zeros((3, 8, 8), dtype=np.int32)},
        "robot_02": {"instance": np.zeros((3, 8, 8), dtype=np.int32)},
    }
    config = {
        "intervention": {
            "target_visibility": {
                "minimum_pixels": 16,
                "minimum_frames": 2,
                "minimum_robots": 1,
                "preferred_robots": 2,
            }
        },
        "navigation": {"minimum_visible_intervention_candidates": 1},
    }
    result = _sparse_intervention_visibility_preflight(
        catalog,
        views,
        config,
        sampled_frame_count=3,
        full_frame_count=60,
    )
    assert result["passed"]
    assert result["eligible_target_ids"] == ["chair_a"]
    assert result["objects"]["chair_a"]["public_instance_id"] == 4


def test_stable_seed_depends_on_identifiers_not_enumeration_position():
    first = stable_seed(17, "Rs_int", "config_003", "episode_002")
    assert first == stable_seed(17, "Rs_int", "config_003", "episode_002")
    assert first != stable_seed(17, "Rs_int", "config_004", "episode_002")


def test_temporal_overlap_uses_connected_union_and_robot_participation():
    frames = [
        {"connected": False, "edges": [["a", "b"]], "near_duplicate_pairs": []},
        {"connected": False, "edges": [["b", "c"]], "near_duplicate_pairs": []},
        {"connected": True, "edges": [["a", "b"], ["b", "c"]], "near_duplicate_pairs": []},
    ]
    result = temporal_overlap_acceptance(
        ("a", "b", "c"), frames, regime="dense_shared",
        regime_connected_fraction_target={
            "dense_shared": 0.6, "partial_chain": 0.3, "exploratory": 0.2,
        },
        regime_shared_keyframe_fraction_target={
            "dense_shared": 0.6, "partial_chain": 0.3, "exploratory": 0.2,
        },
        regime_maximum_consecutive_isolated_keyframes={
            "dense_shared": 1, "partial_chain": 2, "exploratory": 2,
        },
    )
    assert result["passed"]
    assert result["checks"]["union_graph_connected"]
    assert result["connected_fraction"] == pytest.approx(1 / 3)
    assert not result["connected_fraction_target_met"]
    assert result["realized_regime"] == "partial_chain"
    assert not result["regime_target_match"]
    isolated = temporal_overlap_acceptance(
        ("a", "b", "c"), frames, regime="partial_chain",
        regime_connected_fraction_target={
            "dense_shared": 0.6, "partial_chain": 0.3, "exploratory": 0.2,
        },
        regime_shared_keyframe_fraction_target={
            "dense_shared": 0.6, "partial_chain": 0.3, "exploratory": 0.2,
        },
        regime_maximum_consecutive_isolated_keyframes={
            "dense_shared": 0, "partial_chain": 0, "exploratory": 0,
        },
    )


def test_temporal_overlap_preflight_uses_mounted_depth_only():
    robot_ids = ("robot_00", "robot_01", "robot_02")

    class DepthOnlyAdapter:
        def __init__(self):
            self.placed_frames = []

        def place_robots_at_trajectory_frame(self, trajectories, frame_index):
            self.placed_frames.append(frame_index)

        def robot_depth_observations(self):
            return {
                robot_id: {
                    "depth_linear": np.full((8, 16), 2.0),
                    "camera_to_world": np.eye(4),
                }
                for robot_id in robot_ids
            }

        def robot_observations(self):
            raise AssertionError("full multimodal capture must not run in preflight")

    config = {
        "camera": {"hfov_deg": 70.0, "near_m": 0.1, "far_m": 15.0},
        "overlap": {
            "edge_threshold": 0.20,
            "near_duplicate_threshold": 1.01,
            "reprojection_tolerance_m": 0.08,
        },
        "trajectory": {
            "overlap_preflight": {
                "keyframe_count": 3,
                "geometry_width": 16,
                "geometry_height": 8,
                "depth_sample_stride": 2,
                "regime_connected_fraction_target": {
                    "dense_shared": 0.60, "partial_chain": 0.30,
                    "exploratory": 0.15,
                },
                "regime_shared_keyframe_fraction_target": {
                    "dense_shared": 0.60, "partial_chain": 0.30,
                    "exploratory": 0.15,
                },
                "regime_maximum_consecutive_isolated_keyframes": {
                    "dense_shared": 2, "partial_chain": 4, "exploratory": 5,
                },
            }
        },
    }
    trajectories = tuple(
        SimpleNamespace(
            robot_id=robot_id, frames=3,
            metadata={"observation_regime": "dense_shared"},
        )
        for robot_id in robot_ids
    )
    adapter = DepthOnlyAdapter()
    metrics = _temporal_overlap_preflight(adapter, config, trajectories)
    assert metrics["passed"]
    assert adapter.placed_frames == [0, 1, 2, 0]


def test_public_instance_ids_ignore_transient_renderer_values():
    objects = (_object("stable-chair", "/World/scene/chair"),)
    image = np.asarray([[0, 71], [9, 71]], dtype=np.uint32)
    instance, semantic, mapping = remap_public_labels(
        image,
        {0: "BACKGROUND", 71: "/World/scene/chair/visual", 9: "/World/robot_00/chassis"},
        objects,
        {"robot_00": "/World/robot_00", "robot_01": "/World/robot_01", "robot_02": "/World/robot_02"},
    )
    assert instance.tolist() == [[0, 4], [1, 4]]
    assert semantic[0, 1] == stable_semantic_id("chair")
    assert semantic[1, 0] == 2
    assert mapping["public_instance_to_state"]["4"] == "stable-chair"


def test_resume_refuses_changed_resolved_configuration(tmp_path):
    writer = DatasetWriter(tmp_path)
    writer.initialize({"schema_version": "1.1.0"}, resolved_config={"seed": 1})
    writer.initialize({"schema_version": "1.1.0"}, resolved_config={"seed": 1})
    with pytest.raises(ConfigurationError, match="unsafe resume"):
        writer.initialize({"schema_version": "1.1.0"}, resolved_config={"seed": 2})
    assert (tmp_path / "resolved_config.yaml").is_file()
    assert (tmp_path / "taxonomy.json").stat().st_size > 2


def test_runtime_metadata_update_preserves_configuration_fingerprint(tmp_path):
    writer = DatasetWriter(tmp_path)
    writer.initialize(
        {"schema_version": "1.1.0"}, resolved_config={"seed": 1}
    )
    before = json.loads((tmp_path / "dataset_meta.json").read_text())
    writer.update_dataset_metadata(
        {"simulator_versions": {"omnigibson": "3.9.2"}}
    )
    after = json.loads((tmp_path / "dataset_meta.json").read_text())
    assert (
        after["configuration_fingerprint"]
        == before["configuration_fingerprint"]
    )
    assert after["simulator_versions"]["omnigibson"] == "3.9.2"
    with pytest.raises(ConfigurationError, match="configuration_fingerprint"):
        writer.update_dataset_metadata(
            {"configuration_fingerprint": "changed"}
        )


def test_joint_metrics_record_heading_path_and_spatial_diversity():
    paths = (
        np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        np.asarray([[0.0, 1.0], [0.7, 1.7]]),
        np.asarray([[1.0, 0.5], [0.4, 1.3]]),
    )
    trajectories = tuple(
        trajectory_from_spatial_path(f"robot_{i:02d}", path, 0, np.eye(4), frames=20, fps=10)
        for i, path in enumerate(paths)
    )
    metrics = joint_trajectory_metrics(trajectories)
    assert metrics["mean_pairwise_heading_difference_rad"] > 0.5
    assert metrics["spatial_coverage_bbox_area_m2"] > 1.0
    assert "mean_velocity_profile_correlation" in metrics


def test_dense_coverage_soft_score_saturates_without_compactness_reward():
    saturation = {
        "dense_shared": 8.0,
        "partial_chain": 16.0,
        "exploratory": 30.0,
    }
    moderate = {
        "spatial_coverage_bbox_area_m2": 8.0,
        "mean_inter_robot_distance_m": 2.0,
    }
    very_wide = {**moderate, "spatial_coverage_bbox_area_m2": 80.0}
    assert regime_trajectory_soft_score(moderate, "dense_shared", saturation) == 1.0
    assert regime_trajectory_soft_score(very_wide, "dense_shared", saturation) == 1.0



def test_regime_soft_score_rewards_only_each_robots_own_heading_prior():
    saturation = {
        "dense_shared": 8.0,
        "partial_chain": 16.0,
        "exploratory": 30.0,
    }
    common = {
        "spatial_coverage_bbox_area_m2": 8.0,
        "mean_inter_robot_distance_m": 2.0,
        "mean_pairwise_heading_difference_rad": 0.0,
    }
    aligned = {**common, "mean_initial_heading_prior_alignment": 1.0}
    misaligned = {**common, "mean_initial_heading_prior_alignment": 0.0}
    weights = {"dense_shared": 1.0, "partial_chain": 0.75, "exploratory": 0.0}
    assert regime_trajectory_soft_score(
        aligned, "dense_shared", saturation, heading_prior_weights=weights,
    ) == pytest.approx(2.0)
    assert regime_trajectory_soft_score(
        misaligned, "dense_shared", saturation, heading_prior_weights=weights,
    ) == pytest.approx(1.0)

    divergent_headings = {**aligned, "mean_pairwise_heading_difference_rad": 2.0}
    assert regime_trajectory_soft_score(
        divergent_headings, "dense_shared", saturation, heading_prior_weights=weights,
    ) == regime_trajectory_soft_score(
        aligned, "dense_shared", saturation, heading_prior_weights=weights,
    )


def test_complementary_hybrids_are_bounded_deterministic_and_separated():
    robot_ids = ("robot_00", "robot_01", "robot_02")

    def trajectory(robot_id, y, bend):
        poses = np.repeat(np.eye(4)[None], 5, axis=0)
        poses[:, 0, 3] = np.linspace(0.0, 1.0, 5)
        poses[:, 1, 3] = y + bend * np.asarray([0.0, 0.5, 1.0, 0.5, 0.0])
        delta = np.gradient(poses[:, :2, 3], axis=0)
        yaw = np.arctan2(delta[:, 1], delta[:, 0])
        poses[:, 0, 0] = np.cos(yaw)
        poses[:, 0, 1] = -np.sin(yaw)
        poses[:, 1, 0] = np.sin(yaw)
        poses[:, 1, 1] = np.cos(yaw)
        return SimpleNamespace(
            robot_id=robot_id,
            base_to_world=poses,
            path_family="one_waypoint",
            metadata={"initial_heading_prior_error_rad": abs(float(yaw[0]))},
        )

    first_set = tuple(
        trajectory(robot_id, 2.0 * index, 0.10)
        for index, robot_id in enumerate(robot_ids)
    )
    second_set = tuple(
        trajectory(robot_id, 2.0 * index, -0.10)
        for index, robot_id in enumerate(robot_ids)
    )
    candidates = (
        (first_set, {"observation_regime": "dense_shared"}),
        (second_set, {"observation_regime": "dense_shared"}),
    )
    failures = (
        {
            "candidate_rank": 0,
            "reason": "trajectory_temporal_overlap_failed",
            "details": {"union_edges": [["robot_00", "robot_02"]]},
        },
        {
            "candidate_rank": 1,
            "reason": "trajectory_temporal_overlap_failed",
            "details": {"union_edges": [["robot_00", "robot_01"]]},
        },
    )
    kwargs = {
        "minimum_pairwise_distance_m": 0.6,
        "minimum_waypoint_trajectories": 1,
        "formation_degeneracy_limits": {
            "minimum_mean_heading_difference_deg": 8.0,
            "maximum_mean_path_similarity": 0.96,
            "minimum_spatial_coverage_m2": 1.0,
        },
        "coverage_saturation_m2": {"dense_shared": 8.0},
        "heading_prior_weights": {"dense_shared": 1.0},
        "maximum_candidates": 4,
    }
    first = complementary_hybrid_trajectory_sets(candidates, failures, **kwargs)
    second = complementary_hybrid_trajectory_sets(candidates, failures, **kwargs)
    assert 0 < len(first) <= 4
    assert [item[1] for item in first] == [item[1] for item in second]
    for _, source_by_robot, metrics in first:
        assert set(source_by_robot.values()) == {0, 1}
        assert metrics["minimum_inter_robot_distance_m"] >= 0.6
        assert metrics["complementary_hybrid"]["source_union_edges"] == [
            ["robot_00", "robot_01"],
            ["robot_00", "robot_02"],
        ]

    bounded = complementary_hybrid_trajectory_sets(
        candidates, failures, **{**kwargs, "maximum_candidates": 2}
    )
    preserved = {
        tuple(tuple(edge) for edge in metrics["complementary_hybrid"][
            "predicted_preserved_edges"
        ])
        for _, _, metrics in bounded
    }
    assert preserved == {
        (("robot_00", "robot_01"),),
        (("robot_00", "robot_02"),),
    }

    single_edge_failures = (
        {
            "candidate_rank": 0,
            "reason": "trajectory_temporal_overlap_failed",
            "details": {"union_edges": [["robot_00", "robot_02"]]},
        },
        {
            "candidate_rank": 1,
            "reason": "trajectory_temporal_overlap_failed",
            "details": {"union_edges": []},
        },
    )
    bridges = complementary_hybrid_trajectory_sets(
        candidates, single_edge_failures, **kwargs
    )
    assert bridges
    _, bridge_sources, bridge_metrics = bridges[0]
    assert bridge_sources == {
        "robot_00": 0, "robot_01": 1, "robot_02": 0,
    }
    assert bridge_metrics["complementary_hybrid"]["predicted_preserved_edges"] == [
        ["robot_00", "robot_02"]
    ]
    assert bridge_metrics["complementary_hybrid"]["strategy"] == "measured_edge_bridge"



def test_dataset_diagnostics_aggregate_finalized_training_semantics(tmp_path):
    import yaml

    from multi_view_world_dataset.dataset_diagnostics import (
        summarize_generated_dataset,
    )
    from multi_view_world_dataset.utils.config import load_yaml_config

    config = load_yaml_config("configs/default.yaml")
    (tmp_path / "dataset_meta.json").write_text(
        json.dumps({"schema_version": "1.1.0"}), encoding="utf-8"
    )
    (tmp_path / "resolved_config.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    (tmp_path / "taxonomy.json").write_text(json.dumps({
        "instance_catalogs": {
            "Rs_int": [{
                "object_state_id": "target",
                "public_instance_id": 4,
            }]
        }
    }), encoding="utf-8")

    episode = (
        tmp_path / "episodes" / "Rs_int" / "config_000" / "episode_000"
    )
    episode.mkdir(parents=True)
    (episode / "meta.json").write_text(
        json.dumps({"episode_id": "episode_000"}), encoding="utf-8"
    )
    robots = {
        f"robot_{index:02d}": {
            "path_family": "one_waypoint",
            "arc_path_length_m": 1.4 + index * 0.1,
            "start_end_displacement_m": 1.2,
            "tortuosity": 1.2,
            "cumulative_absolute_yaw_change_rad": 0.5,
        }
        for index in range(3)
    }
    keyframes = [
        {
            "edges": [["robot_00", "robot_01"]],
            "overlaps": {
                "robot_00|robot_01": 0.3,
                "robot_00|robot_02": 0.0,
                "robot_01|robot_02": 0.0,
            },
        },
        {
            "edges": [["robot_01", "robot_02"]],
            "overlaps": {
                "robot_00|robot_01": 0.0,
                "robot_00|robot_02": 0.0,
                "robot_01|robot_02": 0.4,
            },
        },
    ]
    metrics = {
        "trajectory": {
            "start_region_ids": ["kitchen_0", "hall_0", "bedroom_0"],
            "traversed_region_ids": {
                "robot_00": ["kitchen_0"],
                "robot_01": ["hall_0"],
                "robot_02": ["bedroom_0"],
            },
            "requested_observation_regime": "partial_chain",
            "observation_regime": "partial_chain",
            "robots": robots,
            "joint_diversity": {
                "spatial_coverage_bbox_area_m2": 12.0,
                "mean_pairwise_heading_difference_rad": 1.0,
                "mean_path_direction_similarity": 0.1,
                "minimum_inter_robot_distance_m": 1.0,
                "maximum_inter_robot_distance_m": 5.0,
            },
            "temporal_overlap": {
                "connected_fraction": 0.0,
                "checks": {"union_graph_connected": True},
                "keyframes": keyframes,
            },
        },
        "intervention_visibility": {
            "objects": {
                "target": {
                    "participating_robot_count": 2,
                    "qualifying_frame_count": 8,
                    "maximum_pixels": 500,
                }
            }
        },
        "post_render_intervention_effect": {
            "changed_pixels": 200,
            "mean_rgb_delta": 9.0,
        },
        "before": {
            "maximum_capture_translation_error_m": 0.001,
            "maximum_capture_rotation_error_rad": 0.002,
            "minimum_multimodal_alignment_fraction": 0.9,
        },
    }
    (episode / "generation_metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    (episode / "events.json").write_text(json.dumps([{
        "intervention_type": "rigid_relocation",
        "target_instance_id": "target",
        "before_object_state": {
            "room_id": "kitchen_0",
            "category": "chair",
        },
    }]), encoding="utf-8")
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    np.savez(
        episode / "trajectories.npz",
        robot_00_base_to_world=poses,
        robot_01_base_to_world=poses + np.asarray([0, 0, 0, 1]),
        robot_02_base_to_world=poses + np.asarray([0, 0, 0, 2]),
    )
    camera = {
        "camera_to_world": np.eye(4).tolist(),
        "world_to_camera": np.eye(4).tolist(),
        "pixel_intrinsics": np.eye(3).tolist(),
        "geometry_pixel_intrinsics": np.eye(3).tolist(),
        "modality_camera_to_world": {},
    }
    (episode / "observations_before.json").write_text(
        json.dumps([{"camera": camera}]), encoding="utf-8"
    )
    (episode / "bev").mkdir()
    np.savez(
        episode / "bev" / "world_before.npz",
        calibration_world_bounds=np.zeros(4),
        calibration_pixel_to_world=np.eye(3),
        calibration_world_to_pixel=np.eye(3),
        calibration_meters_per_pixel=np.asarray(0.04),
        calibration_floor_z=np.asarray(0.0),
    )

    configuration = (
        tmp_path / "configurations" / "Rs_int" / "config_000"
    )
    configuration.mkdir(parents=True)
    (configuration / "config_meta.json").write_text(json.dumps({
        "metadata": {"changed_instance_ids": ["target", "other"]},
        "world_state": {"objects": [
            {
                "instance_id": "target",
                "category": "chair",
                "room_id": "kitchen_0",
            },
            {
                "instance_id": "other",
                "category": "lamp",
                "room_id": "bedroom_0",
            },
        ]},
    }), encoding="utf-8")

    output, report = summarize_generated_dataset(tmp_path)
    assert output.is_file()
    assert report["finalized_episode_count"] == 1
    assert report["overlap"]["union_connected_episode_fraction"] == 1.0
    assert report["intervention"]["visible_robot_count"]["mean"] == 2.0
    assert report["configuration"]["changed_object_count"]["mean"] == 2.0
    assert report["identity_and_calibration"][
        "complete_observation_metadata_episode_count"
    ] == 1
    assert report["identity_and_calibration"][
        "complete_world_bev_calibration_episode_count"
    ] == 1
    assert report["collapse_warning_evaluation_deferred"]["episodes"]

def test_camera_view_proxy_rewards_shared_scene_content_not_parallel_headings():
    angles = np.deg2rad([0.0, 120.0, 240.0])
    positions = np.column_stack((3.0 * np.cos(angles), 3.0 * np.sin(angles)))

    def trajectories(outward: bool):
        result = []
        for index, position in enumerate(positions):
            forward = position / np.linalg.norm(position)
            if not outward:
                forward = -forward
            base = np.repeat(np.eye(4)[None], 5, axis=0)
            base[:, :2, 3] = position
            camera = base.copy()
            camera[:, 0, 2] = forward[0]
            camera[:, 1, 2] = forward[1]
            camera[:, 2, 2] = 0.0
            result.append(SimpleNamespace(
                robot_id=f"robot_{index:02d}",
                base_to_world=base,
                camera_to_world=camera,
                metadata={},
            ))
        return tuple(result)

    converging = joint_trajectory_metrics(trajectories(outward=False))
    looking_away = joint_trajectory_metrics(trajectories(outward=True))
    # The converging camera directions are 120 degrees apart, not parallel,
    # yet most of their sampled view-cone volumes intersect around the target.
    assert converging["temporal_camera_view_connectivity_proxy"] > 0.70
    assert looking_away["temporal_camera_view_connectivity_proxy"] < 0.01

    saturation = {
        "dense_shared": 8.0,
        "partial_chain": 16.0,
        "exploratory": 30.0,
    }
    shared = {
        "spatial_coverage_bbox_area_m2": 8.0,
        "mean_inter_robot_distance_m": 3.0,
        "temporal_camera_view_connectivity_proxy": 1.0,
    }
    sparse = {**shared, "temporal_camera_view_connectivity_proxy": 0.0}
    weights = {"dense_shared": 2.0}
    assert regime_trajectory_soft_score(
        shared, "dense_shared", saturation, view_connectivity_weights=weights,
    ) > regime_trajectory_soft_score(
        sparse, "dense_shared", saturation, view_connectivity_weights=weights,
    )

def test_realized_regime_soft_score_prefers_running_deficit_without_hard_gate():
    weights = {"dense_shared": 0.30, "partial_chain": 0.50, "exploratory": 0.20}
    global_counts = Counter({"dense_shared": 6, "partial_chain": 4})
    split_counts = Counter({"dense_shared": 3, "partial_chain": 2})
    metrics = {"cheap_scene_visibility": {"score": 1.0}}
    exploratory = _gt_valid_candidate_soft_score(
        metrics, 0, 3, "exploratory", weights, global_counts, split_counts,
        lambda_regime=0.75,
    )
    dense = _gt_valid_candidate_soft_score(
        metrics, 0, 3, "dense_shared", weights, global_counts, split_counts,
        lambda_regime=0.75,
    )
    assert exploratory["realized_regime_deficit_preference"] > 0.0
    assert exploratory["final_score"] > dense["final_score"]
    feasible_only = _gt_valid_candidate_soft_score(
        metrics, 2, 3, "dense_shared", weights, global_counts, split_counts,
        lambda_regime=0.75,
    )
    assert np.isfinite(feasible_only["final_score"])


def test_robot_state_serializes_mast_extension_consistently():
    config = {
        "robot": {
            "use_final_robot": True,
            "final_model": "mobile_sensor_robot_v1",
            "development_model": "turtlebot",
        },
        "camera": {"heights_m": [0.8, 1.0, 1.2, 1.4]},
    }
    robot_ids = ("robot_00", "robot_01", "robot_02")
    heights = {"robot_00": 0.8, "robot_01": 1.0, "robot_02": 1.4}
    trajectories = tuple(
        SimpleNamespace(robot_id=robot_id, base_to_world=np.eye(4)[None])
        for robot_id in robot_ids
    )
    states = _robot_states(config, heights, trajectories)
    assert all(isinstance(state, RobotState) for state in states)
    assert [state.mast_joint_value_m for state in states] == pytest.approx(
        [0.0, 0.2, 0.6]
    )
    assert all(
        state.camera_height_m == pytest.approx(0.8 + state.mast_joint_value_m)
        for state in states
    )


def test_semantic_hash_collision_raises_and_writer_persists_context(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(label_module, "stable_semantic_id", lambda category: 99)
    with pytest.raises(SemanticIDCollisionError, match="chair.*table"):
        validate_semantic_id_uniqueness(["chair", "table"])

    writer = DatasetWriter(tmp_path / "dataset")
    writer.initialize({"schema_version": "1.1.0"})
    catalog = (
        _object("chair_a", "/World/chair_a", "chair"),
        _object("table_a", "/World/table_a", "table"),
    )
    with pytest.raises(ConfigurationError, match="semantic ID collision"):
        writer.update_scene_taxonomy("scene", catalog)
    diagnostic = json.loads(
        (writer.root / "semantic_id_collision.json").read_text(encoding="utf-8")
    )
    assert diagnostic["semantic_id"] == 99
    assert diagnostic["category_names"] == ["chair", "table"]
    assert diagnostic["scene_id"] == "scene"


def test_legacy_sampler_entry_points_emit_deprecation_warnings():
    adapter = object.__new__(OmniGibsonAdapter)
    adapter._started = False
    with pytest.warns(DeprecationWarning, match="legacy compatibility"):
        with pytest.raises(SimulatorUnavailableError):
            adapter.place_development_robots(1)

    observations = {
        "robot_00": {
            "base_to_world": np.eye(4),
            "camera_to_base": np.eye(4),
        }
    }
    with pytest.warns(DeprecationWarning, match="legacy compatibility"):
        with pytest.raises(AttributeError):
            adapter.sample_robot_trajectories(
                1, observations_override=observations
            )

    adapter.config = {
        "trajectory": {
            "trajectory_sets_per_placement": 0,
        }
    }
    adapter.robot_observations = lambda: {}
    with pytest.warns(DeprecationWarning, match="legacy compatibility"):
        with pytest.raises(SampleRejected, match="trajectory_set_pool_exhausted"):
            adapter.sample_robot_trajectory_sets(1)


def test_robot_appearance_summary_writes_three_variant_panel(tmp_path):
    rgb = np.full((64, 96, 3), 180, dtype=np.uint8)
    instances = np.zeros((64, 96), dtype=np.int32)
    instances[8:24, 8:24] = 1
    instances[24:40, 36:52] = 2
    instances[40:56, 68:84] = 3
    rgb[instances == 1] = (242, 79, 14)
    rgb[instances == 2] = (10, 87, 235)
    rgb[instances == 3] = (13, 158, 64)

    output = tmp_path / "robot_appearance_summary.png"
    save_robot_appearance_summary(output, rgb, instances)

    assert output.is_file()
    from PIL import Image

    with Image.open(output) as image:
        assert image.mode == "RGB"
        assert image.size == (960, 540)
