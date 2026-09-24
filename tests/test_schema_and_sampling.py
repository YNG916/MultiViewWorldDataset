from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from multi_view_world_dataset.adapters.omnigibson import (
    OmniGibsonAdapter,
    _points_inside_floor_support,
    _resampled_candidate_indices,
    _restore_world_to_map_batch_order,
)
from multi_view_world_dataset.errors import SampleRejected
from multi_view_world_dataset.sampling.configurations import (
    exact_state_hash,
    near_duplicate_configuration,
    nonrigid_configuration_candidates,
)
from multi_view_world_dataset.sampling.placement import (
    select_consensus_local_headings,
    select_local_traversable_heading,
    select_shared_traversable_heading,
    soft_anchor_candidate_order,
)
from multi_view_world_dataset.sampling.interventions import (
    eligible_intervention_targets,
    propose_articulation,
    propose_state_change,
)
from multi_view_world_dataset.sampling.splits import assign_scene_family_splits, infer_scene_family
from multi_view_world_dataset.sampling.trajectories import (
    _angular_density_balanced_weights,
    _sample_route_controls,
    collision_safe_planner_polyline,
    densify_polyline,
    lane_preserving_guides,
    minimum_separation_event,
    sample_geodesic_robot_trajectory_pool,
    sample_geodesic_trajectory_set,
    smooth_collision_safe_path,
    trajectories_equal,
    trajectory_from_spatial_path,
    trajectory_kinematic_metrics,
)
from multi_view_world_dataset.schema.records import (
    ApplicationMode,
    InterventionEvent,
    InterventionType,
    ObjectState,
)


def make_object(instance_id="obj_a", x=0.0):
    transform = np.eye(4)
    transform[0, 3] = x
    return ObjectState(
        instance_id=instance_id,
        asset_uid="asset",
        category="chair",
        native_path="/World/chair",
        structural=False,
        movable=True,
        articulated=False,
        available_states=(),
        object_to_world=transform,
        bbox_min_world=(-0.5, -0.5, 0),
        bbox_max_world=(0.5, 0.5, 1),
        scale=(1, 1, 1),
    )


def _straight_planner(start, goal):
    path = np.asarray([start, goal], dtype=float)
    return path, float(np.linalg.norm(goal - start))


def test_intervention_proposal_budget_resamples_single_visible_target():
    first = list(_resampled_candidate_indices(np.random.default_rng(11), 1, 8))
    second = list(_resampled_candidate_indices(np.random.default_rng(11), 1, 8))
    assert first == [0] * 8
    assert second == first


def test_intervention_proposal_budget_cycles_candidates_fairly():
    sampled = list(_resampled_candidate_indices(np.random.default_rng(19), 3, 8))
    assert len(sampled) == 8
    assert set(sampled[:3]) == {0, 1, 2}
    assert set(sampled[3:6]) == {0, 1, 2}


def test_floor_support_filter_rejects_exterior_traversability_pixels():
    points = np.asarray([
        [0.5, 0.5],
        [2.02, 0.5],
        [-0.2, 0.5],
        [3.0, 3.0],
    ])
    bounds = np.asarray([
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 2.0, 1.0],
    ])
    supported = _points_inside_floor_support(
        points,
        bounds,
        tolerance_m=0.05,
    )
    assert supported.tolist() == [True, True, False, False]


def test_smoothing_can_fall_back_to_dense_collision_safe_polyline():
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])

    def only_accept_original_segments(samples):
        values = np.asarray(samples)
        return bool(np.all(
            (np.abs(values[:, 1]) < 1e-9)
            | (np.abs(values[:, 0] - 1.0) < 1e-9)
        ))

    result = smooth_collision_safe_path(
        points, only_accept_original_segments,
        smoothing_strengths=(1.0, 0.5, 0.0), validation_spacing_m=0.05,
    )
    assert result is not None
    curve, strength = result
    assert strength == 0.0
    assert only_accept_original_segments(curve)


def test_omnigibson_world_to_map_batch_reversal_is_repaired():
    expected = np.asarray([[10, 20], [30, 40], [50, 60]])
    restored, repaired = _restore_world_to_map_batch_order(
        expected[::-1], expected[0], expected[-1]
    )
    assert repaired is True
    assert np.array_equal(restored, expected)

    unchanged, repaired = _restore_world_to_map_batch_order(
        expected, expected[0], expected[-1]
    )
    assert repaired is False
    assert np.array_equal(unchanged, expected)


def test_final_robot_capture_uses_only_raw_aov_for_segmentation():
    adapter = object.__new__(OmniGibsonAdapter)
    adapter.config = {
        "bev": {
            "modalities": ["rgb", "depth_linear", "semantic", "instance"],
            "world_modalities": ["normal", "instance_id"],
        }
    }
    adapter._using_final_robot = True
    assert adapter._configured_bev_sensor_names() == [
        "depth_linear",
        "normal",
        "rgb",
    ]

    adapter._using_final_robot = False
    assert {
        "seg_semantic",
        "seg_instance",
        "seg_instance_id",
    }.issubset(adapter._configured_bev_sensor_names())


def _sample_parallel_trajectories(
    seed, minimum_waypoint_trajectories=0,
    maximum_joint_valid_candidates=None,
):
    starts = {}
    mounts = {}
    candidates = []
    for index, y in enumerate((0.0, 2.0, 4.0)):
        robot_id = f"robot_{index:02d}"
        starts[robot_id] = np.array(
            [[1, 0, 0, 0], [0, 1, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
        )
        mounts[robot_id] = np.eye(4)
        candidates.append([1.0, y])
    validator = lambda xy: bool(np.all((xy[:, 0] >= 0.0) & (xy[:, 0] <= 1.0)))
    return sample_geodesic_trajectory_set(
        starts,
        mounts,
        np.asarray(candidates),
        0.0,
        np.random.default_rng(seed),
        frames=60,
        fps=10,
        path_length_range_m=(0.99, 1.01),
        minimum_pairwise_distance_m=0.6,
        maximum_linear_speed_mps=0.8,
        maximum_angular_speed_radps=1.2,
        maximum_acceleration_mps2=1.5,
        plan_segment=_straight_planner,
        is_path_traversable=validator,
        path_family_weights={"direct": 1.0, "one_waypoint": 0.0, "two_waypoint": 0.0},
        minimum_waypoint_trajectories=minimum_waypoint_trajectories,
        initial_heading_tolerance_rad=np.deg2rad(10.0),
        line_validation_spacing_m=0.02,
        maximum_joint_valid_candidates=maximum_joint_valid_candidates,
        smoothing_validation_spacing_m=0.02,
        smoothing_strengths=(1.0, 0.5),
        candidate_pool_size=2,
        maximum_attempts=10,
        joint_pool_rounds=2,
    )


def test_soft_anchor_order_is_deterministic_without_a_hard_radius():
    candidates = np.column_stack((np.arange(8.0), np.zeros(8)))
    first = soft_anchor_candidate_order(
        np.arange(8), candidates, np.zeros(2), 1.5, np.random.default_rng(19),
    )
    second = soft_anchor_candidate_order(
        np.arange(8), candidates, np.zeros(2), 1.5, np.random.default_rng(19),
    )
    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == list(range(8))
    assert 7 in first

def test_initial_heading_uses_a_directly_traversable_local_exit():
    candidates = np.asarray([[0.5, 0.0], [0.0, 0.5], [0.4, 0.4]])

    def validator(points):
        # A wall blocks the desired +X ray, so use a collision-free local tangent.
        values = np.asarray(points)
        return not bool(np.any((values[:, 0] > 0.2) & (values[:, 1] < 0.1)))

    yaw, error = select_local_traversable_heading(
        np.zeros(2), candidates, 0.0, validator,
        minimum_probe_m=0.1, maximum_probe_m=0.75,
        validation_spacing_m=0.02,
    )
    assert yaw == pytest.approx(np.pi / 4.0)
    assert error == pytest.approx(np.pi / 4.0)



def test_shared_heading_requires_one_traversable_exit_for_every_robot():
    sources = np.asarray([[0.0, 0.0], [0.0, 1.0]])

    def validator(points):
        values = np.asarray(points)
        # +X is blocked only for the upper robot; +Y is safe for both.
        return not bool(np.any(
            (values[:, 0] > 0.2)
            & (values[:, 1] > 0.5)
            & (values[:, 1] < 1.5)
        ))

    yaw, error = select_shared_traversable_heading(
        sources,
        0.0,
        validator,
        probe_distance_m=0.5,
        validation_spacing_m=0.02,
        angular_step_rad=np.pi / 2.0,
    )
    assert yaw == pytest.approx(np.pi / 2.0)
    assert error == pytest.approx(np.pi / 2.0)


def test_local_heading_consensus_searches_feasible_exit_sets():
    sources = np.asarray([[0.0, 0.0], [2.0, 0.0]])
    pools = [
        np.asarray([[0.5, 0.0]]),
        np.asarray([[2.0, 0.5]]),
    ]
    headings, errors, consensus = select_consensus_local_headings(
        sources,
        pools,
        0.0,
        lambda points: True,
        minimum_probe_m=0.1,
        maximum_probe_m=0.75,
        validation_spacing_m=0.02,
        maximum_deviation_rad=np.deg2rad(50.0),
        angular_step_rad=np.deg2rad(45.0),
    )
    assert np.allclose(headings, [0.0, np.pi / 2.0])
    assert max(errors) == pytest.approx(np.pi / 4.0)
    assert consensus == pytest.approx(np.pi / 4.0)


def test_joint_sampler_accepts_per_robot_reachable_components():
    starts = {}
    mounts = {}
    reachable = {}
    for index, y in enumerate((0.0, 2.0, 4.0)):
        robot_id = f"robot_{index:02d}"
        starts[robot_id] = np.array(
            [[1, 0, 0, 0], [0, 1, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        )
        mounts[robot_id] = np.eye(4)
        reachable[robot_id] = np.asarray([[1.0, y]])

    def same_component_planner(start, goal):
        if abs(float(start[1] - goal[1])) > 1e-9:
            return None
        return _straight_planner(start, goal)

    trajectories = sample_geodesic_trajectory_set(
        starts, mounts, reachable, 0.0, np.random.default_rng(17),
        frames=60, fps=10, path_length_range_m=(0.99, 1.01),
        minimum_pairwise_distance_m=0.6, maximum_linear_speed_mps=0.8,
        maximum_angular_speed_radps=1.2, maximum_acceleration_mps2=1.5,
        plan_segment=same_component_planner,
        is_path_traversable=lambda points: True,
        path_family_weights={"direct": 1.0, "one_waypoint": 0.0, "two_waypoint": 0.0},
        minimum_waypoint_trajectories=0, initial_heading_tolerance_rad=np.pi,
        line_validation_spacing_m=0.02, smoothing_validation_spacing_m=0.02,
        smoothing_strengths=(1.0,), candidate_pool_size=1,
        maximum_attempts=3, joint_pool_rounds=1,
    )
    assert len(trajectories) == 3
    assert [item.base_to_world[-1, 1, 3] for item in trajectories] == [0.0, 2.0, 4.0]


def test_joint_sampler_caps_expensive_valid_combination_scoring():
    trajectories = _sample_parallel_trajectories(
        17, maximum_joint_valid_candidates=1,
    )
    for trajectory in trajectories:
        assert trajectory.metadata["joint_valid_candidate_count"] == 1
        assert trajectory.metadata["joint_valid_candidate_limit_reached"] is True
        assert trajectory.metadata["joint_combination_evaluated_count"] >= 1


def test_joint_set_treats_configured_waypoint_minimum_as_deprecated_diagnostic():
    trajectories = _sample_parallel_trajectories(
        17, minimum_waypoint_trajectories=1
    )
    assert len(trajectories) == 3
    assert all(item.path_family == "direct" for item in trajectories)


def test_waypoint_controls_reject_untrackable_direction_reversal():
    candidates = np.asarray([
        [1.0, 0.0],
        [0.5, np.sqrt(3.0) / 2.0],
        [1.0 + np.cos(np.deg2rad(40.0)), np.sin(np.deg2rad(40.0))],
    ])
    controls = _sample_route_controls(
        np.zeros(2), 0.0, candidates, "one_waypoint", 1.0, 2.1,
        np.deg2rad(1.0), np.deg2rad(55.0), np.random.default_rng(7),
    )
    assert controls is not None
    assert np.allclose(controls[1], [1.0, 0.0])
    assert np.allclose(controls[2], candidates[2])


def test_waypoint_bearing_is_soft_when_geodesic_must_turn_first():
    candidates = np.asarray([[0.0, 1.0], [-1.0, 0.0]])
    controls = _sample_route_controls(
        np.zeros(2), 0.0, candidates, "direct", 1.0, 1.1,
        np.deg2rad(10.0), np.deg2rad(55.0), np.random.default_rng(2),
    )
    # Neither far-goal bearing matches +X; planning is still attempted.
    assert controls is not None
    assert controls.shape == (2, 2)


def test_trajectory_tangent_policy_keeps_non_prior_directions_eligible():
    candidates = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    hard = _sample_route_controls(
        np.zeros(2), 0.0, candidates, "direct", 1.0, 1.1,
        np.deg2rad(10.0), np.deg2rad(55.0), np.random.default_rng(2),
    )
    assert hard is not None
    assert np.allclose(hard[-1], [1.0, 0.0])

    sampled_endpoints = {
        tuple(_sample_route_controls(
            np.zeros(2), 0.0, candidates, "direct", 1.0, 1.1,
            np.deg2rad(10.0), np.deg2rad(55.0), np.random.default_rng(seed),
            soft_initial_heading=True,
            initial_heading_probability_floor=1.0,
        )[-1])
        for seed in range(12)
    }
    assert sampled_endpoints == {(1.0, 0.0), (0.0, 1.0)}


def test_tangent_policy_still_enforces_sampled_initial_heading_tolerance():
    start = np.eye(4)
    with pytest.raises(SampleRejected, match="trajectory_no_geodesic_path"):
        sample_geodesic_robot_trajectory_pool(
            "robot_00",
            start,
            np.eye(4),
            np.asarray([[0.0, 1.0]]),
            0.0,
            np.random.default_rng(5),
            frames=60,
            fps=10,
            path_length_range_m=(0.99, 1.01),
            maximum_linear_speed_mps=0.8,
            maximum_angular_speed_radps=1.2,
            maximum_acceleration_mps2=1.5,
            plan_segment=_straight_planner,
            is_path_traversable=lambda points: True,
            path_family_weights={
                "direct": 1.0, "one_waypoint": 0.0, "two_waypoint": 0.0,
            },
            initial_heading_tolerance_rad=np.deg2rad(10.0),
            derive_initial_heading_from_tangent=True,
            initial_heading_probability_floor=1.0,
            maximum_control_turn_rad=np.deg2rad(55.0),
            line_validation_spacing_m=0.02,
            smoothing_validation_spacing_m=0.02,
            smoothing_strengths=(1.0,), candidate_pool_size=1, maximum_attempts=2,
        )



def test_soft_direction_weights_are_not_dominated_by_pixel_count():
    bearings = np.concatenate((np.zeros(1000), [np.pi / 2.0]))
    indices = np.arange(len(bearings), dtype=np.int64)
    balanced = _angular_density_balanced_weights(
        bearings, indices, np.ones(len(indices))
    )
    assert np.isclose(balanced[:-1].sum(), balanced[-1])
    assert np.isclose(balanced.sum(), 2.0)


def test_state_hash_is_order_independent_and_near_duplicate():
    a, b = make_object("a"), make_object("b", 1)
    assert exact_state_hash((a, b)) == exact_state_hash((b, a))
    shifted = replace(a, object_to_world=np.array([[1, 0, 0, 0.01], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    assert near_duplicate_configuration(
        (shifted, b), (a, b), translation_threshold_m=0.03, rotation_threshold_deg=3
    )


def test_nonrigid_configuration_dedup_considers_fixed_base_state_and_joint():
    fixed = replace(make_object("fixed"), movable=False, semantic_states={"Frozen": False})
    frozen = replace(fixed, semantic_states={"Frozen": True})
    options = {"translation_threshold_m": 0.03, "rotation_threshold_deg": 3}
    assert near_duplicate_configuration((frozen,), (fixed,), **options)
    assert not near_duplicate_configuration(
        (frozen,), (fixed,), include_nonrigid=True, **options
    )

    articulated = replace(
        fixed, articulated=True, semantic_states={}, joint_names=("joint",),
        joint_limits=((0.0, 1.0),), joint_values=(0.0,),
    )
    passive_noise = replace(articulated, joint_values=(1.0e-12,))
    real_change = replace(articulated, joint_values=(0.5,))
    assert near_duplicate_configuration(
        (passive_noise,), (articulated,), include_nonrigid=True, **options
    )
    assert not near_duplicate_configuration(
        (real_change,), (articulated,), include_nonrigid=True, **options
    )


def test_nonfinite_joint_value_is_rejected_before_hash_serialization():
    bad = replace(
        make_object("bad"),
        articulated=True,
        joint_names=("joint",),
        joint_limits=((0.0, 1.0),),
        joint_values=(float("nan"),),
    )
    with pytest.raises(SampleRejected, match="nonfinite_configuration_joint_values"):
        exact_state_hash((bad,))


def test_nonrigid_configuration_changes_two_real_fixed_base_targets():
    base = tuple(
        replace(
            make_object(f"fixed_{index}"),
            movable=False,
            articulated=True,
            joint_names=("joint",),
            joint_limits=((0.0, 1.0),),
            joint_values=(0.0,),
        )
        for index in range(2)
    )
    base += (
        replace(
            make_object("passive"),
            structural=True,
            movable=False,
            articulated=True,
            joint_names=("joint",),
            joint_limits=((0.0, 1.0),),
            joint_values=(0.0,),
        ),
    )
    assert len(nonrigid_configuration_candidates(base)) == 2

    class FakeAdapter:
        config = {
            "configuration_sampling": {"nonrigid_changed_objects": 2},
            "generation": {
                "exact_hash_decimals": 8,
                "snapshot_restore_tolerance": 1.0e-5,
            },
        }

        def __init__(self):
            self.values = np.zeros(len(base), dtype=np.float64)

        def dump_snapshot(self):
            return self.values.copy()

        def load_snapshot(self, snapshot):
            self.values = np.asarray(snapshot, dtype=np.float64).copy()

        def object_catalog_with_relations(self):
            return tuple(
                replace(
                    obj,
                    joint_values=(
                        float(self.values[index] + (1.0e-12 if obj.structural else 0.0)),
                    ),
                )
                for index, obj in enumerate(base)
            )

        def apply_atomic_intervention(self, seed, *, forced_type, visible_target_ids):
            assert forced_type is InterventionType.ARTICULATION
            target_id = visible_target_ids[0]
            index = int(target_id.split("_")[-1])
            self.values[index] = 1.0
            return {
                "checks": {"physically_valid": True},
                "changed_instance_ids": [target_id],
                "snapshot": self.dump_snapshot(),
                "event": SimpleNamespace(
                    intervention_type=forced_type,
                    parameters={"joint_index": 0, "value_after": 1.0},
                ),
            }

        def refresh_collision_geometry_cache(self):
            pass

        def _catalog_restore_metrics(self, expected, restored):
            return 0.0, all(
                left.joint_values == right.joint_values
                for left, right in zip(expected, restored, strict=True)
            )

    adapter = FakeAdapter()
    first = OmniGibsonAdapter._randomize_nonrigid_configuration(
        adapter, 91, baseline=base, original_snapshot=adapter.dump_snapshot(),
    )
    assert first["configuration_family"] == "nonrigid_fallback"
    assert first["changed_object_count"] == 2
    assert set(first["changed_instance_ids"]) == {"fixed_0", "fixed_1"}
    assert all(x["change_type"] == "articulation" for x in first["changes"])
    assert first["exact_state_hash"] != first["baseline_exact_state_hash"]
    second_adapter = FakeAdapter()
    second = OmniGibsonAdapter._randomize_nonrigid_configuration(
        second_adapter, 91, baseline=base,
        original_snapshot=second_adapter.dump_snapshot(),
    )
    assert second["exact_state_hash"] == first["exact_state_hash"]
    assert second["changes"] == first["changes"]



def test_scene_family_splits_are_disjoint():
    scenes = ["House_0_int", "House_1_int", "Other_0_garden", "school_lab"]
    assert infer_scene_family("House_1_int") == "House"
    splits = assign_scene_family_splits(scenes, {"train": 0.5, "val": 0.25, "test": 0.25}, seed=3)
    assert splits["House_0_int"] == splits["House_1_int"]


def test_diagonal_planner_step_is_repaired_without_cutting_obstacle():
    def validator(points):
        values = np.asarray(points)
        inside_obstacle = (
            (values[:, 0] > 0.25)
            & (values[:, 0] < 0.75)
            & (values[:, 1] > 0.25)
            & (values[:, 1] < 0.75)
        )
        return not bool(np.any(inside_obstacle))

    repaired = collision_safe_planner_polyline(
        np.asarray([[0.0, 0.0], [1.0, 1.0]]),
        validator,
        validation_spacing_m=0.05,
    )
    assert repaired is not None
    assert len(repaired) == 3
    assert validator(densify_polyline(repaired, 0.05))


def test_continuous_smoothing_can_produce_a_curved_path():
    polyline = np.asarray([[0.0, 0.0], [0.8, 0.0], [0.8, 0.8]])
    result = smooth_collision_safe_path(
        polyline,
        lambda points: bool(np.all((points >= -0.2) & (points <= 1.0))),
        smoothing_strengths=(1.0,),
        validation_spacing_m=0.01,
    )
    assert result is not None
    curve, strength = result
    assert strength == 1.0
    assert len(curve) > len(polyline)
    assert np.any(curve[:, 0] > 0.8)


def test_arc_path_length_is_not_start_end_euclidean_displacement():
    path = np.asarray([[0.0, 0.0], [0.8, 0.0], [0.8, 0.8]])
    trajectory = trajectory_from_spatial_path(
        "robot_00", path, 0.0, np.eye(4), frames=80, fps=10
    )
    metrics = trajectory_kinematic_metrics(trajectory)
    assert metrics["arc_path_length_m"] > metrics["start_end_displacement_m"] * 1.3
    assert metrics["tortuosity"] > 1.3


def test_yaw_follows_tangent_without_constant_yaw_lateral_sliding():
    path = np.column_stack((np.linspace(0.0, 1.2, 200), 0.3 * np.sin(np.linspace(0.0, np.pi, 200))))
    trajectory = trajectory_from_spatial_path(
        "robot_00", path, 0.0, np.eye(4), frames=60, fps=10
    )
    positions = trajectory.base_to_world[:, :2, 3]
    steps = np.diff(positions, axis=0)
    step_headings = np.unwrap(np.arctan2(steps[:, 1], steps[:, 0]))
    body_yaws = np.unwrap(np.arctan2(trajectory.base_to_world[:, 1, 0], trajectory.base_to_world[:, 0, 0]))
    assert np.max(np.abs(step_headings - 0.5 * (body_yaws[:-1] + body_yaws[1:]))) < 0.03
    assert np.ptp(body_yaws) > 0.4
    lateral = np.abs(np.sin(step_headings - body_yaws[:-1]))
    assert lateral.max() < 0.04


def test_all_dense_sampled_points_are_traversable():
    path = np.column_stack((np.linspace(0.0, 1.0, 100), 0.15 * np.sin(np.linspace(0.0, np.pi, 100))))
    trajectory = trajectory_from_spatial_path(
        "robot_00", path, 0.0, np.eye(4), frames=60, fps=10
    )
    dense = densify_polyline(trajectory.base_to_world[:, :2, 3], 0.01)
    assert np.all((dense[:, 0] >= 0.0) & (dense[:, 0] <= 1.0))
    assert np.all((dense[:, 1] >= 0.0) & (dense[:, 1] <= 0.16))


def test_geodesic_sampler_holds_kinematic_and_pairwise_limits():
    trajectories = _sample_parallel_trajectories(9)
    assert len(trajectories) == 3
    for trajectory in trajectories:
        metrics = trajectory_kinematic_metrics(trajectory)
        assert metrics["arc_path_length_m"] == pytest.approx(1.0)
        assert metrics["maximum_linear_speed_mps"] <= 0.8
        assert metrics["maximum_angular_speed_radps"] <= 1.2
        assert metrics["maximum_acceleration_mps2"] <= 1.5
    positions = np.stack([item.base_to_world[:, :2, 3] for item in trajectories])
    assert np.min(np.linalg.norm(positions[0] - positions[1], axis=1)) >= 0.6


def test_geodesic_sampling_is_seed_deterministic_and_pairing_is_exact():
    first = _sample_parallel_trajectories(17)
    second = _sample_parallel_trajectories(17)
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left.base_to_world, right.base_to_world)
        assert np.array_equal(left.camera_to_world, right.camera_to_world)
        assert trajectories_equal(left, right, position_atol_m=0.0, matrix_atol=0.0)
    changed = replace(first[0], base_to_world=first[0].base_to_world.copy())
    changed.base_to_world[10, 0, 3] += 0.01
    assert not trajectories_equal(first[0], changed)


def test_future_timed_event_schema_is_compatible():
    target = make_object()
    event = InterventionEvent(
        "event", InterventionType.RIGID_RELOCATION, target.instance_id,
        ApplicationMode.TIMED, 12, {}, target, None,
    )
    assert event.time_index == 12
    with pytest.raises(ValueError):
        InterventionEvent(
            "bad", InterventionType.RIGID_RELOCATION, target.instance_id,
            ApplicationMode.PRE_ROLLOUT, 12, {},
        )


def test_articulation_and_state_interventions_preserve_before_after_records():
    target = replace(
        make_object(),
        articulated=True,
        joint_names=("door_hinge",),
        joint_limits=((0.0, 1.5),),
        joint_values=(0.1,),
        available_states=("ToggledOn",),
        semantic_states={"ToggledOn": False},
    )
    rng = np.random.default_rng(17)
    assert eligible_intervention_targets((target,), InterventionType.ARTICULATION) == (target,)
    articulation = propose_articulation(target, rng)
    assert articulation.before_object_state == target
    assert articulation.after_object_state.joint_values != target.joint_values
    state = propose_state_change(target, rng)
    assert state.parameters["value_before"] is False
    assert state.parameters["value_after"] is True
    assert state.after_object_state.semantic_states["ToggledOn"] is True


def test_state_hash_accepts_unbounded_joint_limits():
    target = replace(
        make_object(),
        articulated=True,
        joint_names=("continuous",),
        joint_limits=((-float("inf"), float("inf")),),
        joint_values=(0.0,),
    )
    assert len(exact_state_hash((target,))) == 64


def test_snapshot_restore_metrics_allow_float32_pose_noise_but_not_state_drift():
    before = make_object()
    transform = before.object_to_world.copy()
    transform[0, 3] += 2.4e-7
    restored = replace(before, object_to_world=transform)
    error, discrete_equal = OmniGibsonAdapter._catalog_restore_metrics(
        (before,),
        (restored,),
    )
    assert error == pytest.approx(2.4e-7)
    assert discrete_equal


def test_adapter_close_does_not_mask_cli_result_with_system_exit():
    class FakeApp:
        def close(self):
            raise SystemExit(0)

    class FakeOG:
        sim = None
        app = FakeApp()

        @staticmethod
        def cleanup():
            return None

    adapter = object.__new__(OmniGibsonAdapter)
    adapter._started = True
    adapter._env = object()
    adapter._og = FakeOG()
    adapter._runtime_findings = {}
    adapter.close()
    assert not adapter._started
    assert adapter._runtime_findings["simulation_app_close_system_exit"] == "0"


def test_route_control_guide_is_soft_deterministic_and_effective():
    candidates = np.asarray([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
    ])

    def endpoint(seed, guide):
        return _sample_route_controls(
            np.zeros(2), 0.0, candidates, "direct", 0.99, 1.01,
            0.2, 1.0, np.random.default_rng(seed),
            soft_initial_heading=True,
            initial_heading_probability_floor=1.0,
            guide_xy=guide,
            guide_soft_scale_m=0.25,
            guide_probability_floor=0.01,
        )[-1]

    target = np.asarray([0.0, 1.0])
    unguided = [endpoint(seed, None) for seed in range(40)]
    guided = [endpoint(seed, target) for seed in range(40)]
    guided_mean = np.mean([np.linalg.norm(point - target) for point in guided])
    unguided_mean = np.mean([np.linalg.norm(point - target) for point in unguided])
    assert guided_mean < 0.1 * unguided_mean
    assert np.array_equal(endpoint(7, target), endpoint(7, target))


def test_route_control_can_reserve_validated_guide_candidate():
    candidates = np.asarray([
        [0.0, 1.0],
        [0.9, 0.0],
        [1.0, 0.0],
        [-1.0, 0.0],
    ])
    target = np.asarray([0.9, 0.0])
    for seed in range(10):
        controls = _sample_route_controls(
            np.zeros(2), 0.0, candidates, "direct", 0.99, 1.01,
            np.pi, np.pi, np.random.default_rng(seed),
            soft_initial_heading=True,
            initial_heading_probability_floor=1.0,
            guide_xy=target,
            prefer_guide=True,
        )
        assert controls is not None
        assert np.array_equal(controls[-1], [1.0, 0.0])

def test_lane_preserving_guides_keep_relative_start_offsets():
    starts = np.asarray([[0.0, 0.0], [1.0, -0.5], [-0.25, 2.0]])
    guides = lane_preserving_guides(starts, np.pi / 2.0, 2.0)
    assert np.allclose(guides - starts, [[0.0, 2.0]] * 3, atol=1.0e-12)
    assert np.allclose(guides[1:] - guides[0], starts[1:] - starts[0])


def test_minimum_separation_event_identifies_crossing_pair_and_frame():
    trajectories = (
        trajectory_from_spatial_path(
            "robot_00", [[-1.0, 0.0], [1.0, 0.0]], 0.0, np.eye(4), frames=61, fps=10
        ),
        trajectory_from_spatial_path(
            "robot_01", [[0.0, -1.0], [0.0, 1.0]], 0.0, np.eye(4), frames=61, fps=10
        ),
        trajectory_from_spatial_path(
            "robot_02", [[5.0, 0.0], [6.0, 0.0]], 0.0, np.eye(4), frames=61, fps=10
        ),
    )
    event = minimum_separation_event(trajectories)
    assert event["distance_m"] == pytest.approx(0.0, abs=1.0e-12)
    assert event["frame_index"] == 30
    assert event["robot_pair"] == ["robot_00", "robot_01"]
    assert event["path_families"] == {
        "robot_00": "unspecified",
        "robot_01": "unspecified",
        "robot_02": "unspecified",
    }


def test_single_robot_rescue_pool_reuses_validated_geodesic_sampler():
    start = np.eye(4)
    yaw = np.pi / 2.0
    start[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    candidates = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    kwargs = dict(
        frames=60, fps=10, path_length_range_m=(0.99, 1.01),
        maximum_linear_speed_mps=0.8, maximum_angular_speed_radps=1.2,
        maximum_acceleration_mps2=1.5, plan_segment=_straight_planner,
        is_path_traversable=lambda points: bool(np.all(np.abs(points) <= 1.01)),
        path_family_weights={
            "direct": 1.0, "one_waypoint": 0.0, "two_waypoint": 0.0,
        },
        initial_heading_tolerance_rad=np.deg2rad(35.0),
        derive_initial_heading_from_tangent=True,
        initial_heading_probability_floor=0.01,
        maximum_control_turn_rad=np.deg2rad(55.0),
        line_validation_spacing_m=0.02,
        smoothing_validation_spacing_m=0.02,
        smoothing_strengths=(1.0,), candidate_pool_size=1, maximum_attempts=10,
        guide_xy=np.asarray([0.0, 1.0]), guide_soft_scale_m=0.1,
        guide_probability_floor=0.01,
    )
    def sample():
        return sample_geodesic_robot_trajectory_pool(
            "robot_01", start, np.eye(4), candidates, 0.0,
            np.random.default_rng(17), **kwargs,
        )[0]
    first, second = sample(), sample()
    assert np.array_equal(first.base_to_world, second.base_to_world)
    assert first.base_to_world[-1, 1, 3] == pytest.approx(1.0)
    assert trajectory_kinematic_metrics(first)["maximum_linear_speed_mps"] <= 0.8


def test_final_robot_renderer_labels_cache_static_scene_mapping():
    adapter = object.__new__(OmniGibsonAdapter)

    class Helpers:
        def __init__(self):
            self.calls = 0

        def get_instance_mappings(self):
            self.calls += 1
            return [
                {
                    "name": "/World/chair/mesh",
                    "semanticLabel": "chair",
                    "instanceIds": [17],
                }
            ]

    helpers = Helpers()
    adapter._syntheticdata_helpers = helpers
    adapter._runtime_findings = {}
    adapter._final_robot_renderer_label_cache = None

    first = adapter._final_robot_renderer_labels()
    second = adapter._final_robot_renderer_labels()

    assert first is second
    assert helpers.calls == 1
    assert first[0]["17"] == "/World/chair/mesh"
    assert adapter._runtime_findings["final_robot_renderer_mapping_cache_hits"] == 1


def test_final_robot_projection_flush_discards_observation_then_renders():
    adapter = object.__new__(OmniGibsonAdapter)
    calls = []
    adapter._using_final_robot = True
    adapter._runtime_findings = {}
    adapter._get_final_robot_capture_observation = lambda camera: calls.append(
        ("observation", camera)
    )
    adapter._og = SimpleNamespace(
        sim=SimpleNamespace(render=lambda: calls.append(("render", None)))
    )
    camera = object()
    adapter._flush_final_robot_projection_change(
        camera, finding_prefix="rollout_world_bev"
    )
    assert calls[0] == ("observation", camera)
    assert calls[1:] == [("render", None)] * 4
    assert adapter._runtime_findings[
        "rollout_world_bev_projection_flush_count"
    ] == 1
    assert adapter._runtime_findings[
        "rollout_world_bev_projection_flush_render_ticks"
    ] == 4


def test_public_label_catalog_cache_uses_static_identity_only_once():
    adapter = object.__new__(OmniGibsonAdapter)
    catalog = (
        SimpleNamespace(
            instance_id="chair-state-id",
            native_path="/World/chair",
            category="chair",
        ),
    )
    catalog_calls = {"count": 0}

    def object_catalog():
        catalog_calls["count"] += 1
        return catalog

    adapter.object_catalog = object_catalog
    adapter._public_label_catalog_cache = None
    adapter._runtime_findings = {}
    adapter._env = SimpleNamespace(
        robots=[SimpleNamespace(name="robot_00", prim_path="/World/robot_00")]
    )
    for _ in range(2):
        observation = {
            "seg_instance_id": np.asarray([[0, 4], [4, 0]], dtype=np.uint32)
        }
        info = {"seg_instance_id": {"0": "BACKGROUND", "4": "/World/chair/mesh"}}
        public, _ = adapter._publicize_observation_labels(observation, info)
        assert int(public["seg_instance_id"][0, 1]) == 4

    assert catalog_calls["count"] == 1
    assert adapter._runtime_findings["public_label_catalog_cache_size"] == 1
    assert isinstance(public["seg_instance_id"], np.ndarray)


def test_public_labels_use_explicit_torch_tensor_detection_when_available():
    torch = pytest.importorskip("torch")
    adapter = object.__new__(OmniGibsonAdapter)
    adapter._th = torch
    adapter._public_label_catalog_cache = (
        SimpleNamespace(
            instance_id="chair-state-id",
            native_path="/World/chair",
            category="chair",
        ),
    )
    adapter._runtime_findings = {}
    adapter._env = SimpleNamespace(
        robots=[SimpleNamespace(name="robot_00", prim_path="/World/robot_00")]
    )
    observation = {"seg_instance_id": torch.tensor([[0, 4]], dtype=torch.int64)}
    info = {"seg_instance_id": {"0": "BACKGROUND", "4": "/World/chair/mesh"}}
    public, _ = adapter._publicize_observation_labels(observation, info)
    assert isinstance(public["seg_instance_id"], torch.Tensor)
    assert public["seg_instance_id"].device == observation["seg_instance_id"].device


def test_complete_relation_catalog_is_recomputed_and_preserves_unchanged_edges():
    adapter = object.__new__(OmniGibsonAdapter)
    target = make_object("obj_target")
    reference = replace(
        make_object("obj_reference", x=1.0),
        category="table",
        native_path="/World/table",
    )
    adapter.object_catalog = lambda: (target, reference)
    current = {"predicate": "OnTop"}
    calls = {"count": 0}

    def relations(_catalog):
        calls["count"] += 1
        return ({
            "predicate": current["predicate"],
            "target_instance_id": target.instance_id,
            "reference_instance_id": reference.instance_id,
            "reference_category": reference.category,
        },)

    adapter.relation_candidates = relations
    adapter._relation_cache = None
    first = adapter.object_catalog_with_relations()
    current["predicate"] = "Inside"
    second = adapter.object_catalog_with_relations()
    assert calls["count"] == 2
    assert first[0].relations[0]["predicate"] == "OnTop"
    assert second[0].relations[0]["predicate"] == "Inside"
    assert second[1].relations == ()
    assert exact_state_hash(first) != exact_state_hash(second)
