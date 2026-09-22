from pathlib import Path
from types import SimpleNamespace

import numpy as np

from multi_view_world_dataset.adapters.navigation import (
    _bind_route,
    _cohort_spatial_weights,
    _cheap_visibility_metrics,
    _cheap_visibility_score_from_pairwise,
    _incomplete_view_cohort_regions,
    _shared_surface_target_visibility,
    measured_overlap_route_mutations,
)
import multi_view_world_dataset.adapters.navigation as navigation_adapter_module
from multi_view_world_dataset.diagnostics import _kit_log_has_gpu_device_loss
from multi_view_world_dataset.sampling.diversity import temporal_overlap_acceptance
from multi_view_world_dataset.sampling.navigation import (
    RouteCandidate,
    RegionGraph,
    build_region_graph,
    build_robot_footprint_model,
    compute_pairwise_route_compatibility,
    oriented_safe_masks,
    route_start_regions_connected,
    select_joint_route_candidates,
)
from multi_view_world_dataset.sampling.se2 import (
    plan_se2_grid,
    se2_plan_is_safe,
    swept_rotation_is_safe,
)
from multi_view_world_dataset.sampling.trajectories import (
    trajectory_from_se2_poses,
    trajectory_from_spatial_path,
    trajectory_kinematic_metrics,
)
from multi_view_world_dataset.utils.config import load_yaml_config


REPOSITORY = Path(__file__).resolve().parents[1]


def _route(route_id: str, points: list[list[float]], family: str = "one_waypoint") -> RouteCandidate:
    trajectory = trajectory_from_spatial_path(
        route_id,
        np.asarray(points, dtype=np.float64),
        0.0,
        np.eye(4),
        frames=60,
        fps=10.0,
        path_family=family,
        control_waypoints_xy=np.asarray(points, dtype=np.float64),
        planner_path_xy=np.asarray(points, dtype=np.float64),
        smoothed_path_xy=np.asarray(points, dtype=np.float64),
    )
    return RouteCandidate(
        route_id=route_id,
        floor_index=0,
        start_region=f"start_{route_id}",
        goal_region=f"goal_{route_id}",
        traversed_regions=(f"start_{route_id}", f"goal_{route_id}"),
        trajectory=trajectory,
        route_seed=7,
        minimum_footprint_clearance_m=0.03,
    )


def test_measured_overlap_route_mutation_preserves_edge_and_replaces_isolated_route(
    monkeypatch,
):
    routes = (
        _route("route_00", [[0.0, 0.0], [1.0, 0.0]]),
        _route("route_01", [[0.0, 1.0], [1.0, 1.0]]),
        _route("route_02", [[0.0, 4.0], [1.0, 4.0]]),
        _route("route_03", [[0.0, 2.0], [1.0, 2.0]]),
    )
    visible = (
        tuple(frozenset({1, 2, 3}) for _ in range(7)),
        tuple(frozenset({1, 2}) for _ in range(7)),
        tuple(frozenset({8, 9}) for _ in range(7)),
        tuple(frozenset({2, 3}) for _ in range(7)),
    )
    compatibility = SimpleNamespace(compatible=np.ones((4, 4), dtype=bool))
    context = SimpleNamespace(
        route_bank=routes,
        cheap_visible_cells=visible,
        compatibility=compatibility,
        invalid_start_pose_blacklist=set(),
    )
    config = {
        "camera": {"hfov_deg": 70.0},
        "navigation": {
            "cheap_visibility_edge_threshold": 0.20,
            "cheap_visibility_maximum_isolated_fraction": 0.71,
            "cheap_visibility_max_range_m": 8.0,
            "start_blacklist_position_quantization_m": 0.05,
            "start_blacklist_yaw_bins": 32,
        },
        "placement": {"formation_degeneracy": {
            "minimum_mean_heading_difference_deg": 8.0,
            "maximum_mean_path_similarity": 0.96,
            "minimum_spatial_coverage_m2": 1.0,
        }},
    }
    camera_mount = np.eye(4)
    camera_mount[:3, :3] = [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
    adapter = SimpleNamespace(
        config=config,
        _navigation_contexts={0: context},
        _development_camera_mounts={
            robot_id: camera_mount
            for robot_id in ("robot_00", "robot_01", "robot_02")
        },
    )
    monkeypatch.setattr(
        navigation_adapter_module,
        "_sparse_physics_preflight",
        lambda adapter, context, trajectories: None,
    )
    trajectories = tuple(
        _bind_route(route, robot_id, camera_mount)
        for robot_id, route in zip(
            ("robot_00", "robot_01", "robot_02"), routes[:3], strict=True
        )
    )
    source_metrics = {
        "floor_index": 0,
        "route_ids": ["route_00", "route_01", "route_02"],
        "joint_route_search": {},
        "nested_trajectory_sets": {},
    }
    failures = [{
        "candidate_rank": 0,
        "candidate_kind": "base",
        "reason": "trajectory_temporal_overlap_failed",
        "details": {
            "union_edges": [["robot_00", "robot_01"]],
            "maximum_consecutive_isolated_keyframes": {
                "robot_00": 0, "robot_01": 0, "robot_02": 7,
            },
            "allowed_consecutive_isolated_keyframes": 5,
            "longest_isolation_intervals": {
                "robot_02": {
                    "frame_start": 10, "frame_end": 49,
                    "sample_start_index": 1, "sample_end_index": 5,
                },
            },
            "checks": {
                "union_graph_connected": False,
                "meaningful_shared_moment": True,
                "every_robot_participates": False,
                "no_severe_isolation": False,
                "no_near_duplicate_views": True,
            },
            "dense_confirmation": {
                "passed_universal_hard_checks": True,
                "checks": {"no_severe_isolation": False},
                "union_edges": [["robot_00", "robot_01"]],
                "maximum_consecutive_isolated_keyframes": {
                    "robot_00": 0, "robot_01": 0, "robot_02": 11,
                },
                "allowed_consecutive_isolated_keyframes": 6,
                "longest_isolation_intervals": {
                    "robot_02": {
                        "frame_start": 20, "frame_end": 40,
                        "sample_start_index": 4, "sample_end_index": 8,
                    },
                },
                "keyframes": [{
                    "frame_index": 30,
                    "edges": [["robot_00", "robot_01"]],
                    "shared_surface_centroids_world": {
                        "robot_00|robot_01": [3.0, 2.0, 0.0],
                    },
                }],
            },
        },
    }]
    mutated = measured_overlap_route_mutations(
        adapter, ((trajectories, source_metrics),), failures,
        maximum_candidates=2,
    )
    assert len(mutated) == 1
    replacement_trajectories, metrics = mutated[0]
    assert metrics["route_ids"] == ["route_00", "route_01", "route_03"]
    assert metrics["nested_trajectory_sets"]["candidate_kind"] == (
        "measured_overlap_route_mutation"
    )
    evidence = metrics["joint_diversity"]["measured_overlap_route_mutation"]
    assert evidence["preserved_measured_edge"] == ["robot_00", "robot_01"]
    assert evidence["isolated_robot_id"] == "robot_02"
    assert evidence["repaired_robot_id"] == "robot_02"
    assert evidence["isolated_interval"]["frame_start"] == 20
    assert evidence["overlap_target_pair"] == ["robot_00", "robot_01"]
    assert evidence["cheap_interval_anchor_count"] >= 0
    assert evidence["target_shared_surface_visibility"][
        "visible_target_count"
    ] == 1
    assert metrics["cheap_scene_visibility"]["passed"]
    assert replacement_trajectories[2].metadata["route_id"] == "route_03"


def test_shared_surface_target_visibility_prefers_route_facing_gt_centroid():
    forward_mount = np.eye(4)
    forward_mount[:3, :3] = [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
    facing = _bind_route(
        _route("facing", [[0.0, 0.0], [1.0, 0.0]]),
        "robot_00",
        forward_mount,
    )
    away = _bind_route(
        _route("away", [[0.0, 0.0], [-1.0, 0.0]]),
        "robot_00",
        forward_mount,
    )
    targets = [{
        "frame_index": 30,
        "pair": "robot_01|robot_02",
        "centroid_world": [3.0, 0.0, 0.0],
    }]
    facing_score = _shared_surface_target_visibility(
        facing, targets, camera_hfov_deg=70.0, maximum_range_m=8.0
    )
    away_score = _shared_surface_target_visibility(
        away, targets, camera_hfov_deg=70.0, maximum_range_m=8.0
    )
    assert facing_score["visible_target_count"] == 1
    assert away_score["visible_target_count"] == 0
    assert facing_score["mean_alignment"] > away_score["mean_alignment"]


def test_route_first_config_has_no_straight_exit_or_shared_heading_gate():
    config = load_yaml_config(REPOSITORY / "configs" / "default.yaml")
    assert "initial_heading_probe_min_m" not in config["placement"]
    assert "initial_heading_probe_max_m" not in config["placement"]
    assert "heading_consensus_search_step_deg" not in config["placement"]
    assert "regime_trajectory_separation_headroom_m" not in config["placement"]


def test_kit_log_device_loss_detection_is_specific(tmp_path: Path):
    log = tmp_path / "kit_test.log"
    log.write_text("ordinary renderer warning\n", encoding="utf-8")
    assert not _kit_log_has_gpu_device_loss(log)
    log.write_text(
        "ordinary renderer warning\nVkResult: ERROR_DEVICE_LOST\n",
        encoding="utf-8",
    )
    assert _kit_log_has_gpu_device_loss(log)


def test_route_can_turn_immediately_and_yaw_follows_accepted_tangent():
    route = _route("turn", [[0.0, 0.0], [0.2, 0.0], [0.2, 1.0]])
    positions = route.trajectory.base_to_world[:, :2, 3]
    yaws = np.unwrap(np.arctan2(
        route.trajectory.base_to_world[:, 1, 0],
        route.trajectory.base_to_world[:, 0, 0],
    ))
    tangent = np.gradient(positions, axis=0)
    expected = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
    assert np.allclose(yaws, expected)
    assert yaws[-1] - yaws[0] > 1.0
    assert np.linalg.norm(positions[1] - positions[0]) < 0.2


def test_robot_footprint_uses_chassis_wheels_and_casters():
    model = build_robot_footprint_model(
        {
            "chassis_link": [[-0.30, -0.20, 0.1], [0.30, -0.20, 0.1], [0.30, 0.20, 0.1], [-0.30, 0.20, 0.1]],
            "wheel_left": [[-0.15, 0.28, 0.0], [0.15, 0.28, 0.0]],
            "wheel_right": [[-0.15, -0.28, 0.0], [0.15, -0.28, 0.0]],
            "caster_wheel_left": [[-0.34, 0.18, 0.0], [-0.28, 0.24, 0.0]],
            "caster_wheel_right": [[-0.34, -0.18, 0.0], [-0.28, -0.24, 0.0]],
        },
        safety_margin_m=0.03,
        yaw_bins=16,
        old_reset_aabb_extent_xy_m=(0.68, 0.56),
    )
    assert set(model.source_collision_links) == {
        "chassis_link", "wheel_left", "wheel_right",
        "caster_wheel_left", "caster_wheel_right",
    }
    assert model.length_m >= 0.68
    assert model.width_m >= 0.56
    assert model.area_m2 > 0.3
    assert model.circumscribed_radius_m > 0.4


def test_orientation_aware_footprint_mask_blocks_wall_contact():
    free = np.ones((9, 9), dtype=bool)
    free[:, 7:] = False
    offsets = (
        np.asarray([[0, 0], [0, 1], [0, 2]]),
        np.asarray([[0, 0], [1, 0], [2, 0]]),
    )
    safe = oriented_safe_masks(free, offsets)
    assert not safe[0, 4, 6]
    assert safe[1, 4, 6]


def _right_angle_se2_masks() -> np.ndarray:
    masks = np.zeros((8, 7, 7), dtype=bool)
    masks[0, 3, 1:4] = True
    masks[:, 3, 3] = True
    masks[2, 3:6, 3] = True
    return masks


def test_se2_planner_supports_collision_checked_stop_and_turn():
    masks = _right_angle_se2_masks()
    plan = plan_se2_grid(
        masks, (3, 1), (5, 3), start_yaw_index=0,
        rotation_cost_cells=0.1,
    )
    assert plan is not None
    assert plan.contains_stationary_turn
    assert plan.actions == (
        "forward", "forward", "rotate_left", "rotate_left", "forward", "forward"
    )
    assert se2_plan_is_safe(plan, masks)
    assert np.isclose(plan.translation_length_cells, 4.0)


def test_swept_rotation_rejects_unsafe_intermediate_yaw():
    masks = np.ones((8, 3, 3), dtype=bool)
    masks[1, 1, 1] = False
    assert not swept_rotation_is_safe(
        masks, 1, 1, 0, 2, direction=1
    )
    assert swept_rotation_is_safe(
        masks, 1, 1, 0, 2, direction=-1
    )


def test_se2_planner_is_seed_free_deterministic_and_never_uses_unsafe_pose():
    masks = _right_angle_se2_masks()
    first = plan_se2_grid(masks, (3, 1), (5, 3), start_yaw_index=0)
    second = plan_se2_grid(masks, (3, 1), (5, 3), start_yaw_index=0)
    assert first == second
    assert first is not None
    assert all(masks[item.yaw_index, item.row, item.column] for item in first.states)


def test_se2_forward_primitive_never_approximates_a_misaligned_yaw():
    masks = np.ones((16, 5, 5), dtype=bool)
    # yaw bin 1 is 22.5 degrees, but an 8-neighbor raster has no matching
    # direction. The planner must rotate to a true tangent before translating.
    plan = plan_se2_grid(
        masks, (2, 1), (2, 3), start_yaw_index=1,
        rotation_cost_cells=0.1,
    )
    assert plan is not None
    assert plan.actions[0].startswith("rotate_")
    for left, right, action in zip(
        plan.states[:-1], plan.states[1:], plan.actions, strict=True
    ):
        if action == "forward":
            yaw = 2.0 * np.pi * left.yaw_index / 16
            tangent = np.arctan2(right.row - left.row, right.column - left.column)
            assert abs((yaw - tangent + np.pi) % (2.0 * np.pi) - np.pi) < 1.0e-9


def test_se2_time_parameterization_has_stationary_turns_without_lateral_slip():
    poses = np.asarray([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.0, np.pi / 2.0],
        [0.5, 0.5, np.pi / 2.0],
    ])
    trajectory = trajectory_from_se2_poses(
        "robot_00", poses, 0.0, np.eye(4), frames=60, fps=10.0,
        maximum_linear_speed_mps=0.8,
        maximum_angular_speed_radps=1.2,
        maximum_acceleration_mps2=1.5,
        path_family="one_waypoint",
    )
    metrics = trajectory_kinematic_metrics(trajectory)
    assert trajectory.frames == 60
    assert metrics["stationary_turn_count"] == 1
    assert metrics["stationary_turn_frame_count"] > 0
    assert metrics["maximum_lateral_slip_m"] < 1.0e-9
    assert metrics["maximum_linear_speed_mps"] <= 0.8
    assert metrics["maximum_angular_speed_radps"] <= 1.2
    assert metrics["maximum_acceleration_mps2"] <= 1.5


def test_region_graph_uses_cell_topology_and_preserves_connector():
    free = np.ones((3, 5), dtype=bool)
    labels = np.asarray([
        ["room_a", "room_a", "", "room_b", "room_b"],
        ["room_a", "room_a", "", "room_b", "room_b"],
        ["room_a", "room_a", "", "room_b", "room_b"],
    ], dtype=object)
    graph, filled = build_region_graph(free, labels)
    connector = graph.connector_nodes[0]
    assert tuple(sorted(("room_a", connector))) in graph.edges
    assert tuple(sorted((connector, "room_b"))) in graph.edges
    assert ("room_a", "room_b") not in graph.edges
    assert np.all(filled[:, 2] == connector)


def test_route_start_regions_must_share_footprint_topology_component():
    routes = (
        _route("a", [[0.0, 0.0], [1.0, 0.0]]),
        _route("b", [[0.0, 1.0], [1.0, 1.0]]),
        _route("c", [[0.0, 2.0], [1.0, 2.0]]),
    )
    routes = (
        RouteCandidate(**{**routes[0].__dict__, "start_region": "living"}),
        RouteCandidate(**{**routes[1].__dict__, "start_region": "entryway"}),
        RouteCandidate(**{**routes[2].__dict__, "start_region": "kitchen"}),
    )
    connected = RegionGraph(
        nodes=("living", "entryway", "kitchen"),
        edges=(("entryway", "living"), ("entryway", "kitchen")),
        cell_counts={"living": 10, "entryway": 2, "kitchen": 8},
    )
    disconnected = RegionGraph(
        nodes=("living", "entryway", "kitchen"),
        edges=(("entryway", "kitchen"),),
        cell_counts={"living": 10, "entryway": 2, "kitchen": 8},
    )
    assert route_start_regions_connected(routes, connected)
    assert not route_start_regions_connected(routes, disconnected)


def test_pairwise_compatibility_rejects_synchronized_route_collision():
    first = _route("east", [[-1.0, 0.0], [1.0, 0.0]])
    second = _route("north", [[0.0, -1.0], [0.0, 1.0]])
    separated = _route("far", [[-1.0, 2.0], [1.0, 2.0]])
    compatibility = compute_pairwise_route_compatibility(
        (first, second, separated), minimum_pairwise_distance_m=0.6
    )
    assert not compatibility.compatible[0, 1]
    assert compatibility.compatible[0, 2]
    assert compatibility.minimum_temporal_distance_m[0, 1] < 0.05
    assert np.isclose(compatibility.maximum_temporal_distance_m[0, 1], np.sqrt(2.0))
    assert compatibility.spatial_path_intersection[0, 1]
    assert compatibility.close_approach[0, 1]
    assert compatibility.heading_relation[0, 1] == "crossing"


def test_pairwise_compatibility_rejects_real_footprint_overlap():
    footprint = build_robot_footprint_model(
        {
            "chassis": [
                [-0.36, -0.36], [0.36, -0.36],
                [0.36, 0.36], [-0.36, 0.36],
            ],
        },
        safety_margin_m=0.03,
        yaw_bins=16,
        old_reset_aabb_extent_xy_m=(0.72, 0.72),
    )
    first = _route("left", [[0.0, 0.0], [1.0, 0.0]])
    overlapping = _route("overlap", [[0.0, 0.65], [1.0, 0.65]])
    separated = _route("separated", [[0.0, 0.80], [1.0, 0.80]])
    compatibility = compute_pairwise_route_compatibility(
        (first, overlapping, separated),
        minimum_pairwise_distance_m=0.6,
        footprint=footprint,
    )
    assert compatibility.minimum_temporal_distance_m[0, 1] > 0.6
    assert compatibility.footprint_collision[0, 1]
    assert not compatibility.compatible[0, 1]
    assert not compatibility.footprint_collision[0, 2]
    assert compatibility.compatible[0, 2]


def test_bounded_joint_search_is_deterministic_and_allows_nonparallel_routes():
    routes = (
        _route("east0", [[0.0, 0.0], [1.0, 0.0]]),
        _route("north0", [[2.0, 0.0], [2.0, 1.0]]),
        _route("west0", [[4.0, 0.0], [3.0, 0.0]]),
        _route("east1", [[0.0, 3.0], [1.0, 3.0]], "direct"),
        _route("north1", [[2.0, 3.0], [2.0, 4.0]]),
        _route("west1", [[4.0, 3.0], [3.0, 3.0]]),
    )
    compatibility = compute_pairwise_route_compatibility(
        routes, minimum_pairwise_distance_m=0.6
    )
    first = select_joint_route_candidates(
        routes, compatibility, np.random.default_rng(17),
        top_k=4, search_budget=256, minimum_waypoint_trajectories=1,
    )
    second = select_joint_route_candidates(
        routes, compatibility, np.random.default_rng(17),
        top_k=4, search_budget=256, minimum_waypoint_trajectories=1,
    )
    assert first == second
    assert first
    selected = [routes[index] for index in first[0]]
    start_yaws = [
        float(np.arctan2(route.trajectory.base_to_world[0, 1, 0], route.trajectory.base_to_world[0, 0, 0]))
        for route in selected
    ]
    assert np.ptp(np.unwrap(start_yaws)) > 1.0


def test_unclassified_overlap_uses_union_connectivity_then_realized_label():
    keyframes = [
        {"edges": [["r0", "r1"]], "connected": False, "near_duplicate_pairs": []},
        {"edges": [["r1", "r2"]], "connected": False, "near_duplicate_pairs": []},
        {"edges": [], "connected": False, "near_duplicate_pairs": []},
    ]
    result = temporal_overlap_acceptance(
        ("r0", "r1", "r2"),
        keyframes,
        regime="unclassified",
        regime_connected_fraction_target={
            "dense_shared": 0.6, "partial_chain": 0.3, "exploratory": 0.15,
        },
        regime_shared_keyframe_fraction_target={
            "dense_shared": 0.6, "partial_chain": 0.3, "exploratory": 0.15,
        },
        regime_maximum_consecutive_isolated_keyframes={
            "dense_shared": 2, "partial_chain": 4, "exploratory": 5,
        },
    )
    assert result["passed"]
    assert result["checks"]["union_graph_connected"]
    assert result["realized_regime"] == "exploratory"
    assert result["regime_target_match"]


def test_cheap_visibility_uses_temporal_union_graph_not_all_pairs():
    routes = (
        _route("a", [[0.0, 0.0], [1.0, 0.0]]),
        _route("b", [[0.0, 1.0], [1.0, 1.0]]),
        _route("c", [[0.0, 2.0], [1.0, 2.0]]),
    )
    visible = (
        (frozenset({1, 2}), frozenset({1, 2}), frozenset({8, 9})),
        (frozenset({1, 2}), frozenset({2, 3}), frozenset({3, 4})),
        (frozenset({8, 9}), frozenset({3, 4}), frozenset({3, 4})),
    )
    metrics = _cheap_visibility_metrics(
        routes, routes, visible, edge_threshold=0.5,
        maximum_isolated_fraction=0.67,
    )
    assert metrics["passed"]
    assert metrics["union_graph_connected"]
    assert metrics["robot_participates"] == [True, True, True]
    assert metrics["connected_keyframe_fraction"] < 1.0
    assert np.isfinite(metrics["score"])


def test_cheap_visibility_rejects_robot_that_never_participates():
    routes = (
        _route("a", [[0.0, 0.0], [1.0, 0.0]]),
        _route("b", [[0.0, 1.0], [1.0, 1.0]]),
        _route("c", [[0.0, 2.0], [1.0, 2.0]]),
    )
    visible = (
        (frozenset({1, 2}),) * 3,
        (frozenset({1, 2}),) * 3,
        (frozenset({8, 9}),) * 3,
    )
    metrics = _cheap_visibility_metrics(
        routes, routes, visible, edge_threshold=0.5,
        maximum_isolated_fraction=0.67,
    )
    assert not metrics["passed"]
    assert "union_graph_disconnected" in metrics["failure_reasons"]
    assert "robot_never_participates" in metrics["failure_reasons"]
    assert np.isfinite(metrics["score"])
    assert metrics["score"] < 0.0

def test_cheap_visibility_score_prefers_persistent_robot_participation():
    routes = (
        _route("a", [[0.0, 0.0], [1.0, 0.0]]),
        _route("b", [[0.0, 1.0], [1.0, 1.0]]),
        _route("c", [[0.0, 2.0], [1.0, 2.0]]),
    )
    persistent = (
        (frozenset({1, 2}),) * 4,
        (frozenset({1, 2}),) * 4,
        (frozenset({2, 3}),) * 4,
    )
    fleeting = (
        (frozenset({1, 2}), frozenset({8}), frozenset({8}), frozenset({8})),
        (frozenset({1, 2}), frozenset({9}), frozenset({9}), frozenset({9})),
        (frozenset({2, 3}), frozenset({10}), frozenset({10}), frozenset({10})),
    )
    persistent_metrics = _cheap_visibility_metrics(
        routes, routes, persistent, edge_threshold=0.5,
        maximum_isolated_fraction=0.75,
    )
    fleeting_metrics = _cheap_visibility_metrics(
        routes, routes, fleeting, edge_threshold=0.5,
        maximum_isolated_fraction=0.75,
    )
    assert persistent_metrics["passed"]
    assert fleeting_metrics["passed"]
    assert persistent_metrics["score"] > fleeting_metrics["score"]
    assert persistent_metrics["minimum_robot_participation_fraction"] == 1.0


def test_joint_search_reserves_visibility_priority_shortlist_slots():
    routes = tuple(
        _route(
            f"r{index}",
            [[float(index) * 2.0, 0.0], [float(index) * 2.0 + 1.0, 0.0]],
        )
        for index in range(6)
    )
    compatibility = compute_pairwise_route_compatibility(
        routes, minimum_pairwise_distance_m=0.6
    )

    def visibility_score(selected):
        ids = frozenset(route.route_id for route in selected)
        return 10.0 if ids == frozenset({"r0", "r1", "r2"}) else 1.0

    selected = select_joint_route_candidates(
        routes,
        compatibility,
        np.random.default_rng(23),
        top_k=1,
        search_budget=4096,
        minimum_waypoint_trajectories=0,
        cheap_visibility_score=visibility_score,
        visibility_priority_fraction=1.0,
        visibility_score_weight=0.0,
    )
    assert selected == ((0, 1, 2),)

def test_precomputed_cheap_visibility_score_matches_detailed_score():
    routes = (
        _route("a", [[0.0, 0.0], [1.0, 0.0]]),
        _route("b", [[0.0, 1.0], [1.0, 1.0]]),
        _route("c", [[0.0, 2.0], [1.0, 2.0]]),
    )
    visible = (
        (frozenset({1, 2}), frozenset({1, 2}), frozenset({8, 9})),
        (frozenset({1, 2}), frozenset({2, 3}), frozenset({3, 4})),
        (frozenset({8, 9}), frozenset({3, 4}), frozenset({3, 4})),
    )
    overlap = np.ones((3, 3, 3), dtype=np.float32)
    for left in range(3):
        for right in range(left + 1, 3):
            for frame in range(3):
                denominator = max(
                    1, min(len(visible[left][frame]), len(visible[right][frame]))
                )
                value = len(
                    visible[left][frame] & visible[right][frame]
                ) / denominator
                overlap[left, right, frame] = value
                overlap[right, left, frame] = value
    detailed = _cheap_visibility_metrics(
        routes, routes, visible, edge_threshold=0.5,
        maximum_isolated_fraction=0.67,
    )
    fast = _cheap_visibility_score_from_pairwise(
        (0, 1, 2), overlap, edge_threshold=0.5,
        maximum_isolated_fraction=0.67,
    )
    assert np.isclose(fast, detailed["score"])


def test_disconnected_precomputed_visibility_matches_detailed_penalty():
    routes = (
        _route("a", [[0.0, 0.0], [1.0, 0.0]]),
        _route("b", [[0.0, 2.0], [1.0, 2.0]]),
        _route("c", [[0.0, 4.0], [1.0, 4.0]]),
    )
    visible = tuple(
        tuple(frozenset({10 * robot + frame}) for frame in range(3))
        for robot in range(3)
    )
    overlap = np.zeros((3, 3, 3), dtype=np.float32)
    for index in range(3):
        overlap[index, index, :] = 1.0
    detailed = _cheap_visibility_metrics(
        routes, routes, visible, edge_threshold=0.2,
        maximum_isolated_fraction=0.85,
    )
    fast = _cheap_visibility_score_from_pairwise(
        (0, 1, 2), overlap, edge_threshold=0.2,
        maximum_isolated_fraction=0.85,
    )
    assert detailed["score"] == -3.0
    assert np.isclose(fast, detailed["score"])


def test_view_cohort_focus_only_targets_productive_incomplete_regions():
    from collections import Counter

    counts = Counter({"room_a": 2, "room_b": 3, "room_c": 1})
    assert _incomplete_view_cohort_regions(
        ["room_a", "room_b", "room_c", "room_d"], counts, 3
    ) == ("room_a", "room_c")


def test_view_cohort_spatial_weights_are_soft_normalized_and_anchor_biased():
    weights = _cohort_spatial_weights(
        np.asarray([[0.0, 0.0], [1.0, 0.0], [8.0, 0.0]]),
        np.asarray([0.0, 0.0]), scale_m=2.0, probability_floor=0.05,
    )
    assert np.isclose(weights.sum(), 1.0)
    assert weights[0] > weights[1] > weights[2] > 0.0
