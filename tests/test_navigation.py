from pathlib import Path

import numpy as np

from multi_view_world_dataset.adapters.navigation import (
    _cheap_visibility_metrics,
    _cheap_visibility_score_from_pairwise,
)
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
from multi_view_world_dataset.sampling.trajectories import trajectory_from_spatial_path
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


def test_route_first_config_has_no_straight_exit_or_shared_heading_gate():
    config = load_yaml_config(REPOSITORY / "configs" / "default.yaml")
    assert "initial_heading_probe_min_m" not in config["placement"]
    assert "initial_heading_probe_max_m" not in config["placement"]
    assert "heading_consensus_search_step_deg" not in config["placement"]
    assert "regime_trajectory_separation_headroom_m" not in config["placement"]


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
    assert result["realized_regime"] == "partial_chain"
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
    assert not np.isfinite(metrics["score"])

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

