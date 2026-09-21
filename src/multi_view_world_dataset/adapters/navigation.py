from __future__ import annotations

import time
from collections import Counter
from dataclasses import replace
from typing import Any

import numpy as np

from multi_view_world_dataset.errors import SampleRejected, SimulatorUnavailableError
from multi_view_world_dataset.sampling.diversity import (
    formation_degenerate,
    joint_trajectory_metrics,
    stable_seed,
)
from multi_view_world_dataset.sampling.navigation import (
    NavigationContext,
    RouteCandidate,
    build_region_graph,
    build_robot_footprint_model,
    compute_pairwise_route_compatibility,
    connected_components,
    oriented_safe_masks,
    route_candidate_from_trajectory,
    sample_footprint_interior,
    select_joint_route_candidates,
)
from multi_view_world_dataset.sampling.se2 import (
    SE2GridPlan,
    plan_se2_waypoints,
    se2_plan_is_safe,
)
from multi_view_world_dataset.sampling.trajectories import (
    _normalised_family_distribution,
    _sample_route_controls,
    densify_polyline,
    trajectory_from_se2_poses,
    trajectory_from_spatial_path,
)
from multi_view_world_dataset.schema.records import Trajectory


def _robot_footprint(adapter: Any) -> Any:
    robot = adapter._env.robots[0]
    base_to_world = adapter._pose_matrix(robot)
    world_to_base = np.linalg.inv(base_to_world)
    disabled = set(getattr(robot, "disabled_collision_link_names", ()))
    points_by_link: dict[str, np.ndarray] = {}
    for link_name, link in sorted(robot.links.items()):
        if link_name in disabled or not bool(getattr(link, "has_collision_meshes", False)):
            continue
        points = getattr(link, "collision_boundary_points_world", None)
        if points is None:
            continue
        points = adapter._native_value(points).astype(np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            continue
        homogeneous = np.column_stack((points, np.ones(len(points))))
        local = homogeneous @ world_to_base.T
        points_by_link[str(link_name)] = local[:, :3]
    navigation = adapter.config["navigation"]
    return build_robot_footprint_model(
        points_by_link,
        safety_margin_m=float(navigation["footprint_safety_margin_m"]),
        yaw_bins=int(navigation["footprint_yaw_bins"]),
        old_reset_aabb_extent_xy_m=adapter._native_value(
            robot.reset_joint_pos_aabb_extent[:2]
        ),
    )


def _point_free_mask(adapter: Any, floor_index: int) -> np.ndarray:
    scene = adapter._require_scene()
    source = adapter._floor_supported_traversability_source(floor_index)
    point_free = adapter._native_value(source) == 255
    floor_id = f"floor_{floor_index:02d}"
    floor_height = float(scene.get_floor_height(floor_index))
    obstacle_height = float(
        adapter.config["placement"]["dynamic_object_path_obstacle_height_m"]
    )
    margin = float(adapter.config["navigation"]["dynamic_obstacle_margin_m"])
    objects = [
        obj
        for obj in adapter.object_catalog()
        if not obj.structural
        and obj.floor_id == floor_id
        and obj.bbox_max_world[2] > floor_height + 0.10
        and obj.bbox_min_world[2] < floor_height + obstacle_height
    ]
    pixels = np.argwhere(point_free)
    if len(pixels):
        world = adapter._native_value(
            scene.trav_map.map_to_world(
                adapter._th.as_tensor(pixels, dtype=adapter._th.int64)
            )
        ).astype(np.float64)
        keep = np.ones(len(pixels), dtype=bool)
        for obj in objects:
            keep &= ~(
                (world[:, 0] >= obj.bbox_min_world[0] - margin)
                & (world[:, 0] <= obj.bbox_max_world[0] + margin)
                & (world[:, 1] >= obj.bbox_min_world[1] - margin)
                & (world[:, 1] <= obj.bbox_max_world[1] + margin)
            )
        blocked = pixels[~keep]
        point_free[blocked[:, 0], blocked[:, 1]] = False
    adapter._runtime_findings["navigation_dynamic_obstacles"] = {
        "floor_index": floor_index,
        "object_count": len(objects),
        "aabb_numerical_margin_m": margin,
    }
    return point_free


def _footprint_offsets(adapter: Any, footprint: Any, floor_index: int) -> tuple[np.ndarray, ...]:
    trav_map = adapter._require_scene().trav_map
    resolution = float(trav_map.map_resolution)
    fraction = float(
        adapter.config["navigation"]["footprint_sampling_spacing_fraction"]
    )
    local_samples = sample_footprint_interior(
        footprint.polygon_xy, resolution * fraction
    )
    shape = tuple(trav_map.floor_map[floor_index].shape)
    center_pixel = np.asarray([shape[0] // 2, shape[1] // 2], dtype=np.int64)
    center_world = adapter._native_value(
        trav_map.map_to_world(adapter._th.as_tensor(center_pixel))
    ).astype(np.float64)
    offsets: list[np.ndarray] = []
    for yaw_index in range(footprint.yaw_bins):
        yaw = 2.0 * np.pi * yaw_index / footprint.yaw_bins
        cosine, sine = np.cos(yaw), np.sin(yaw)
        rotation = np.asarray([[cosine, -sine], [sine, cosine]])
        world = center_world + local_samples @ rotation.T
        mapped = adapter._world_to_map_preserving_batch(trav_map, world)
        offsets.append(np.unique(mapped - center_pixel, axis=0).astype(np.int64))
    return tuple(offsets)


def _map_points(adapter: Any, points_xy: np.ndarray) -> np.ndarray:
    return adapter._world_to_map_preserving_batch(
        adapter._require_scene().trav_map, np.asarray(points_xy, dtype=np.float64)
    )


def _calibrated_heading_primitives(
    adapter: Any, yaw_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate global mathematical yaw against the installed map axes."""
    cache = getattr(adapter, "_se2_heading_steps_cache", None)
    if cache is None:
        cache = {}
        adapter._se2_heading_steps_cache = cache
    scene_key = (str(getattr(adapter, "_scene_id", "")), int(yaw_bins))
    if scene_key in cache:
        return cache[scene_key]
    trav_map = adapter._require_scene().trav_map
    shape = np.asarray(trav_map.floor_map[0].shape, dtype=np.int64)
    center = shape // 2
    deltas = np.asarray([
        [-1, -1], [-1, 0], [-1, 1], [0, -1],
        [0, 1], [1, -1], [1, 0], [1, 1],
    ], dtype=np.int64)
    pixels = np.vstack((center, center[None, :] + deltas))
    world = adapter._native_value(
        trav_map.map_to_world(adapter._th.as_tensor(pixels, dtype=adapter._th.int64))
    ).astype(np.float64)
    directions = world[1:, :2] - world[0, :2]
    angles = np.arctan2(directions[:, 1], directions[:, 0])
    desired = 2.0 * np.pi * np.arange(yaw_bins, dtype=np.float64) / yaw_bins
    difference = np.abs(
        (desired[:, None] - angles[None, :] + np.pi) % (2.0 * np.pi) - np.pi
    )
    best = np.argmin(difference, axis=1)
    calibrated = deltas[best]
    # With 32 footprint bins and an 8-neighbor raster, only eight orientations
    # have a forward primitive. Intermediate bins remain essential for swept
    # collision checking during stop-and-turn actions, but allowing them to
    # translate would introduce up to 22.5 degrees of lateral slip.
    forward_enabled = difference[np.arange(yaw_bins), best] <= 1.0e-7
    cache[scene_key] = (calibrated, forward_enabled)
    return cache[scene_key]


def _se2_route_candidate(
    adapter: Any,
    safe_masks: np.ndarray,
    controls_xy: np.ndarray,
    *,
    robot_id: str,
    floor_z: float,
    camera_mount: np.ndarray,
    path_family: str,
) -> tuple[Trajectory, SE2GridPlan] | None:
    """Plan and time-parameterize all control segments in true SE(2)."""
    navigation = adapter.config["navigation"]
    trajectory_config = adapter.config["trajectory"]
    cells = _map_points(adapter, controls_xy)
    heading_steps, forward_enabled = _calibrated_heading_primitives(
        adapter, len(safe_masks)
    )
    plan = plan_se2_waypoints(
        safe_masks,
        cells,
        heading_steps=heading_steps,
        forward_enabled_by_yaw=forward_enabled,
        rotation_cost_cells=float(navigation["se2_rotation_cost_cells"]),
        yaw_freedom_penalty=float(navigation["yaw_freedom_ranking_weight"]),
        maximum_expansions=int(navigation["se2_maximum_expansions"]),
    )
    if plan is None or not se2_plan_is_safe(plan, safe_masks):
        return None
    state_pixels = np.asarray(
        [[state.row, state.column] for state in plan.states], dtype=np.int64
    )
    world_xy = adapter._native_value(
        adapter._require_scene().trav_map.map_to_world(
            adapter._th.as_tensor(state_pixels, dtype=adapter._th.int64)
        )
    ).astype(np.float64)
    yaws = 2.0 * np.pi * np.asarray(
        [state.yaw_index for state in plan.states], dtype=np.float64
    ) / len(safe_masks)
    # Unwrap adjacent one-bin rotations while leaving translations unchanged.
    yaws = np.unwrap(yaws)
    poses = np.column_stack((world_xy[:, :2], yaws))
    try:
        trajectory = trajectory_from_se2_poses(
            robot_id,
            poses,
            floor_z,
            camera_mount,
            frames=int(adapter.config["dataset"]["frames"]),
            fps=float(adapter.config["dataset"]["fps"]),
            maximum_linear_speed_mps=float(
                trajectory_config["maximum_linear_speed_mps"]
            ),
            maximum_angular_speed_radps=float(
                trajectory_config["maximum_angular_speed_radps"]
            ),
            maximum_acceleration_mps2=float(
                trajectory_config["maximum_acceleration_mps2"]
            ),
            path_family=path_family,
            control_waypoints_xy=controls_xy,
            planner_path_xy=world_xy[:, :2],
            metadata={
                "planner": "project_se2_orientation_lattice",
                "se2_expanded_state_count": plan.expanded_state_count,
                "se2_rotation_angle_rad": plan.rotation_angle_rad,
                "se2_contains_stationary_turn": plan.contains_stationary_turn,
                "planner_geodesic_length_m": plan.translation_length_cells
                * float(adapter._require_scene().trav_map.map_resolution),
            },
        )
    except ValueError:
        return None
    # Revalidate every physical output frame against its exact orientation bin.
    frame_cells = _map_points(adapter, trajectory.base_to_world[:, :2, 3])
    frame_yaws = np.arctan2(
        trajectory.base_to_world[:, 1, 0], trajectory.base_to_world[:, 0, 0]
    )
    bins = np.rint(
        (frame_yaws % (2.0 * np.pi)) * len(safe_masks) / (2.0 * np.pi)
    ).astype(np.int64) % len(safe_masks)
    height, width = safe_masks.shape[1:]
    inside = (
        (frame_cells[:, 0] >= 0) & (frame_cells[:, 0] < height)
        & (frame_cells[:, 1] >= 0) & (frame_cells[:, 1] < width)
    )
    if not np.all(inside) or not np.all(
        safe_masks[bins, frame_cells[:, 0], frame_cells[:, 1]]
    ):
        return None
    return trajectory, plan


def _path_validator(adapter: Any, context_masks: np.ndarray):
    height, width = context_masks.shape[1:]
    yaw_bins = len(context_masks)

    def validate(points_xy: np.ndarray) -> bool:
        points = np.asarray(points_xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not len(points):
            return False
        pixels = _map_points(adapter, points)
        tangents = np.gradient(points, axis=0) if len(points) > 1 else np.asarray([[1.0, 0.0]])
        yaws = np.arctan2(tangents[:, 1], tangents[:, 0])
        bins = np.rint((yaws % (2.0 * np.pi)) * yaw_bins / (2.0 * np.pi)).astype(int) % yaw_bins
        inside = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < height)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < width)
        )
        return bool(
            np.all(inside)
            and np.all(context_masks[bins, pixels[:, 0], pixels[:, 1]])
        )

    return validate


def _planner(adapter: Any, floor_index: int, planner_mask: np.ndarray):
    scene = adapter._require_scene()
    trav_map = scene.trav_map
    original_shape = tuple(trav_map.floor_map[floor_index].shape)
    if planner_mask.shape != original_shape:
        raise ValueError("planner mask shape differs from installed traversability map")
    planning_source = adapter._th.where(
        adapter._th.as_tensor(planner_mask, dtype=adapter._th.bool),
        adapter._th.full_like(trav_map.floor_map[floor_index], 255),
        adapter._th.zeros_like(trav_map.floor_map[floor_index]),
    )

    def plan(source_xy: np.ndarray, target_xy: np.ndarray):
        original_interval = int(trav_map.waypoint_interval)
        original_map = trav_map.floor_map[floor_index]
        original_radius = float(trav_map.default_erosion_radius)
        trav_map.waypoint_interval = 1
        trav_map.floor_map[floor_index] = planning_source
        trav_map.default_erosion_radius = 0.5 * float(trav_map.map_resolution)
        try:
            path, distance = scene.get_shortest_path(
                floor_index,
                adapter._th.as_tensor(source_xy, dtype=adapter._th.float32),
                adapter._th.as_tensor(target_xy, dtype=adapter._th.float32),
                entire_path=True,
                robot=None,
            )
        finally:
            trav_map.floor_map[floor_index] = original_map
            trav_map.waypoint_interval = original_interval
            trav_map.default_erosion_radius = original_radius
        if path is None or distance is None:
            return None
        return adapter._native_value(path).astype(np.float64), float(
            adapter._native_value(distance)
        )

    return plan


def _region_labels_for_points(
    adapter: Any, label_grid: np.ndarray, points_xy: np.ndarray
) -> tuple[str, ...]:
    pixels = _map_points(adapter, points_xy)
    height, width = label_grid.shape
    labels = []
    for row, column in pixels:
        if 0 <= row < height and 0 <= column < width and label_grid[row, column]:
            labels.append(str(label_grid[row, column]))
        else:
            labels.append("outside_navigation")
    return tuple(labels)

def _cheap_scene_view_cache(
    adapter: Any,
    point_free_mask: np.ndarray,
    routes: list[RouteCandidate],
) -> tuple[
    tuple[tuple[frozenset[int], ...], ...],
    np.ndarray,
    np.ndarray,
]:
    """Approximate scene visibility on the occupancy raster for ranking only."""
    navigation = adapter.config["navigation"]
    frame_count = int(navigation["cheap_visibility_keyframes"])
    ray_count = int(navigation["cheap_visibility_ray_count"])
    maximum_range = float(navigation["cheap_visibility_max_range_m"])
    resolution = float(adapter._require_scene().trav_map.map_resolution)
    hfov = np.deg2rad(float(adapter.config["camera"]["hfov_deg"]))
    ray_offsets = np.linspace(-0.5 * hfov, 0.5 * hfov, ray_count)
    ranges = np.arange(resolution, maximum_range + 0.5 * resolution, resolution)
    height, width = point_free_mask.shape
    cached: list[tuple[frozenset[int], ...]] = []
    for route in routes:
        poses = route.trajectory.base_to_world
        keyframes = np.unique(
            np.rint(np.linspace(0, len(poses) - 1, frame_count)).astype(int)
        )
        route_views: list[frozenset[int]] = []
        for frame_index in keyframes:
            pose = poses[int(frame_index)]
            origin = pose[:2, 3]
            yaw = float(np.arctan2(pose[1, 0], pose[0, 0]))
            visible: set[int] = set()
            angles = yaw + ray_offsets
            directions = np.column_stack((np.cos(angles), np.sin(angles)))
            samples = (
                origin[None, None, :]
                + directions[:, None, :] * ranges[None, :, None]
            )
            # world_to_map accepts a batch. Mapping all rays in one call avoids
            # hundreds of tiny tensor conversions per route keyframe while
            # preserving the per-ray first-obstacle stopping rule below.
            mapped_rays = _map_points(
                adapter, samples.reshape(-1, 2)
            ).reshape(ray_count, len(ranges), 2)
            for pixels in mapped_rays:
                for row, column in pixels:
                    if not (0 <= row < height and 0 <= column < width):
                        break
                    if not point_free_mask[row, column]:
                        # Preserve the first occupied cell: it is the scene
                        # surface (wall / furniture) seen by this ray.
                        visible.add(int(row) * width + int(column))
                        break
                    visible.add(int(row) * width + int(column))
            route_views.append(frozenset(visible))
        cached.append(tuple(route_views))
    pairwise = np.eye(len(routes), dtype=np.float64)
    overlap_by_keyframe = np.ones(
        (len(routes), len(routes), frame_count), dtype=np.float32
    )
    for left in range(len(routes)):
        for right in range(left + 1, len(routes)):
            similarities = []
            for keyframe_index, (left_view, right_view) in enumerate(zip(
                cached[left], cached[right], strict=True
            )):
                denominator = max(1, min(len(left_view), len(right_view)))
                similarity = len(left_view & right_view) / denominator
                similarities.append(similarity)
                overlap_by_keyframe[left, right, keyframe_index] = similarity
                overlap_by_keyframe[right, left, keyframe_index] = similarity
            value = float(np.mean(similarities)) if similarities else 0.0
            pairwise[left, right] = pairwise[right, left] = value
    return tuple(cached), pairwise, overlap_by_keyframe


def _cheap_visibility_metrics(
    selected: tuple[RouteCandidate, ...],
    routes: tuple[RouteCandidate, ...],
    visible_cells: tuple[tuple[frozenset[int], ...], ...],
    *,
    edge_threshold: float,
    maximum_isolated_fraction: float,
) -> dict[str, Any]:
    """Evaluate the cheap proxy with the same graph semantics as GT depth.

    A triplet may be a temporal chain: it need not be connected at every
    keyframe and no individual pair is required to overlap throughout.  The
    union graph must connect all robots, every robot must participate in at
    least one edge, and no robot may remain isolated for almost the entire
    route. Failed combinations retain a finite, penalized score: if no proxy
    candidate passes, the physically best Top-K still reaches exact GT-depth
    preflight instead of turning an approximate proxy into a hard gate.
    """
    route_indices = {route.route_id: index for index, route in enumerate(routes)}
    indices = [route_indices[route.route_id] for route in selected]
    robot_count = len(indices)
    frame_count = min((len(visible_cells[index]) for index in indices), default=0)
    if robot_count < 2 or frame_count == 0:
        return {
            "passed": False,
            "score": -1.0e6,
            "failure_reason": "empty_visibility_proxy",
            "keyframe_count": frame_count,
        }

    union_adjacency = np.eye(robot_count, dtype=bool)
    isolated_runs = np.zeros(robot_count, dtype=np.int64)
    maximum_isolated_runs = np.zeros(robot_count, dtype=np.int64)
    participation_counts = np.zeros(robot_count, dtype=np.int64)
    connected_count = 0
    shared_count = 0
    pair_values: list[float] = []
    keyframes: list[dict[str, Any]] = []
    for frame_index in range(frame_count):
        adjacency = np.eye(robot_count, dtype=bool)
        edges: list[list[int]] = []
        overlaps: dict[str, float] = {}
        for left in range(robot_count):
            for right in range(left + 1, robot_count):
                left_view = visible_cells[indices[left]][frame_index]
                right_view = visible_cells[indices[right]][frame_index]
                denominator = max(1, min(len(left_view), len(right_view)))
                overlap = len(left_view & right_view) / denominator
                pair_values.append(overlap)
                overlaps[f"{left}-{right}"] = float(overlap)
                if overlap >= edge_threshold:
                    adjacency[left, right] = adjacency[right, left] = True
                    edges.append([left, right])
        union_adjacency |= adjacency
        reached = {0}
        frontier = [0]
        while frontier:
            node = frontier.pop()
            for neighbor in np.flatnonzero(adjacency[node]):
                neighbor = int(neighbor)
                if neighbor not in reached:
                    reached.add(neighbor)
                    frontier.append(neighbor)
        connected = len(reached) == robot_count
        connected_count += int(connected)
        participating = np.any(adjacency & ~np.eye(robot_count, dtype=bool), axis=1)
        participation_counts += participating.astype(np.int64)
        shared_count += int(bool(edges))
        isolated_runs = np.where(participating, 0, isolated_runs + 1)
        maximum_isolated_runs = np.maximum(maximum_isolated_runs, isolated_runs)
        keyframes.append({
            "index": frame_index,
            "edges": edges,
            "overlap_by_pair": overlaps,
            "connected": connected,
            "isolated_robots": np.flatnonzero(~participating).astype(int).tolist(),
        })

    reached = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for neighbor in np.flatnonzero(union_adjacency[node]):
            neighbor = int(neighbor)
            if neighbor not in reached:
                reached.add(neighbor)
                frontier.append(neighbor)
    union_connected = len(reached) == robot_count
    participates = np.any(
        union_adjacency & ~np.eye(robot_count, dtype=bool), axis=1
    )
    allowed_isolated = int(np.floor(maximum_isolated_fraction * frame_count))
    isolation_ok = bool(np.all(maximum_isolated_runs <= allowed_isolated))
    passed = bool(union_connected and np.all(participates) and isolation_ok)
    connected_fraction = connected_count / frame_count
    shared_fraction = shared_count / frame_count
    minimum_participation_fraction = float(
        np.min(participation_counts) / frame_count
    )
    mean_overlap = float(np.mean(pair_values)) if pair_values else 0.0
    # Saturate overlap at the edge threshold: once an edge is useful, route
    # diversity and temporal connectivity matter more than maximizing overlap.
    useful_overlap = min(mean_overlap / max(edge_threshold, 1e-9), 1.0)
    worst_isolated_fraction = float(np.max(maximum_isolated_runs) / frame_count)
    score = (
        1.5 * connected_fraction
        + float(union_connected)
        + shared_fraction
        + 1.5 * minimum_participation_fraction
        + 0.5 * useful_overlap
        - worst_isolated_fraction
        - (0.0 if passed else 2.0)
    )
    failure_reasons = []
    if not union_connected:
        failure_reasons.append("union_graph_disconnected")
    if not np.all(participates):
        failure_reasons.append("robot_never_participates")
    if not isolation_ok:
        failure_reasons.append("excessive_isolation")
    return {
        "passed": passed,
        "score": float(score),
        "failure_reasons": failure_reasons,
        "edge_threshold": float(edge_threshold),
        "keyframe_count": frame_count,
        "connected_keyframe_fraction": float(connected_fraction),
        "shared_keyframe_fraction": float(shared_fraction),
        "robot_participation_counts": participation_counts.astype(int).tolist(),
        "minimum_robot_participation_fraction": minimum_participation_fraction,
        "union_graph_connected": union_connected,
        "robot_participates": participates.tolist(),
        "maximum_consecutive_isolated_keyframes": maximum_isolated_runs.astype(int).tolist(),
        "allowed_consecutive_isolated_keyframes": allowed_isolated,
        "mean_pair_overlap": mean_overlap,
        "keyframes": keyframes,
    }


def _cheap_visibility_score_from_pairwise(
    selected_indices: tuple[int, ...],
    overlap_by_keyframe: np.ndarray,
    *,
    edge_threshold: float,
    maximum_isolated_fraction: float,
) -> float:
    # Fast soft score over precomputed route-pair overlap values.
    robot_count = len(selected_indices)
    frame_count = (
        int(overlap_by_keyframe.shape[2])
        if overlap_by_keyframe.ndim == 3
        else 0
    )
    if robot_count < 2 or frame_count == 0:
        return -1.0e6
    index = np.asarray(selected_indices, dtype=np.int64)
    pairwise = overlap_by_keyframe[index[:, None], index[None, :], :]
    identity = np.eye(robot_count, dtype=bool)
    union_adjacency = identity.copy()
    isolated_runs = np.zeros(robot_count, dtype=np.int64)
    maximum_isolated_runs = np.zeros(robot_count, dtype=np.int64)
    participation_counts = np.zeros(robot_count, dtype=np.int64)
    connected_count = 0
    shared_count = 0
    upper = np.triu_indices(robot_count, 1)
    for frame_index in range(frame_count):
        adjacency = pairwise[:, :, frame_index] >= edge_threshold
        union_adjacency |= adjacency
        reached = {0}
        frontier = [0]
        while frontier:
            node = frontier.pop()
            for neighbor in np.flatnonzero(adjacency[node]):
                neighbor = int(neighbor)
                if neighbor not in reached:
                    reached.add(neighbor)
                    frontier.append(neighbor)
        connected_count += int(len(reached) == robot_count)
        participating = np.any(adjacency & ~identity, axis=1)
        participation_counts += participating.astype(np.int64)
        shared_count += int(np.any(adjacency[upper]))
        isolated_runs = np.where(participating, 0, isolated_runs + 1)
        maximum_isolated_runs = np.maximum(
            maximum_isolated_runs, isolated_runs
        )

    reached = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for neighbor in np.flatnonzero(union_adjacency[node]):
            neighbor = int(neighbor)
            if neighbor not in reached:
                reached.add(neighbor)
                frontier.append(neighbor)
    participates = np.any(union_adjacency & ~identity, axis=1)
    allowed_isolated = int(np.floor(
        maximum_isolated_fraction * frame_count
    ))
    passed = bool(
        len(reached) == robot_count
        and np.all(participates)
        and np.all(maximum_isolated_runs <= allowed_isolated)
    )
    mean_overlap = float(np.mean(pairwise[upper]))
    useful_overlap = min(mean_overlap / max(edge_threshold, 1e-9), 1.0)
    return float(
        1.5 * connected_count / frame_count
        + 1.0
        + shared_count / frame_count
        + 1.5 * float(np.min(participation_counts)) / frame_count
        + 0.5 * useful_overlap
        - float(np.max(maximum_isolated_runs)) / frame_count
        - (0.0 if passed else 2.0)
    )


def _cheap_visibility_scorer(
    routes: tuple[RouteCandidate, ...],
    overlap_by_keyframe: np.ndarray,
    *,
    edge_threshold: float,
    maximum_isolated_fraction: float,
):
    route_indices = {
        route.route_id: index for index, route in enumerate(routes)
    }

    def score(selected: tuple[RouteCandidate, ...]) -> float:
        indices = tuple(route_indices[route.route_id] for route in selected)
        return _cheap_visibility_score_from_pairwise(
            indices,
            overlap_by_keyframe,
            edge_threshold=edge_threshold,
            maximum_isolated_fraction=maximum_isolated_fraction,
        )

    return score

def _minimum_route_footprint_clearance_m(
    adapter: Any,
    safe_masks: np.ndarray,
    trajectory: Trajectory,
    safety_margin_m: float,
) -> float:
    """Measure configuration-space clearance beyond the expanded footprint."""
    positions = trajectory.base_to_world[:, :2, 3]
    pixels = _map_points(adapter, positions)
    yaws = np.arctan2(
        trajectory.base_to_world[:, 1, 0],
        trajectory.base_to_world[:, 0, 0],
    )
    yaw_bins = len(safe_masks)
    bins = np.rint(
        (yaws % (2.0 * np.pi)) * yaw_bins / (2.0 * np.pi)
    ).astype(int) % yaw_bins
    resolution = float(adapter._require_scene().trav_map.map_resolution)
    minimum_pixels = np.inf
    for yaw_index in np.unique(bins):
        route_pixels = pixels[bins == yaw_index].astype(np.float64)
        unsafe = np.argwhere(~safe_masks[int(yaw_index)]).astype(np.float64)
        if not len(unsafe):
            continue
        for start in range(0, len(unsafe), 1024):
            distances = np.linalg.norm(
                route_pixels[:, None, :] - unsafe[None, start: start + 1024, :],
                axis=2,
            )
            minimum_pixels = min(minimum_pixels, float(np.min(distances)))
    if not np.isfinite(minimum_pixels):
        return float("inf")
    # Cell centres one pixel apart share a boundary halfway between them. The
    # footprint itself was expanded by safety_margin_m before rasterization.
    return float(safety_margin_m + max(0.0, minimum_pixels - 0.5) * resolution)



def _build_floor_context(
    adapter: Any,
    footprint: Any,
    floor_index: int,
    seed: int,
) -> NavigationContext:
    started = time.perf_counter()
    scene = adapter._require_scene()
    trav_map = scene.trav_map
    navigation = adapter.config["navigation"]
    trajectory_config = adapter.config["trajectory"]
    point_free = _point_free_mask(adapter, floor_index)
    offsets = _footprint_offsets(adapter, footprint, floor_index)
    safe_masks = oriented_safe_masks(point_free, offsets)
    # Pose validity is defined by safe_masks[yaw,row,column]. The union is only
    # a 2-D visualization / seed surface; yaw freedom is a soft A* ranking term
    # and must never delete a genuinely feasible SE(2) pose.
    permissive_mask = np.any(safe_masks, axis=0)
    yaw_freedom = np.mean(safe_masks, axis=0)
    seed_mask = permissive_mask.copy()
    planner_mask = permissive_mask.copy()
    component_labels, component_sizes = connected_components(planner_mask)
    pixels = np.argwhere(planner_mask)
    if not len(pixels):
        raise SampleRejected(
            "navigation_no_footprint_safe_cells", {"floor_index": floor_index}
        )
    world_xy = adapter._native_value(
        trav_map.map_to_world(adapter._th.as_tensor(pixels, dtype=adapter._th.int64))
    ).astype(np.float64)
    world_labels = adapter._observation_region_labels(floor_index, world_xy)
    label_grid = np.full(planner_mask.shape, "", dtype=object)
    label_grid[pixels[:, 0], pixels[:, 1]] = world_labels
    region_graph, label_grid = build_region_graph(planner_mask, label_grid)
    rng = np.random.default_rng(seed)
    region_pixels = {
        region: np.argwhere(seed_mask & (label_grid == region))
        for region in region_graph.nodes
    }
    route_counts_by_start: Counter[str] = Counter()
    reject_counts: Counter[str] = Counter()
    routes: list[RouteCandidate] = []
    signatures: set[tuple[int, int, int, int]] = set()
    target_size = int(navigation["route_bank_target_size"])
    maximum_attempts = int(navigation["route_bank_max_raw_attempts"])
    attempts_per_raw = int(navigation["route_candidate_attempts_per_raw"])
    floor_z = float(scene.get_floor_height(floor_index))
    families, family_probabilities = _normalised_family_distribution(
        trajectory_config["path_family_weights"]
    )
    for raw_attempt in range(maximum_attempts):
        if len(routes) >= target_size:
            break
        eligible_regions = [name for name, values in region_pixels.items() if len(values)]
        if not eligible_regions:
            break
        weights = np.asarray([
            1.0 / (1.0 + route_counts_by_start[name]) for name in eligible_regions
        ])
        start_region = str(rng.choice(eligible_regions, p=weights / weights.sum()))
        start_pixel = region_pixels[start_region][
            int(rng.integers(len(region_pixels[start_region])))
        ]
        component = int(component_labels[tuple(start_pixel)])
        component_pixels = np.argwhere(
            (component_labels == component) & seed_mask
        )
        if len(component_pixels) < 2:
            reject_counts["component_too_small"] += 1
            continue
        candidates = adapter._native_value(
            trav_map.map_to_world(
                adapter._th.as_tensor(component_pixels, dtype=adapter._th.int64)
            )
        ).astype(np.float64)
        start_xy = adapter._native_value(
            trav_map.map_to_world(adapter._th.as_tensor(start_pixel, dtype=adapter._th.int64))
        ).astype(np.float64)
        route_seed = stable_seed(seed, "route", raw_attempt)
        route_rng = np.random.default_rng(route_seed)
        trajectory = None
        for _ in range(attempts_per_raw):
            family = str(route_rng.choice(families, p=family_probabilities))
            controls = _sample_route_controls(
                start_xy,
                0.0,
                candidates,
                family,
                float(trajectory_config["path_length_min_m"]),
                float(trajectory_config["path_length_max_m"]),
                np.pi,
                np.deg2rad(float(trajectory_config["maximum_control_turn_deg"])),
                route_rng,
                soft_initial_heading=True,
                initial_heading_probability_floor=1.0,
            )
            if controls is None:
                reject_counts["se2_no_control_candidates"] += 1
                continue
            result = _se2_route_candidate(
                adapter,
                safe_masks,
                controls,
                robot_id=f"route_{raw_attempt:04d}",
                floor_z=floor_z,
                camera_mount=np.eye(4, dtype=np.float64),
                path_family=family,
            )
            if result is None:
                reject_counts["se2_plan_or_time_parameterization"] += 1
                continue
            candidate, _ = result
            arc_length = float(candidate.metadata["smoothed_arc_length_m"])
            if not (
                float(trajectory_config["path_length_min_m"])
                <= arc_length
                <= float(trajectory_config["path_length_max_m"])
            ):
                reject_counts["se2_arc_length"] += 1
                continue
            trajectory = candidate
            break
        if trajectory is None:
            continue
        goal_pixel = _map_points(adapter, trajectory.base_to_world[-1:, :2, 3])[0]
        signature = (
            int(start_pixel[0]), int(start_pixel[1]), int(goal_pixel[0]), int(goal_pixel[1])
        )
        if signature in signatures:
            reject_counts["duplicate_route"] += 1
            continue
        signatures.add(signature)
        route = route_candidate_from_trajectory(
            f"route_{len(routes):04d}",
            floor_index,
            replace(trajectory, robot_id=f"route_{len(routes):04d}"),
            route_seed,
            lambda points: _region_labels_for_points(adapter, label_grid, points),
            minimum_footprint_clearance_m=_minimum_route_footprint_clearance_m(
                adapter,
                safe_masks,
                trajectory,
                float(navigation["footprint_safety_margin_m"]),
            ),
        )
        routes.append(route)
        route_counts_by_start[route.start_region] += 1
    compatibility = compute_pairwise_route_compatibility(
        routes,
        minimum_pairwise_distance_m=float(
            adapter.config["placement"]["minimum_pairwise_distance_m"]
        ),
        footprint=footprint,
    )
    (
        cheap_visible_cells,
        cheap_view_similarity,
        cheap_view_overlap_by_keyframe,
    ) = _cheap_scene_view_cache(adapter, point_free, routes)
    compatibility = replace(compatibility, cheap_view_similarity=cheap_view_similarity)
    diagnostics = {
        "raw_route_attempts": min(maximum_attempts, raw_attempt + 1 if maximum_attempts else 0),
        "accepted_route_count": len(routes),
        "route_acceptance_rate": len(routes) / max(1, min(maximum_attempts, raw_attempt + 1)),
        "route_reject_counts": dict(reject_counts),
        "route_generation_seconds": time.perf_counter() - started,
        "start_region_distribution": dict(route_counts_by_start),
        "old_omnigibson_eroded_cell_count": int(np.count_nonzero(
            adapter._native_value(
                adapter._robot_eroded_traversability(floor_index, adapter._env.robots[0])
            ) == 255
        )),
        "point_free_cell_count": int(np.count_nonzero(point_free)),
        "permissive_footprint_cell_count": int(np.count_nonzero(permissive_mask)),
        "footprint_safe_cell_count": int(np.count_nonzero(planner_mask)),
        "route_seed_cell_count": int(np.count_nonzero(seed_mask)),
    }
    return NavigationContext(
        floor_index=floor_index,
        map_resolution_m=float(trav_map.map_resolution),
        point_free_mask=point_free,
        footprint_safe_masks=safe_masks,
        planner_mask=planner_mask,
        component_labels=component_labels,
        component_sizes=component_sizes,
        region_label_grid=label_grid,
        region_graph=region_graph,
        footprint=footprint,
        route_bank=tuple(routes),
        compatibility=compatibility,
        cheap_visible_cells=cheap_visible_cells,
        cheap_view_overlap_by_keyframe=cheap_view_overlap_by_keyframe,
        diagnostics=diagnostics,
    )


def build_navigation_contexts(
    adapter: Any,
    configuration_token: str,
    seed: int,
    *,
    force: bool = False,
) -> dict[int, NavigationContext]:
    """Build and cache all configuration-level navigation products once."""
    if (
        not force
        and adapter._navigation_configuration_token == configuration_token
        and adapter._navigation_contexts
    ):
        return adapter._navigation_contexts
    footprint = _robot_footprint(adapter)
    contexts: dict[int, NavigationContext] = {}
    failures: dict[int, dict[str, Any]] = {}
    route_count_target = int(adapter.config["navigation"]["route_bank_target_size"])
    for floor_index in range(int(adapter._require_scene().n_floors)):
        try:
            context = _build_floor_context(
                adapter,
                footprint,
                floor_index,
                stable_seed(seed, "navigation-floor", floor_index),
            )
            context.diagnostics["route_count_target"] = route_count_target
            context.diagnostics["route_count_target_met"] = bool(
                len(context.route_bank) >= route_count_target
            )
            probe = select_joint_route_candidates(
                context.route_bank,
                context.compatibility,
                np.random.default_rng(stable_seed(seed, "joint-feasibility", floor_index)),
                top_k=1,
                search_budget=int(adapter.config["navigation"]["joint_route_search_budget"]),
                minimum_waypoint_trajectories=0,
                region_graph=(
                    context.region_graph
                    if adapter.config["navigation"][
                        "require_connected_start_regions"
                    ]
                    else None
                ),
            )
            if not probe:
                raise SampleRejected(
                    "navigation_no_compatible_route_triplet",
                    {
                        "floor_index": floor_index,
                        "route_count": len(context.route_bank),
                        "compatible_pair_fraction": (
                            context.compatibility.compatible_pair_fraction
                        ),
                    },
                )
            contexts[floor_index] = context
        except SampleRejected as error:
            failures[floor_index] = {"reason": error.reason, "details": error.details}
    if not contexts:
        raise SampleRejected(
            "configuration_navigation_infeasible", {"floor_failures": failures}
        )
    adapter._navigation_contexts = contexts
    adapter._navigation_configuration_token = str(configuration_token)
    adapter._runtime_findings["navigation_context"] = {
        "configuration_token": str(configuration_token),
        "feasible_floors": sorted(contexts),
        "floor_failures": failures,
        "robot_footprint": footprint.metadata(),
        "floors": {
            str(index): context.metadata(include_routes=False)
            for index, context in contexts.items()
        },
    }
    return contexts


def _configure_episode_cameras(adapter: Any, rng: np.random.Generator) -> dict[str, float]:
    robots = sorted(adapter._env.robots, key=lambda robot: robot.name)
    if len(robots) != 3:
        raise SimulatorUnavailableError("Dataset v1 requires exactly three robots")
    heights = list(map(float, adapter.config["camera"]["heights_m"]))
    pitch = np.deg2rad(float(adapter.config["camera"]["pitch_deg"]))
    cosine, sine = np.cos(pitch), np.sin(pitch)
    camera_rotation = np.asarray(
        [[0.0, sine, cosine], [-1.0, 0.0, 0.0], [0.0, -cosine, sine]],
        dtype=np.float64,
    )
    sampled: dict[str, float] = {}
    adapter._development_camera_mounts.clear()
    for robot in robots:
        height = float(rng.choice(heights))
        sampled[robot.name] = height
        if adapter._using_final_robot:
            mast_joint = next(
                joint
                for name, joint in robot.joints.items()
                if name.endswith("mvwd_mast_joint")
            )
            mast_joint.set_pos(height - min(heights), drive=False)
        camera_to_base = np.eye(4, dtype=np.float64)
        camera_to_base[:3, :3] = camera_rotation
        camera_to_base[:3, 3] = [
            0.08 if adapter._using_final_robot else 0.0,
            0.0,
            height,
        ]
        adapter._development_camera_mounts[robot.name] = camera_to_base
    adapter._runtime_findings["development_camera_heights_m"] = sampled
    return sampled


def _bind_route(
    route: RouteCandidate,
    robot_id: str,
    camera_to_base: np.ndarray,
) -> Trajectory:
    base = route.trajectory.base_to_world.copy()
    cameras = np.stack([pose @ camera_to_base for pose in base])
    return Trajectory(
        robot_id=robot_id,
        fps=route.trajectory.fps,
        base_to_world=base,
        camera_to_world=cameras,
        path_family=route.trajectory.path_family,
        control_waypoints_xy=route.trajectory.control_waypoints_xy.copy(),
        planner_path_xy=route.trajectory.planner_path_xy.copy(),
        smoothed_path_xy=route.trajectory.smoothed_path_xy.copy(),
        simplified_path_xy=route.trajectory.simplified_path_xy.copy(),
        metadata={
            **route.trajectory.metadata,
            "route_id": route.route_id,
            "route_seed": route.route_seed,
            "start_region": route.start_region,
            "goal_region": route.goal_region,
            "traversed_regions": list(route.traversed_regions),
            "minimum_footprint_clearance_m": route.minimum_footprint_clearance_m,
            "observation_regime": "unclassified",
            "requested_observation_regime": "unclassified",
        },
    )


def _start_blacklist_key(adapter: Any, trajectory: Trajectory) -> tuple[int, int, int]:
    position_quantization = float(
        adapter.config["navigation"]["start_blacklist_position_quantization_m"]
    )
    yaw_bins = int(adapter.config["navigation"]["start_blacklist_yaw_bins"])
    pose = trajectory.base_to_world[0]
    yaw = float(np.arctan2(pose[1, 0], pose[0, 0]))
    return (
        int(np.rint(pose[0, 3] / position_quantization)),
        int(np.rint(pose[1, 3] / position_quantization)),
        int(np.rint((yaw % (2.0 * np.pi)) * yaw_bins / (2.0 * np.pi))) % yaw_bins,
    )


def _sparse_physics_preflight(
    adapter: Any,
    context: NavigationContext,
    trajectories: tuple[Trajectory, ...],
) -> None:
    by_id = {trajectory.robot_id: trajectory for trajectory in trajectories}
    robots = {robot.name: robot for robot in adapter._env.robots}
    floors = adapter._robot_support_surfaces()
    configured = adapter.config["navigation"]["sparse_physics_keyframes"]
    frames = trajectories[0].frames
    keyframes = sorted({min(frames - 1, int(frame)) for frame in configured})
    for frame_index in keyframes:
        # Sparse physics must remain cheaper than any image preflight. Set the
        # three synchronized poses and advance PhysX once; camera render refresh
        # belongs to the later GT-depth stage.
        for robot_id, robot in robots.items():
            planned = by_id[robot_id].base_to_world[frame_index]
            position, orientation = adapter._transform_utils.mat2pose(
                adapter._th.as_tensor(planned, dtype=adapter._th.float32)
            )
            robot.set_position_orientation(
                position=position, orientation=orientation
            )
            adapter._restore_final_robot_mast_mount(robot)
            robot.keep_still()
        adapter._og.sim.step_physics()
        for robot_id, robot in robots.items():
            pairs = adapter._external_robot_contact_pairs(robot, floors)
            if not pairs:
                continue
            key = _start_blacklist_key(adapter, by_id[robot_id])
            if frame_index == 0:
                context.invalid_start_pose_blacklist.add(key)
            raise SampleRejected(
                "route_triplet_sparse_physics_collision",
                {
                    "stage": "start_physx" if frame_index == 0 else "sparse_physx",
                    "frame_index": frame_index,
                    "robot_id": robot_id,
                    "route_id": by_id[robot_id].metadata["route_id"],
                    "contact_pairs": pairs[:50],
                    "contact_pose": by_id[robot_id].base_to_world[frame_index].tolist(),
                },
            )


def footprint_physx_diagnostic(
    adapter: Any,
    floor_index: int,
    seed: int,
) -> dict[str, Any]:
    """Compare orientation-aware raster predictions with exact current contacts."""
    context = adapter._navigation_contexts[floor_index]
    robots = sorted(adapter._env.robots, key=lambda robot: robot.name)
    robot = robots[0]
    floors = adapter._robot_support_surfaces()
    saved_poses = [adapter._pose_matrix(item) for item in robots]
    peer_xy = np.asarray([pose[:2, 3] for pose in saved_poses[1:]])
    peer_exclusion_radius_m = (
        2.0 * context.footprint.circumscribed_radius_m
        + context.footprint.safety_margin_m
    )
    free = context.point_free_mask
    near_free = np.zeros_like(free)
    for row_delta, column_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        near_free[max(0, row_delta): free.shape[0] + min(0, row_delta),
                  max(0, column_delta): free.shape[1] + min(0, column_delta)] |= (
            free[max(0, -row_delta): free.shape[0] - max(0, row_delta),
                 max(0, -column_delta): free.shape[1] - max(0, column_delta)]
        )
    yaw_bins = len(context.footprint_safe_masks)
    any_yaw = np.any(context.footprint_safe_masks, axis=0)
    yaw_freedom = np.mean(context.footprint_safe_masks, axis=0)
    boundary = free & ~any_yaw
    # Separate movable-furniture proximity from static wall proximity using
    # the same current-state AABBs used to construct the point map.
    all_pixels = np.indices(free.shape).reshape(2, -1).T
    world_grid = adapter._native_value(
        adapter._require_scene().trav_map.map_to_world(
            adapter._th.as_tensor(all_pixels, dtype=adapter._th.int64)
        )
    ).astype(np.float64)
    furniture_near = np.zeros(len(all_pixels), dtype=bool)
    radius = context.footprint.circumscribed_radius_m
    for obj in adapter.object_catalog():
        if obj.structural:
            continue
        low = np.asarray(obj.bbox_min_world[:2], dtype=np.float64) - radius
        high = np.asarray(obj.bbox_max_world[:2], dtype=np.float64) + radius
        furniture_near |= np.all(
            (world_grid[:, :2] >= low) & (world_grid[:, :2] <= high), axis=1
        )
    furniture_near = furniture_near.reshape(free.shape)
    blocked_neighbors = np.zeros(free.shape, dtype=np.int64)
    for row_delta, column_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted = np.zeros_like(free)
        shifted[max(0, row_delta): free.shape[0] + min(0, row_delta),
                max(0, column_delta): free.shape[1] + min(0, column_delta)] = (
            ~free[max(0, -row_delta): free.shape[0] - max(0, row_delta),
                  max(0, -column_delta): free.shape[1] - max(0, column_delta)]
        )
        blocked_neighbors += shifted
    connector = np.isin(
        context.region_label_grid,
        np.asarray(context.region_graph.connector_nodes, dtype=object),
    )
    category_masks = {
        "open": np.broadcast_to(np.all(context.footprint_safe_masks, axis=0), context.footprint_safe_masks.shape),
        "wall": np.broadcast_to(boundary & ~furniture_near, context.footprint_safe_masks.shape),
        "furniture": np.broadcast_to(boundary & furniture_near, context.footprint_safe_masks.shape),
        "corridor": np.broadcast_to(any_yaw & (yaw_freedom <= 0.50), context.footprint_safe_masks.shape),
        "door": np.broadcast_to(any_yaw & connector, context.footprint_safe_masks.shape),
        "corner": np.broadcast_to(boundary & (blocked_neighbors >= 2), context.footprint_safe_masks.shape),
    }
    triples: dict[str, np.ndarray] = {
        name: np.argwhere(mask) for name, mask in category_masks.items()
    }
    rng = np.random.default_rng(seed)
    requested = int(
        adapter.config["navigation"]["footprint_physx_probe_count_per_category"]
    )
    trav_map = adapter._require_scene().trav_map
    floor_z = float(adapter._require_scene().get_floor_height(floor_index))
    records: list[dict[str, Any]] = []
    try:
        for category, values in triples.items():
            accepted = 0
            for yaw_index, row, column in values[rng.permutation(len(values))]:
                pixel = np.asarray([row, column], dtype=np.int64)
                xy = adapter._native_value(
                    trav_map.map_to_world(adapter._th.as_tensor(pixel))
                ).astype(np.float64)
                if len(peer_xy) and np.min(np.linalg.norm(peer_xy - xy, axis=1)) < peer_exclusion_radius_m:
                    continue
                yaw = 2.0 * np.pi * int(yaw_index) / yaw_bins
                pose = np.eye(4, dtype=np.float64)
                cosine, sine = np.cos(yaw), np.sin(yaw)
                pose[:2, :2] = ((cosine, -sine), (sine, cosine))
                pose[:2, 3] = xy
                pose[2, 3] = floor_z
                position, orientation = adapter._transform_utils.mat2pose(
                    adapter._th.as_tensor(pose, dtype=adapter._th.float32)
                )
                robot.set_position_orientation(position=position, orientation=orientation)
                adapter._restore_final_robot_mast_mount(robot)
                robot.keep_still()
                adapter._og.sim.step_physics()
                contacts = adapter._external_robot_contact_pairs(robot, floors)
                raster_safe = bool(
                    context.footprint_safe_masks[int(yaw_index), int(row), int(column)]
                )
                records.append({
                    "category": category,
                    "pixel": [int(row), int(column)],
                    "xy": xy.tolist(),
                    "yaw_bin": int(yaw_index),
                    "yaw_rad": float(yaw),
                    "raster_safe": raster_safe,
                    "physx_safe": not bool(contacts),
                    "contact_pairs": contacts[:20],
                })
                accepted += 1
                if accepted >= requested:
                    break
    finally:
        for item, pose in zip(robots, saved_poses, strict=True):
            position, orientation = adapter._transform_utils.mat2pose(
                adapter._th.as_tensor(pose, dtype=adapter._th.float32)
            )
            item.set_position_orientation(position=position, orientation=orientation)
            adapter._restore_final_robot_mast_mount(item)
            item.keep_still()
        adapter._og.sim.step_physics()
    matches = [item["raster_safe"] == item["physx_safe"] for item in records]
    false_safe = [
        item for item in records if item["raster_safe"] and not item["physx_safe"]
    ]
    conservative = [
        item for item in records if not item["raster_safe"] and item["physx_safe"]
    ]
    by_category = {}
    for category in triples:
        subset = [item for item in records if item["category"] == category]
        by_category[category] = {
            "count": len(subset),
            "agreement_fraction": float(np.mean([
                item["raster_safe"] == item["physx_safe"] for item in subset
            ])) if subset else None,
            "raster_safe_physx_collision_count": sum(
                item["raster_safe"] and not item["physx_safe"] for item in subset
            ),
            "raster_unsafe_physx_free_count": sum(
                not item["raster_safe"] and item["physx_safe"] for item in subset
            ),
        }
    return {
        "probe_count": len(records),
        "requested_per_category": requested,
        "peer_exclusion_radius_m": float(peer_exclusion_radius_m),
        "agreement_fraction": float(np.mean(matches)) if matches else None,
        "raster_safe_physx_collision_count": len(false_safe),
        "raster_unsafe_physx_free_count": len(conservative),
        "by_category": by_category,
        "records": records,
    }



def sample_route_first_trajectory_sets(
    adapter: Any,
    seed: int,
    *,
    discouraged_region_ids: tuple[str, ...] = (),
) -> tuple[dict[str, float], tuple[tuple[tuple[Trajectory, ...], dict[str, Any]], ...]]:
    if not adapter._navigation_contexts:
        raise SampleRejected("navigation_context_not_prepared")
    rng = np.random.default_rng(seed)
    floor_indices = sorted(adapter._navigation_contexts)
    floor_weights = np.asarray([
        len(adapter._navigation_contexts[index].route_bank) for index in floor_indices
    ], dtype=np.float64)
    floor_index = int(rng.choice(floor_indices, p=floor_weights / floor_weights.sum()))
    context = adapter._navigation_contexts[floor_index]
    heights = _configure_episode_cameras(adapter, rng)
    visibility_scorer = _cheap_visibility_scorer(
        context.route_bank,
        context.cheap_view_overlap_by_keyframe,
        edge_threshold=float(
            adapter.config["navigation"]["cheap_visibility_edge_threshold"]
        ),
        maximum_isolated_fraction=float(
            adapter.config["navigation"][
                "cheap_visibility_maximum_isolated_fraction"
            ]
        ),
    )
    joint_search_diagnostics: dict[str, Any] = {}
    triplets = select_joint_route_candidates(
        context.route_bank,
        context.compatibility,
        rng,
        top_k=max(map(int, adapter.config["navigation"]["exact_validation_batches"])),
        cheap_visibility_score=visibility_scorer,
        visibility_priority_fraction=float(
            adapter.config["navigation"][
                "joint_route_visibility_priority_fraction"
            ]
        ),
        visibility_score_weight=float(
            adapter.config["navigation"]["joint_route_visibility_score_weight"]
        ),
        search_budget=int(adapter.config["navigation"]["joint_route_search_budget"]),
        minimum_waypoint_trajectories=0,
        discouraged_regions=discouraged_region_ids,
        region_graph=(
            context.region_graph
            if adapter.config["navigation"]["require_connected_start_regions"]
            else None
        ),
        diagnostics=joint_search_diagnostics,
    )
    if not triplets:
        raise SampleRejected(
            "joint_route_search_empty",
            {
                "floor_index": floor_index,
                "route_count": len(context.route_bank),
                "compatible_pair_fraction": context.compatibility.compatible_pair_fraction,
            },
        )
    robot_ids = tuple(sorted(robot.name for robot in adapter._env.robots))
    accepted = []
    failures = []
    for candidate_index, indices in enumerate(triplets):
        assignment = tuple(rng.permutation(indices).tolist())
        selected_routes = tuple(context.route_bank[index] for index in assignment)
        trajectories = tuple(
            _bind_route(
                route,
                robot_id,
                adapter._development_camera_mounts[robot_id],
            )
            for robot_id, route in zip(robot_ids, selected_routes, strict=True)
        )
        if any(
            _start_blacklist_key(adapter, trajectory)
            in context.invalid_start_pose_blacklist
            for trajectory in trajectories
        ):
            failures.append({
                "candidate_index": candidate_index,
                "reason": "blacklisted_start_pose",
                "route_ids": [route.route_id for route in selected_routes],
            })
            continue
        try:
            _sparse_physics_preflight(adapter, context, trajectories)
        except SampleRejected as error:
            failures.append({
                "candidate_index": candidate_index,
                "reason": error.reason,
                "details": error.details,
                "route_ids": [route.route_id for route in selected_routes],
            })
            continue
        joint = joint_trajectory_metrics(
            trajectories,
            camera_hfov_deg=float(adapter.config["camera"]["hfov_deg"]),
        )
        metrics = {
            "floor_index": floor_index,
            "observation_regime": "unclassified",
            "requested_observation_regime": "unclassified",
            "route_ids": [route.route_id for route in selected_routes],
            "start_region_ids": [route.start_region for route in selected_routes],
            "goal_region_ids": [route.goal_region for route in selected_routes],
            "traversed_region_ids": {
                robot_id: list(route.traversed_regions)
                for robot_id, route in zip(robot_ids, selected_routes, strict=True)
            },
            "unique_traversed_region_count": len({
                region for route in selected_routes for region in route.traversed_regions
            }),
            "minimum_pairwise_distance_m": float(
                joint["minimum_inter_robot_distance_m"]
            ),
            "joint_diversity": joint,
            "cheap_scene_visibility": _cheap_visibility_metrics(
                selected_routes,
                context.route_bank,
                context.cheap_visible_cells,
                edge_threshold=float(
                    adapter.config["navigation"]["cheap_visibility_edge_threshold"]
                ),
                maximum_isolated_fraction=float(
                    adapter.config["navigation"][
                        "cheap_visibility_maximum_isolated_fraction"
                    ]
                ),
            ),
            "robots": {
                robot_id: route.metadata()
                for robot_id, route in zip(robot_ids, selected_routes, strict=True)
            },
            "navigation_context": {
                "route_bank_size": len(context.route_bank),
                "compatible_pair_fraction": context.compatibility.compatible_pair_fraction,
                "region_graph_nodes": len(context.region_graph.nodes),
                "region_graph_edges": len(context.region_graph.edges),
                "invalid_start_pose_blacklist_size": len(
                    context.invalid_start_pose_blacklist
                ),
            },
            "joint_route_search": {
                "candidate_index": candidate_index,
                "shortlisted_triplet_count": len(triplets),
                "assignment_indices": list(assignment),
                **joint_search_diagnostics,
            },
        }
        family_counts = Counter(
            route.trajectory.path_family for route in selected_routes
        )
        metrics["route_family_diversity"] = {
            "counts": dict(family_counts),
            "unique_family_count": len(family_counts),
            "waypoint_route_count": sum(
                route.trajectory.path_family != "direct"
                for route in selected_routes
            ),
            "hard_minimum_enforced": False,
        }
        metrics["joint_route_search"]["cheap_proxy_passed"] = bool(
            metrics["cheap_scene_visibility"].get("passed", False)
        )
        metrics["joint_route_search"]["proxy_fallback_to_exact_gt"] = bool(
            not metrics["cheap_scene_visibility"].get("passed", False)
        )
        accepted.append((trajectories, metrics))
    if not accepted:
        raise SampleRejected(
            "route_triplet_exact_preflight_exhausted",
            {
                "floor_index": floor_index,
                "shortlisted_triplets": len(triplets),
                "failures": failures,

                "blacklist_size": len(context.invalid_start_pose_blacklist),
            },
        )
    sparse_failure_counts = Counter(
        str(record["reason"]) for record in failures
    )
    for _, metrics in accepted:
        metrics["joint_route_search"].update({
            "sparse_physics_candidate_failure_counts": dict(sparse_failure_counts),
            "sparse_physics_candidate_failure_count": len(failures),
            "sparse_physics_candidate_failures": failures,
        })
    adapter._runtime_findings["route_first_episode_sampling"] = {
        "floor_index": floor_index,
        "shortlisted_triplets": len(triplets),
        **joint_search_diagnostics,
        "sparse_physx_accepted": len(accepted),
        "failures": failures,
    }
    adapter._runtime_findings["sampled_floor_index"] = floor_index
    return heights, tuple(accepted)


def measured_overlap_route_mutations(
    adapter: Any,
    candidate_sets: tuple[tuple[tuple[Trajectory, ...], dict[str, Any]], ...],
    candidate_failures: list[dict[str, Any]],
    *,
    maximum_candidates: int,
) -> tuple[tuple[tuple[Trajectory, ...], dict[str, Any]], ...]:
    """Repair GT-measured isolation with bounded native RouteBank mutations.

    Keep both independently planned routes on a measured overlap edge and
    replace only the third robot's route. Every mutation remains subject to
    RouteBank footprint compatibility, formation checks, sparse PhysX, and the
    caller's exact GT-depth validation. The cheap cache only ranks mutations.
    """
    if maximum_candidates <= 0 or not candidate_sets:
        return ()
    failures = {
        int(item["candidate_rank"]): item
        for item in candidate_failures
        if isinstance(item.get("candidate_rank"), int)
        and item.get("candidate_kind", "base")
        in {"base", "measured_overlap_route_mutation"}
        and item.get("reason") == "trajectory_temporal_overlap_failed"
    }
    robot_ids = tuple(sorted(item.robot_id for item in candidate_sets[0][0]))
    if not failures or len(robot_ids) != 3:
        return ()
    floor_index = int(candidate_sets[0][1]["floor_index"])
    context = adapter._navigation_contexts[floor_index]
    routes = context.route_bank
    route_index = {route.route_id: index for index, route in enumerate(routes)}
    seen = {
        tuple(map(str, metrics.get("route_ids", ())))
        for _, metrics in candidate_sets
        if len(metrics.get("route_ids", ())) == len(robot_ids)
    }
    edge_threshold = float(
        adapter.config["navigation"]["cheap_visibility_edge_threshold"]
    )
    maximum_isolated_fraction = float(
        adapter.config["navigation"][
            "cheap_visibility_maximum_isolated_fraction"
        ]
    )
    proposals: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    duplicate_triplets_removed = 0
    for source_rank, (_, source_metrics) in enumerate(candidate_sets):
        failure = failures.get(source_rank)
        if failure is None:
            continue
        details = failure.get("details", {})
        edges = {
            tuple(sorted(map(str, edge)))
            for edge in details.get("union_edges", ())
            if len(edge) == 2
        }
        isolation = {
            str(key): int(value)
            for key, value in details.get(
                "maximum_consecutive_isolated_keyframes", {}
            ).items()
        }
        allowed = int(details.get("allowed_consecutive_isolated_keyframes", 0))
        isolation_intervals = details.get("longest_isolation_intervals", {})
        measured_keyframes = tuple(details.get("keyframes", ()))
        source_ids = tuple(map(str, source_metrics.get("route_ids", ())))
        if len(source_ids) != len(robot_ids):
            continue
        try:
            source_indices = tuple(route_index[value] for value in source_ids)
        except KeyError:
            continue
        for isolated_id in sorted(
            robot_ids, key=lambda key: (-isolation.get(key, 0), key)
        ):
            isolated_index = robot_ids.index(isolated_id)
            preserved_edge = tuple(sorted(
                key for key in robot_ids if key != isolated_id
            ))
            if preserved_edge not in edges:
                continue
            for replacement_index, replacement in enumerate(routes):
                assignment = list(source_indices)
                assignment[isolated_index] = replacement_index
                assignment = tuple(assignment)
                assignment_ids = tuple(routes[index].route_id for index in assignment)
                if len(set(assignment)) != len(assignment):
                    continue
                if assignment_ids in seen:
                    duplicate_triplets_removed += 1
                    continue
                if any(
                    not bool(context.compatibility.compatible[assignment[left], assignment[right]])
                    for left in range(len(robot_ids))
                    for right in range(left + 1, len(robot_ids))
                ):
                    continue
                selected_routes = tuple(routes[index] for index in assignment)
                cheap = _cheap_visibility_metrics(
                    selected_routes,
                    routes,
                    context.cheap_visible_cells,
                    edge_threshold=edge_threshold,
                    maximum_isolated_fraction=maximum_isolated_fraction,
                )
                trajectories = tuple(
                    _bind_route(
                        route, robot_id,
                        adapter._development_camera_mounts[robot_id],
                    )
                    for robot_id, route in zip(
                        robot_ids, selected_routes, strict=True
                    )
                )
                joint = joint_trajectory_metrics(
                    trajectories,
                    camera_hfov_deg=float(adapter.config["camera"]["hfov_deg"]),
                )
                if formation_degenerate(
                    joint,
                    adapter.config["placement"]["formation_degeneracy"],
                ):
                    continue
                seen.add(assignment_ids)
                cheap_isolation = max(map(
                    int,
                    cheap.get("maximum_consecutive_isolated_keyframes", [10**6]),
                ))
                repaired_participation = int(
                    cheap.get("robot_participation_counts", [0] * len(robot_ids))[
                        isolated_index
                    ]
                )
                interval = dict(isolation_intervals.get(isolated_id, {}))
                frame_start = interval.get("frame_start")
                frame_end = interval.get("frame_end")
                interval_proxy_indices: list[int] = []
                if frame_start is not None and frame_end is not None:
                    denominator = max(1, int(candidate_sets[0][0][0].frames) - 1)
                    cheap_count = max(1, int(cheap.get("keyframe_count", 1)))
                    interval_proxy_indices = [
                        index for index in range(cheap_count)
                        if int(frame_start) <= round(index * denominator / max(1, cheap_count - 1))
                        <= int(frame_end)
                    ]
                interval_anchor_count = 0
                for proxy_index in interval_proxy_indices:
                    proxy_edges = cheap.get("keyframes", [])[proxy_index].get("edges", ())
                    interval_anchor_count += int(any(
                        isolated_index in edge for edge in proxy_edges
                    ))
                source_shared_centroids = {
                    pair: centroid
                    for frame in measured_keyframes
                    if (
                        frame_start is None
                        or int(frame_start) <= int(frame.get("frame_index", -1))
                        <= int(frame_end)
                    )
                    for pair, centroid in frame.get(
                        "shared_surface_centroids_world", {}
                    ).items()
                }
                priority = (
                    0 if cheap.get("union_graph_connected", False) else 1,
                    sum(
                        not bool(value)
                        for value in cheap.get("robot_participates", ())
                    ),
                    cheap_isolation,
                    max(0, max(isolation.values(), default=allowed) - allowed),
                    -repaired_participation,
                    -interval_anchor_count,
                    -float(cheap.get("shared_keyframe_fraction", 0.0)),
                    -float(cheap.get("connected_keyframe_fraction", 0.0)),
                    -float(cheap.get("score", -1.0e6)),
                    source_rank,
                    isolated_id,
                    replacement.route_id,
                )
                evidence = {
                    "strategy": "measured_overlap_route_mutation",
                    "source_candidate_rank": source_rank,
                    "preserved_measured_edge": list(preserved_edge),
                    "isolated_robot_id": isolated_id,
                    "replaced_route_id": source_ids[isolated_index],
                    "replacement_route_id": replacement.route_id,
                    "source_maximum_consecutive_isolated_keyframes": isolation,
                    "source_allowed_consecutive_isolated_keyframes": allowed,
                    "repaired_robot_id": isolated_id,
                    "isolated_interval": interval,
                    "overlap_target_pair": list(preserved_edge),
                    "target_shared_surfaces_world": source_shared_centroids,
                    "cheap_interval_anchor_count": interval_anchor_count,
                    "cheap_repaired_robot_participation_count": (
                        repaired_participation
                    ),
                    "cheap_mutation_preflight": cheap,
                }
                proposals.append((priority, {
                    "source_rank": source_rank,
                    "isolated_id": isolated_id,
                    "source_metrics": source_metrics,
                    "selected_routes": selected_routes,
                    "assignment": assignment,
                    "trajectories": trajectories,
                    "joint": joint,
                    "cheap": cheap,
                    "evidence": evidence,
                }))

    proposals.sort(key=lambda item: item[0])
    ordered: list[dict[str, Any]] = []
    selected_targets: set[tuple[int, str]] = set()
    for _, proposal in proposals:
        target = (int(proposal["source_rank"]), str(proposal["isolated_id"]))
        if target not in selected_targets:
            selected_targets.add(target)
            ordered.append(proposal)
    selected_proposal_ids = {id(proposal) for proposal in ordered}
    ordered.extend(
        proposal
        for _, proposal in proposals
        if id(proposal) not in selected_proposal_ids
    )

    accepted: list[tuple[tuple[Trajectory, ...], dict[str, Any]]] = []
    physics_failures: list[dict[str, Any]] = []
    for proposal in ordered:
        trajectories = proposal["trajectories"]
        if any(
            _start_blacklist_key(adapter, trajectory)
            in context.invalid_start_pose_blacklist
            for trajectory in trajectories
        ):
            continue
        try:
            _sparse_physics_preflight(adapter, context, trajectories)
        except SampleRejected as error:
            physics_failures.append({
                "reason": error.reason,
                "details": error.details,
                "route_ids": [
                    route.route_id for route in proposal["selected_routes"]
                ],
            })
            continue
        source = proposal["source_metrics"]
        selected_routes = proposal["selected_routes"]
        evidence = proposal["evidence"]
        joint = dict(proposal["joint"])
        joint.update({
            "formation_degenerate": False,
            "measured_overlap_route_mutation": evidence,
        })
        traversed = {
            robot_id: list(route.traversed_regions)
            for robot_id, route in zip(robot_ids, selected_routes, strict=True)
        }
        nested = dict(source.get("nested_trajectory_sets", {}))
        nested.update({
            "candidate_kind": "measured_overlap_route_mutation",
            "source_overlap_evidence": evidence,
        })
        family_counts = Counter(
            route.trajectory.path_family for route in selected_routes
        )
        metrics = {
            **source,
            "route_ids": [route.route_id for route in selected_routes],
            "start_region_ids": [route.start_region for route in selected_routes],
            "goal_region_ids": [route.goal_region for route in selected_routes],
            "traversed_region_ids": traversed,
            "unique_traversed_region_count": len({
                region for values in traversed.values() for region in values
            }),
            "minimum_pairwise_distance_m": float(
                joint["minimum_inter_robot_distance_m"]
            ),
            "joint_diversity": joint,
            "cheap_scene_visibility": proposal["cheap"],
            "robots": {
                robot_id: route.metadata()
                for robot_id, route in zip(robot_ids, selected_routes, strict=True)
            },
            "route_family_diversity": {
                "counts": dict(family_counts),
                "unique_family_count": len(family_counts),
                "waypoint_route_count": sum(
                    route.trajectory.path_family != "direct"
                    for route in selected_routes
                ),
                "hard_minimum_enforced": False,
            },
            "joint_route_search": {
                **source.get("joint_route_search", {}),
                "assignment_indices": list(proposal["assignment"]),
                "rescue_strategy": "measured_overlap_route_mutation",
                "rescue_source_candidate_rank": proposal["source_rank"],
                "rescue_sparse_physics_failures": physics_failures[:20],
            },
            "nested_trajectory_sets": nested,
            "rescue_candidate_accounting": {
                "duplicate_route_triplets_removed": duplicate_triplets_removed,
            },
        }
        accepted.append((trajectories, metrics))
        if len(accepted) >= maximum_candidates:
            break
    return tuple(accepted)
