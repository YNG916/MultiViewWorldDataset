from __future__ import annotations

import time
from collections import Counter
from dataclasses import replace
from typing import Any

import numpy as np

from multi_view_world_dataset.errors import SampleRejected, SimulatorUnavailableError
from multi_view_world_dataset.sampling.diversity import joint_trajectory_metrics, stable_seed
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
from multi_view_world_dataset.sampling.trajectories import (
    sample_geodesic_robot_trajectory_pool,
    densify_polyline,
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
    route.  Failed combinations receive a non-finite score so the bounded
    selector excludes them before PhysX or rendering.
    """
    route_indices = {route.route_id: index for index, route in enumerate(routes)}
    indices = [route_indices[route.route_id] for route in selected]
    robot_count = len(indices)
    frame_count = min((len(visible_cells[index]) for index in indices), default=0)
    if robot_count < 2 or frame_count == 0:
        return {
            "passed": False,
            "score": float("-inf"),
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
    ) if passed else float("-inf")
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
    # Fast hard-gate / score over precomputed route-pair overlap values.
    robot_count = len(selected_indices)
    frame_count = (
        int(overlap_by_keyframe.shape[2])
        if overlap_by_keyframe.ndim == 3
        else 0
    )
    if robot_count < 2 or frame_count == 0:
        return float("-inf")
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
    if not passed:
        return float("-inf")
    mean_overlap = float(np.mean(pairwise[upper]))
    useful_overlap = min(mean_overlap / max(edge_threshold, 1e-9), 1.0)
    return float(
        1.5 * connected_count / frame_count
        + 1.0
        + shared_count / frame_count
        + 1.5 * float(np.min(participation_counts)) / frame_count
        + 0.5 * useful_overlap
        - float(np.max(maximum_isolated_runs)) / frame_count
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
    # A cell that admits only one arbitrary yaw is too fragile for an XY
    # shortest-path proposal: the planner can cross it in a direction that the
    # actual rectangular robot cannot assume. Keep the permissive union for
    # diagnostics, but plan on cells with a configurable amount of yaw freedom.
    # The final tangent-derived yaw is still densely checked against the exact
    # orientation bin, so this improves proposals without relaxing collisions.
    permissive_mask = np.any(safe_masks, axis=0)
    yaw_freedom = np.mean(safe_masks, axis=0)
    seed_mask = permissive_mask & (
        yaw_freedom >= float(navigation["route_seed_minimum_yaw_fraction"])
    )
    if np.count_nonzero(seed_mask) < 3:
        seed_mask = permissive_mask.copy()
    planner_mask = seed_mask.copy()
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
    path_valid = _path_validator(adapter, safe_masks)
    plan_segment = _planner(adapter, floor_index, planner_mask)
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
        start = np.eye(4, dtype=np.float64)
        start[:2, 3] = start_xy
        start[2, 3] = floor_z
        route_seed = stable_seed(seed, "route", raw_attempt)
        try:
            pool = sample_geodesic_robot_trajectory_pool(
                f"route_{raw_attempt:04d}",
                start,
                np.eye(4, dtype=np.float64),
                candidates,
                floor_z,
                np.random.default_rng(route_seed),
                frames=int(adapter.config["dataset"]["frames"]),
                fps=float(adapter.config["dataset"]["fps"]),
                path_length_range_m=(
                    float(trajectory_config["path_length_min_m"]),
                    float(trajectory_config["path_length_max_m"]),
                ),
                maximum_linear_speed_mps=float(
                    trajectory_config["maximum_linear_speed_mps"]
                ),
                maximum_angular_speed_radps=float(
                    trajectory_config["maximum_angular_speed_radps"]
                ),
                maximum_acceleration_mps2=float(
                    trajectory_config["maximum_acceleration_mps2"]
                ),
                plan_segment=plan_segment,
                is_path_traversable=path_valid,
                path_family_weights=trajectory_config["path_family_weights"],
                initial_heading_tolerance_rad=np.pi,
                derive_initial_heading_from_tangent=True,
                initial_heading_probability_floor=1.0,
                maximum_control_turn_rad=np.deg2rad(
                    float(trajectory_config["maximum_control_turn_deg"])
                ),
                line_validation_spacing_m=float(
                    trajectory_config["line_validation_spacing_m"]
                ),
                smoothing_validation_spacing_m=float(
                    trajectory_config["smoothing_validation_spacing_m"]
                ),
                smoothing_strengths=trajectory_config["smoothing_strengths"],
                candidate_pool_size=1,
                maximum_attempts=attempts_per_raw,
            )
        except SampleRejected as error:
            reject_counts[error.reason] += 1
            for nested_reason, count in error.details.get(
                "rejection_counts", {}
            ).items():
                reject_counts[f"{error.reason}:{nested_reason}"] += int(count)
            continue
        trajectory = pool[0]
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
        # The opposite traversal direction is a distinct, useful route and is
        # almost free to propose. Recompute tangent yaw and revalidate it because
        # the final robot footprint is not assumed to be 180-degree symmetric.
        if len(routes) < target_size:
            reverse_id = f"route_{len(routes):04d}"
            reverse_path = trajectory.smoothed_path_xy[::-1].copy()
            reverse_trajectory = trajectory_from_spatial_path(
                reverse_id,
                reverse_path,
                floor_z,
                np.eye(4, dtype=np.float64),
                frames=int(adapter.config["dataset"]["frames"]),
                fps=float(adapter.config["dataset"]["fps"]),
                path_family=trajectory.path_family,
                control_waypoints_xy=trajectory.control_waypoints_xy[::-1].copy(),
                planner_path_xy=trajectory.planner_path_xy[::-1].copy(),
                simplified_path_xy=trajectory.simplified_path_xy[::-1].copy(),
                smoothed_path_xy=reverse_path,
                metadata={
                    **trajectory.metadata,
                    "route_variant": "time_reverse",
                },
            )
            reverse_start_pixel = goal_pixel
            reverse_goal_pixel = start_pixel
            reverse_signature = (
                int(reverse_start_pixel[0]), int(reverse_start_pixel[1]),
                int(reverse_goal_pixel[0]), int(reverse_goal_pixel[1]),
            )
            if (
                reverse_signature not in signatures
                and path_valid(densify_polyline(
                    reverse_trajectory.base_to_world[:, :2, 3],
                    float(trajectory_config["smoothing_validation_spacing_m"]),
                ))
            ):
                signatures.add(reverse_signature)
                reverse_route = route_candidate_from_trajectory(
                    reverse_id,
                    floor_index,
                    reverse_trajectory,
                    stable_seed(route_seed, "time-reverse"),
                    lambda points: _region_labels_for_points(
                        adapter, label_grid, points
                    ),
                    minimum_footprint_clearance_m=_minimum_route_footprint_clearance_m(
                        adapter,
                        safe_masks,
                        reverse_trajectory,
                        float(navigation["footprint_safety_margin_m"]),
                    ),
                )
                routes.append(reverse_route)
                route_counts_by_start[reverse_route.start_region] += 1
            else:
                reject_counts["reverse_variant_invalid"] += 1
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
    minimum_routes = int(adapter.config["navigation"]["route_bank_minimum_size"])
    for floor_index in range(int(adapter._require_scene().n_floors)):
        try:
            context = _build_floor_context(
                adapter,
                footprint,
                floor_index,
                stable_seed(seed, "navigation-floor", floor_index),
            )
            if len(context.route_bank) < minimum_routes:
                raise SampleRejected(
                    "navigation_route_bank_too_small",
                    {
                        "floor_index": floor_index,
                        "accepted_routes": len(context.route_bank),
                        "required_routes": minimum_routes,
                        **context.diagnostics,
                    },
                )
            probe = select_joint_route_candidates(
                context.route_bank,
                context.compatibility,
                np.random.default_rng(stable_seed(seed, "joint-feasibility", floor_index)),
                top_k=1,
                search_budget=int(adapter.config["navigation"]["joint_route_search_budget"]),
                minimum_waypoint_trajectories=int(
                    adapter.config["trajectory"]["minimum_waypoint_trajectories"]
                ),
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
    floors = [
        obj
        for obj in adapter._require_scene().objects
        if str(getattr(obj, "category", "")) == "floors"
    ]
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
    floors = [
        obj for obj in adapter._require_scene().objects
        if str(getattr(obj, "category", "")) == "floors"
    ]
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
    triples: dict[str, np.ndarray] = {
        "open": np.argwhere(context.footprint_safe_masks),
        "boundary": np.argwhere(
            (~context.footprint_safe_masks)
            & free[None, :, :]
            & near_free[None, :, :]
        ),
        "blocked": np.column_stack((
            np.zeros(np.count_nonzero((~free) & near_free), dtype=np.int64),
            np.argwhere((~free) & near_free),
        )),
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
    triplets = select_joint_route_candidates(
        context.route_bank,
        context.compatibility,
        rng,
        top_k=int(adapter.config["navigation"]["top_triplets_for_exact_validation"]),
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
        minimum_waypoint_trajectories=int(
            adapter.config["trajectory"]["minimum_waypoint_trajectories"]
        ),
        discouraged_regions=discouraged_region_ids,
        region_graph=(
            context.region_graph
            if adapter.config["navigation"]["require_connected_start_regions"]
            else None
        ),
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
            },
        }
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
    adapter._runtime_findings["route_first_episode_sampling"] = {
        "floor_index": floor_index,
        "shortlisted_triplets": len(triplets),
        "sparse_physx_accepted": len(accepted),
        "failures": failures,
    }
    adapter._runtime_findings["sampled_floor_index"] = floor_index
    return heights, tuple(accepted)
