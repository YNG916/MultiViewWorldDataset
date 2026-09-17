from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from multi_view_world_dataset.errors import SampleRejected
from multi_view_world_dataset.sampling.diversity import joint_trajectory_metrics
from multi_view_world_dataset.sampling.trajectories import trajectory_kinematic_metrics
from multi_view_world_dataset.schema.records import Trajectory

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


def convex_hull_xy(points: Sequence[Sequence[float]]) -> FloatArray:
    """Return a deterministic counter-clockwise 2-D convex hull."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("points must be a finite [N,2] array")
    unique = np.unique(values, axis=0)
    if len(unique) < 3:
        raise ValueError("a footprint requires at least three non-collinear points")
    ordered = sorted(map(tuple, unique.tolist()))

    def cross(origin: tuple[float, float], left: tuple[float, float], right: tuple[float, float]) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (
            left[1] - origin[1]
        ) * (right[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    if len(hull) < 3 or abs(polygon_area(hull)) <= 1.0e-10:
        raise ValueError("footprint points are collinear")
    return hull


def polygon_area(polygon_xy: Sequence[Sequence[float]]) -> float:
    polygon = np.asarray(polygon_xy, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        return 0.0
    following = np.roll(polygon, -1, axis=0)
    return 0.5 * float(np.sum(
        polygon[:, 0] * following[:, 1] - polygon[:, 1] * following[:, 0]
    ))


def _expanded_convex_hull(polygon_xy: FloatArray, margin_m: float) -> FloatArray:
    if margin_m <= 0.0:
        return polygon_xy.copy()
    angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    disk = margin_m * np.column_stack((np.cos(angles), np.sin(angles)))
    minkowski_points = (
        polygon_xy[:, None, :] + disk[None, :, :]
    ).reshape(-1, 2)
    return convex_hull_xy(minkowski_points)


@dataclass(frozen=True)
class RobotFootprintModel:
    """Horizontal footprint extracted from collision-enabled robot links."""

    source_collision_links: tuple[str, ...]
    polygon_xy: FloatArray
    raw_polygon_xy: FloatArray
    width_m: float
    length_m: float
    area_m2: float
    circumscribed_radius_m: float
    safety_margin_m: float
    yaw_bins: int
    old_reset_aabb_extent_xy_m: tuple[float, float]

    def __post_init__(self) -> None:
        for name in ("polygon_xy", "raw_polygon_xy"):
            polygon = np.asarray(getattr(self, name), dtype=np.float64)
            if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
                raise ValueError(f"{name} must be a polygon with shape [N,2]")
            object.__setattr__(self, name, polygon)
        if self.yaw_bins < 4 or self.safety_margin_m < 0.0:
            raise ValueError("yaw_bins must be >= 4 and safety_margin_m non-negative")

    def metadata(self) -> dict[str, Any]:
        return {
            "source_collision_links": list(self.source_collision_links),
            "polygon_xy": self.polygon_xy.tolist(),
            "raw_polygon_xy": self.raw_polygon_xy.tolist(),
            "width_m": self.width_m,
            "length_m": self.length_m,
            "area_m2": self.area_m2,
            "circumscribed_radius_m": self.circumscribed_radius_m,
            "safety_margin_m": self.safety_margin_m,
            "yaw_bins": self.yaw_bins,
            "old_reset_aabb_extent_xy_m": list(self.old_reset_aabb_extent_xy_m),
        }


def build_robot_footprint_model(
    collision_points_by_link: Mapping[str, Sequence[Sequence[float]]],
    *,
    safety_margin_m: float,
    yaw_bins: int,
    old_reset_aabb_extent_xy_m: Sequence[float],
) -> RobotFootprintModel:
    """Build a convex footprint from collision points expressed in base frame."""
    usable: dict[str, FloatArray] = {}
    for link_name, points in collision_points_by_link.items():
        values = np.asarray(points, dtype=np.float64)
        if values.ndim == 2 and values.shape[1] >= 2 and len(values):
            finite = np.isfinite(values[:, :2]).all(axis=1)
            if np.any(finite):
                usable[str(link_name)] = values[finite, :2]
    if not usable:
        raise SampleRejected("robot_footprint_no_collision_geometry")
    raw = convex_hull_xy(np.concatenate(list(usable.values()), axis=0))
    expanded = _expanded_convex_hull(raw, float(safety_margin_m))
    extent = np.ptp(expanded, axis=0)
    return RobotFootprintModel(
        source_collision_links=tuple(sorted(usable)),
        polygon_xy=expanded,
        raw_polygon_xy=raw,
        length_m=float(extent[0]),
        width_m=float(extent[1]),
        area_m2=abs(polygon_area(expanded)),
        circumscribed_radius_m=float(np.linalg.norm(expanded, axis=1).max()),
        safety_margin_m=float(safety_margin_m),
        yaw_bins=int(yaw_bins),
        old_reset_aabb_extent_xy_m=tuple(
            map(float, np.asarray(old_reset_aabb_extent_xy_m).reshape(2))
        ),
    )


def points_in_convex_polygon(points_xy: FloatArray, polygon_xy: FloatArray) -> BoolArray:
    points = np.asarray(points_xy, dtype=np.float64)
    polygon = np.asarray(polygon_xy, dtype=np.float64)
    edges = np.roll(polygon, -1, axis=0) - polygon
    relative = points[:, None, :] - polygon[None, :, :]
    cross = edges[None, :, 0] * relative[:, :, 1] - edges[None, :, 1] * relative[:, :, 0]
    tolerance = 1.0e-9
    return np.all(cross >= -tolerance, axis=1) | np.all(cross <= tolerance, axis=1)


def sample_footprint_interior(polygon_xy: FloatArray, spacing_m: float) -> FloatArray:
    """Densely sample the complete polygon area, including its boundary."""
    polygon = np.asarray(polygon_xy, dtype=np.float64)
    if spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive")
    lower = polygon.min(axis=0)
    upper = polygon.max(axis=0)
    xs = np.arange(lower[0], upper[0] + 0.5 * spacing_m, spacing_m)
    ys = np.arange(lower[1], upper[1] + 0.5 * spacing_m, spacing_m)
    grid = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
    inside = grid[points_in_convex_polygon(grid, polygon)]
    boundary: list[np.ndarray] = []
    for left, right in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        count = max(2, int(np.ceil(np.linalg.norm(right - left) / spacing_m)) + 1)
        boundary.append(np.linspace(left, right, count))
    return np.unique(np.concatenate((inside, *boundary), axis=0), axis=0)


def oriented_safe_masks(
    point_free_mask: BoolArray,
    footprint_offsets_by_yaw: Sequence[IntArray],
) -> BoolArray:
    """Return [Y,H,W] base-pose masks for pre-rasterized footprint offsets."""
    free = np.asarray(point_free_mask, dtype=bool)
    if free.ndim != 2:
        raise ValueError("point_free_mask must be 2-D")
    height, width = free.shape
    result = np.empty((len(footprint_offsets_by_yaw), height, width), dtype=bool)
    for yaw_index, offsets in enumerate(footprint_offsets_by_yaw):
        offsets = np.unique(np.asarray(offsets, dtype=np.int64), axis=0)
        safe = np.ones_like(free)
        for row_delta, column_delta in offsets:
            shifted = np.zeros_like(free)
            source_rows = slice(max(0, row_delta), min(height, height + row_delta))
            source_columns = slice(max(0, column_delta), min(width, width + column_delta))
            target_rows = slice(max(0, -row_delta), min(height, height - row_delta))
            target_columns = slice(max(0, -column_delta), min(width, width - column_delta))
            shifted[target_rows, target_columns] = free[source_rows, source_columns]
            safe &= shifted
        result[yaw_index] = safe
    return result


def connected_components(mask: BoolArray) -> tuple[IntArray, tuple[int, ...]]:
    """Label 8-connected free-space components without an extra dependency."""
    free = np.asarray(mask, dtype=bool)
    labels = np.full(free.shape, -1, dtype=np.int64)
    sizes: list[int] = []
    height, width = free.shape
    for row, column in np.argwhere(free):
        if labels[row, column] >= 0:
            continue
        component = len(sizes)
        labels[row, column] = component
        queue = deque([(int(row), int(column))])
        size = 0
        while queue:
            current_row, current_column = queue.popleft()
            size += 1
            for row_delta in (-1, 0, 1):
                for column_delta in (-1, 0, 1):
                    if row_delta == 0 and column_delta == 0:
                        continue
                    neighbor_row = current_row + row_delta
                    neighbor_column = current_column + column_delta
                    if (
                        0 <= neighbor_row < height
                        and 0 <= neighbor_column < width
                        and free[neighbor_row, neighbor_column]
                        and labels[neighbor_row, neighbor_column] < 0
                    ):
                        labels[neighbor_row, neighbor_column] = component
                        queue.append((neighbor_row, neighbor_column))
        sizes.append(size)
    return labels, tuple(sizes)


@dataclass(frozen=True)
class RegionGraph:
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    cell_counts: dict[str, int]
    connector_nodes: tuple[str, ...] = ()

    def neighbors(self, node: str) -> tuple[str, ...]:
        return tuple(sorted(
            right if left == node else left
            for left, right in self.edges
            if left == node or right == node
        ))

    def metadata(self) -> dict[str, Any]:
        return {
            "nodes": list(self.nodes),
            "edges": [list(edge) for edge in self.edges],
            "cell_counts": dict(self.cell_counts),
            "connector_nodes": list(self.connector_nodes),
        }


def build_region_graph(
    navigable_mask: BoolArray,
    region_labels: NDArray[np.object_] | NDArray[np.str_],
) -> tuple[RegionGraph, NDArray[np.str_]]:
    """Build topology from neighboring navigable cells, preserving connectors."""
    free = np.asarray(navigable_mask, dtype=bool)
    labels = np.asarray(region_labels, dtype=object).copy()
    if labels.shape != free.shape:
        raise ValueError("region_labels must have the same shape as navigable_mask")
    labels[~free] = ""
    missing = free & (labels == "")
    connector_nodes: list[str] = []
    connector_index = 0
    height, width = free.shape
    for start_row, start_column in np.argwhere(missing):
        if labels[start_row, start_column] != "":
            continue
        name = f"connector_{connector_index:03d}"
        connector_index += 1
        connector_nodes.append(name)
        labels[start_row, start_column] = name
        queue = deque([(int(start_row), int(start_column))])
        while queue:
            row, column = queue.popleft()
            for row_delta, column_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                other_row, other_column = row + row_delta, column + column_delta
                if (
                    0 <= other_row < height
                    and 0 <= other_column < width
                    and missing[other_row, other_column]
                    and labels[other_row, other_column] == ""
                ):
                    labels[other_row, other_column] = name
                    queue.append((other_row, other_column))
    edges: set[tuple[str, str]] = set()
    for row_delta, column_delta in ((1, 0), (0, 1)):
        first = labels[: height - row_delta or None, : width - column_delta or None]
        second = labels[row_delta:, column_delta:]
        boundary = (first != "") & (second != "") & (first != second)
        for left, right in zip(first[boundary], second[boundary], strict=True):
            edges.add(tuple(sorted((str(left), str(right)))))
    nodes, counts = np.unique(labels[free].astype(str), return_counts=True)
    graph = RegionGraph(
        nodes=tuple(map(str, nodes.tolist())),
        edges=tuple(sorted(edges)),
        cell_counts=dict(zip(map(str, nodes.tolist()), map(int, counts.tolist()), strict=True)),
        connector_nodes=tuple(connector_nodes),
    )
    return graph, labels.astype(str)


@dataclass(frozen=True)
class RouteCandidate:
    route_id: str
    floor_index: int
    start_region: str
    goal_region: str
    traversed_regions: tuple[str, ...]
    trajectory: Trajectory
    route_seed: int
    minimum_footprint_clearance_m: float
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def start_xy(self) -> FloatArray:
        return self.trajectory.base_to_world[0, :2, 3]

    @property
    def goal_xy(self) -> FloatArray:
        return self.trajectory.base_to_world[-1, :2, 3]

    def metadata(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "floor_index": self.floor_index,
            "start_xy": self.start_xy.tolist(),
            "goal_xy": self.goal_xy.tolist(),
            "start_region": self.start_region,
            "goal_region": self.goal_region,
            "traversed_regions": list(self.traversed_regions),
            "path_family": self.trajectory.path_family,
            "control_waypoints_xy": self.trajectory.control_waypoints_xy.tolist(),
            "planner_polyline_xy": self.trajectory.planner_path_xy.tolist(),
            "smoothed_path_xy": self.trajectory.smoothed_path_xy.tolist(),
            "simplified_path_xy": self.trajectory.simplified_path_xy.tolist(),
            "route_seed": self.route_seed,
            "minimum_footprint_clearance_m": self.minimum_footprint_clearance_m,
            **self.metrics,
        }


@dataclass(frozen=True)
class RouteCompatibility:
    compatible: BoolArray
    minimum_temporal_distance_m: FloatArray
    maximum_temporal_distance_m: FloatArray
    start_distance_m: FloatArray
    path_similarity: FloatArray
    mean_heading_difference_rad: FloatArray
    near_duplicate: BoolArray
    footprint_collision: BoolArray
    spatial_path_intersection: BoolArray
    close_approach: BoolArray
    heading_relation: NDArray[np.object_]
    region_relation: NDArray[np.object_]
    cheap_view_similarity: FloatArray | None = None

    @property
    def compatible_pair_fraction(self) -> float:
        count = len(self.compatible)
        if count < 2:
            return 0.0
        upper = np.triu_indices(count, 1)
        return float(np.mean(self.compatible[upper]))

    def metadata(self) -> dict[str, Any]:
        upper = np.triu_indices(len(self.compatible), 1)
        return {
            "route_count": len(self.compatible),
            "compatible_pair_fraction": self.compatible_pair_fraction,
            "minimum_temporal_distance_m": _distribution(
                self.minimum_temporal_distance_m[upper]
            ),
            "maximum_temporal_distance_m": _distribution(
                self.maximum_temporal_distance_m[upper]
            ),
            "start_distance_m": _distribution(self.start_distance_m[upper]),
            "path_similarity": _distribution(self.path_similarity[upper]),
            "cheap_scene_view_similarity": _distribution(
                self.cheap_view_similarity[upper]
            ) if self.cheap_view_similarity is not None else {"count": 0},
            "near_duplicate_pair_count": int(np.count_nonzero(self.near_duplicate[upper])),
            "footprint_collision_pair_count": int(
                np.count_nonzero(self.footprint_collision[upper])
            ),
            "spatial_path_intersection_pair_count": int(
                np.count_nonzero(self.spatial_path_intersection[upper])
            ),
            "close_approach_pair_count": int(
                np.count_nonzero(self.close_approach[upper])
            ),
            "heading_relation_counts": dict(Counter(
                map(str, self.heading_relation[upper].tolist())
            )),
            "region_relation_counts": dict(Counter(
                map(str, self.region_relation[upper].tolist())
            )),
        }


def _route_pair_metrics(left: RouteCandidate, right: RouteCandidate) -> tuple[float, float, float, float, bool]:
    left_positions = left.trajectory.base_to_world[:, :2, 3]
    right_positions = right.trajectory.base_to_world[:, :2, 3]
    distances = np.linalg.norm(left_positions - right_positions, axis=1)
    left_steps = np.diff(left_positions, axis=0)
    right_steps = np.diff(right_positions, axis=0)
    left_norms = np.linalg.norm(left_steps, axis=1)
    right_norms = np.linalg.norm(right_steps, axis=1)
    valid = (left_norms > 1.0e-9) & (right_norms > 1.0e-9)
    similarities = (
        np.sum(left_steps[valid] * right_steps[valid], axis=1)
        / (left_norms[valid] * right_norms[valid])
        if np.any(valid) else np.asarray([1.0])
    )
    left_yaw = np.unwrap(np.arctan2(
        left.trajectory.base_to_world[:, 1, 0], left.trajectory.base_to_world[:, 0, 0]
    ))
    right_yaw = np.unwrap(np.arctan2(
        right.trajectory.base_to_world[:, 1, 0], right.trajectory.base_to_world[:, 0, 0]
    ))
    heading_difference = np.abs((left_yaw - right_yaw + np.pi) % (2.0 * np.pi) - np.pi)
    path_similarity = float(np.mean(similarities))
    mean_heading_difference = float(np.mean(heading_difference))
    near_duplicate = bool(
        float(distances.max()) < 0.35
        and path_similarity > 0.97
        and mean_heading_difference < np.deg2rad(8.0)
    )
    return (
        float(distances.min()),
        float(np.linalg.norm(left_positions[0] - right_positions[0])),
        path_similarity,
        mean_heading_difference,
        near_duplicate,
    )


def _spatial_paths_intersect(left_xy: FloatArray, right_xy: FloatArray) -> bool:
    """Return whether two 2-D route polylines intersect at any segment."""
    def cross(first: np.ndarray, second: np.ndarray) -> float:
        return float(first[0] * second[1] - first[1] * second[0])

    tolerance = 1.0e-9
    for left_start, left_end in zip(left_xy[:-1], left_xy[1:], strict=True):
        left_vector = left_end - left_start
        for right_start, right_end in zip(
            right_xy[:-1], right_xy[1:], strict=True
        ):
            right_vector = right_end - right_start
            denominator = cross(left_vector, right_vector)
            offset = right_start - left_start
            if abs(denominator) <= tolerance:
                if abs(cross(offset, left_vector)) > tolerance:
                    continue
                left_min = np.minimum(left_start, left_end)
                left_max = np.maximum(left_start, left_end)
                right_min = np.minimum(right_start, right_end)
                right_max = np.maximum(right_start, right_end)
                if np.all(
                    np.maximum(left_min, right_min)
                    <= np.minimum(left_max, right_max) + tolerance
                ):
                    return True
                continue
            left_parameter = cross(offset, right_vector) / denominator
            right_parameter = cross(offset, left_vector) / denominator
            if (
                -tolerance <= left_parameter <= 1.0 + tolerance
                and -tolerance <= right_parameter <= 1.0 + tolerance
            ):
                return True
    return False


def _convex_polygons_overlap(left: FloatArray, right: FloatArray) -> bool:
    """Return whether two convex polygons overlap or touch using SAT."""
    tolerance = 1.0e-9
    for polygon in (left, right):
        edges = np.roll(polygon, -1, axis=0) - polygon
        axes = np.column_stack((-edges[:, 1], edges[:, 0]))
        for axis in axes:
            left_projection = left @ axis
            right_projection = right @ axis
            if (
                float(left_projection.max()) < float(right_projection.min()) - tolerance
                or float(right_projection.max()) < float(left_projection.min()) - tolerance
            ):
                return False
    return True


def _routes_have_footprint_collision(
    left: RouteCandidate,
    right: RouteCandidate,
    footprint: RobotFootprintModel,
) -> bool:
    """Check synchronized physical robot polygons at every sampled frame."""
    local_polygon = footprint.raw_polygon_xy
    left_poses = left.trajectory.base_to_world
    right_poses = right.trajectory.base_to_world
    if len(left_poses) != len(right_poses):
        raise ValueError("route trajectories must have identical frame counts")
    for left_pose, right_pose in zip(left_poses, right_poses, strict=True):
        left_polygon = local_polygon @ left_pose[:2, :2].T + left_pose[:2, 3]
        right_polygon = local_polygon @ right_pose[:2, :2].T + right_pose[:2, 3]
        if _convex_polygons_overlap(left_polygon, right_polygon):
            return True
    return False


def compute_pairwise_route_compatibility(
    routes: Sequence[RouteCandidate],
    *,
    minimum_pairwise_distance_m: float,
    footprint: RobotFootprintModel | None = None,
) -> RouteCompatibility:
    count = len(routes)
    compatible = np.eye(count, dtype=bool)
    minimum_distance = np.zeros((count, count), dtype=np.float64)
    maximum_distance = np.zeros((count, count), dtype=np.float64)
    start_distance = np.zeros((count, count), dtype=np.float64)
    similarity = np.eye(count, dtype=np.float64)
    heading_difference = np.zeros((count, count), dtype=np.float64)
    near_duplicate = np.eye(count, dtype=bool)
    footprint_collision = np.zeros((count, count), dtype=bool)
    spatial_intersection = np.zeros((count, count), dtype=bool)
    close_approach = np.zeros((count, count), dtype=bool)
    heading_relation = np.full((count, count), "same_route", dtype=object)
    region_relation = np.full((count, count), "same_route", dtype=object)
    for left in range(count):
        for right in range(left + 1, count):
            values = _route_pair_metrics(routes[left], routes[right])
            minimum_distance[left, right] = minimum_distance[right, left] = values[0]
            left_positions = routes[left].trajectory.base_to_world[:, :2, 3]
            right_positions = routes[right].trajectory.base_to_world[:, :2, 3]
            pair_maximum_distance = float(np.max(
                np.linalg.norm(left_positions - right_positions, axis=1)
            ))
            maximum_distance[left, right] = maximum_distance[right, left] = (
                pair_maximum_distance
            )
            start_distance[left, right] = start_distance[right, left] = values[1]
            similarity[left, right] = similarity[right, left] = values[2]
            heading_difference[left, right] = heading_difference[right, left] = values[3]
            near_duplicate[left, right] = near_duplicate[right, left] = values[4]
            bodies_collide = bool(
                footprint is not None
                and _routes_have_footprint_collision(
                    routes[left], routes[right], footprint
                )
            )
            footprint_collision[left, right] = footprint_collision[right, left] = (
                bodies_collide
            )
            intersects = _spatial_paths_intersect(left_positions, right_positions)
            spatial_intersection[left, right] = spatial_intersection[right, left] = intersects
            approaches = values[0] < 1.5 * minimum_pairwise_distance_m
            close_approach[left, right] = close_approach[right, left] = approaches
            heading_label = (
                "aligned" if values[3] < np.deg2rad(30.0)
                else "opposed" if values[3] > np.deg2rad(150.0)
                else "crossing"
            )
            heading_relation[left, right] = heading_relation[right, left] = heading_label
            left_regions = set(routes[left].traversed_regions)
            right_regions = set(routes[right].traversed_regions)
            if routes[left].start_region == routes[right].start_region:
                region_label = "shared_start_region"
            elif left_regions & right_regions:
                region_label = "shared_traversed_region"
            else:
                region_label = "disjoint_regions"
            region_relation[left, right] = region_relation[right, left] = region_label
            passed = (
                values[0] >= minimum_pairwise_distance_m
                and not values[4]
                and not bodies_collide
            )
            compatible[left, right] = compatible[right, left] = passed
    return RouteCompatibility(
        compatible=compatible,
        minimum_temporal_distance_m=minimum_distance,
        maximum_temporal_distance_m=maximum_distance,
        start_distance_m=start_distance,
        path_similarity=similarity,
        mean_heading_difference_rad=heading_difference,
        near_duplicate=near_duplicate,
        footprint_collision=footprint_collision,
        spatial_path_intersection=spatial_intersection,
        close_approach=close_approach,
        heading_relation=heading_relation,
        region_relation=region_relation,
    )


def _triplet_score(routes: tuple[RouteCandidate, RouteCandidate, RouteCandidate]) -> float:
    metrics = joint_trajectory_metrics(tuple(route.trajectory for route in routes))
    regions = {region for route in routes for region in route.traversed_regions}
    families = {route.trajectory.path_family for route in routes}
    diversity = 1.0 - max(0.0, float(metrics["mean_path_direction_similarity"]))
    return float(
        1.5 * len(regions)
        + 0.5 * len(families)
        + min(float(metrics["spatial_coverage_bbox_area_m2"]) / 12.0, 1.0)
        + diversity
        + 0.25 * min(float(metrics["mean_pairwise_heading_difference_rad"]) / np.pi, 1.0)
    )


def route_start_regions_connected(
    routes: Sequence[RouteCandidate],
    region_graph: RegionGraph,
) -> bool:
    """Return whether all route starts share one footprint-topology component."""
    targets = {route.start_region for route in routes}
    if len(targets) <= 1:
        return True
    if not targets.issubset(region_graph.nodes):
        return False
    reached = {next(iter(targets))}
    frontier = list(reached)
    while frontier:
        node = frontier.pop()
        for neighbor in region_graph.neighbors(node):
            if neighbor not in reached:
                reached.add(neighbor)
                frontier.append(neighbor)
    return targets.issubset(reached)


def select_joint_route_candidates(
    routes: Sequence[RouteCandidate],
    compatibility: RouteCompatibility,
    rng: np.random.Generator,
    *,
    top_k: int,
    search_budget: int,
    minimum_waypoint_trajectories: int,
    discouraged_regions: Sequence[str] = (),
    cheap_visibility_score: Callable[[tuple[RouteCandidate, ...]], float] | None = None,
    visibility_priority_fraction: float = 0.0,
    visibility_score_weight: float = 1.0,
    region_graph: RegionGraph | None = None,
) -> tuple[tuple[int, int, int], ...]:
    """Bounded compatibility-aware three-route search; never enumerates B^3."""
    count = len(routes)
    if count < 3 or top_k < 1 or search_budget < 1:
        return ()
    if not 0.0 <= visibility_priority_fraction <= 1.0:
        raise ValueError("visibility_priority_fraction must lie in [0, 1]")
    if visibility_score_weight < 0.0:
        raise ValueError("visibility_score_weight must be non-negative")
    discouraged = set(map(str, discouraged_regions))
    region_frequency: dict[str, int] = {}
    for route in routes:
        region_frequency[route.start_region] = region_frequency.get(route.start_region, 0) + 1
    first_weights = np.asarray([
        1.0 / max(1, region_frequency[route.start_region])
        * (0.35 if route.start_region in discouraged else 1.0)
        for route in routes
    ], dtype=np.float64)
    first_weights /= first_weights.sum()
    # Keep connectivity and diversity as separate objectives so raw region
    # coverage cannot crowd every useful-overlap candidate out of Top-K.
    found: dict[tuple[int, int, int], tuple[float, float]] = {}
    for _ in range(search_budget):
        first = int(rng.choice(count, p=first_weights))
        second_pool = np.flatnonzero(compatibility.compatible[first])
        second_pool = second_pool[second_pool != first]
        if not len(second_pool):
            continue
        second = int(rng.choice(second_pool))
        third_pool = np.flatnonzero(
            compatibility.compatible[first] & compatibility.compatible[second]
        )
        third_pool = third_pool[(third_pool != first) & (third_pool != second)]
        if not len(third_pool):
            continue
        third = int(rng.choice(third_pool))
        indices = tuple(sorted((first, second, third)))
        selected = tuple(routes[index] for index in indices)
        if (
            region_graph is not None
            and not route_start_regions_connected(selected, region_graph)
        ):
            continue
        if sum(route.trajectory.path_family != "direct" for route in selected) < minimum_waypoint_trajectories:
            continue
        score = _triplet_score(selected)
        visibility_score = 0.0
        if cheap_visibility_score is not None:
            visibility_score = float(cheap_visibility_score(selected))
            if not np.isfinite(visibility_score):
                continue
            score += visibility_score_weight * visibility_score
        prior = found.get(indices)
        value = (float(score), float(visibility_score))
        if prior is None or value[0] > prior[0]:
            found[indices] = value

    combined_ranked = sorted(
        found, key=lambda item: (-found[item][0], -found[item][1], item)
    )
    if cheap_visibility_score is None or visibility_priority_fraction <= 0.0:
        return tuple(combined_ranked[:top_k])
    visibility_ranked = sorted(
        found, key=lambda item: (-found[item][1], -found[item][0], item)
    )
    priority_count = min(
        top_k, int(np.ceil(top_k * visibility_priority_fraction))
    )
    shortlisted = list(visibility_ranked[:priority_count])
    shortlisted_set = set(shortlisted)
    for item in combined_ranked:
        if len(shortlisted) >= top_k:
            break
        if item in shortlisted_set:
            continue
        shortlisted.append(item)
        shortlisted_set.add(item)
        if len(shortlisted) >= top_k:
            break
    return tuple(shortlisted)


@dataclass
class NavigationContext:
    floor_index: int
    map_resolution_m: float
    point_free_mask: BoolArray
    footprint_safe_masks: BoolArray
    planner_mask: BoolArray
    component_labels: IntArray
    component_sizes: tuple[int, ...]
    region_label_grid: NDArray[np.str_]
    region_graph: RegionGraph
    footprint: RobotFootprintModel
    route_bank: tuple[RouteCandidate, ...]
    compatibility: RouteCompatibility
    cheap_visible_cells: tuple[tuple[frozenset[int], ...], ...] = ()
    cheap_view_overlap_by_keyframe: FloatArray = field(
        default_factory=lambda: np.empty((0, 0, 0), dtype=np.float32)
    )
    diagnostics: dict[str, Any] = field(default_factory=dict)
    invalid_start_pose_blacklist: set[tuple[int, int, int]] = field(default_factory=set)

    def metadata(self, *, include_routes: bool = True) -> dict[str, Any]:
        report = {
            "floor_index": self.floor_index,
            "map_resolution_m": self.map_resolution_m,
            "supported_floor_cell_count": int(np.count_nonzero(self.point_free_mask)),
            "footprint_safe_cell_count": int(np.count_nonzero(self.planner_mask)),
            "connected_component_sizes": list(map(int, self.component_sizes)),
            "region_graph": self.region_graph.metadata(),
            "robot_footprint": self.footprint.metadata(),
            "route_bank_size": len(self.route_bank),
            "pairwise_compatibility": self.compatibility.metadata(),
            "diagnostics": self.diagnostics,
            "invalid_start_pose_blacklist_size": len(self.invalid_start_pose_blacklist),
        }
        if include_routes:
            report["routes"] = [route.metadata() for route in self.route_bank]
        return report


def route_candidate_from_trajectory(
    route_id: str,
    floor_index: int,
    trajectory: Trajectory,
    route_seed: int,
    region_for_points: Callable[[FloatArray], Sequence[str]],
    *,
    minimum_footprint_clearance_m: float,
) -> RouteCandidate:
    path_regions = tuple(map(str, region_for_points(
        trajectory.base_to_world[:, :2, 3]
    )))
    metrics = trajectory_kinematic_metrics(trajectory)
    return RouteCandidate(
        route_id=route_id,
        floor_index=floor_index,
        start_region=path_regions[0],
        goal_region=path_regions[-1],
        traversed_regions=tuple(dict.fromkeys(path_regions)),
        trajectory=trajectory,
        route_seed=int(route_seed),
        minimum_footprint_clearance_m=float(minimum_footprint_clearance_m),
        metrics={
            **metrics,
            "planner_geodesic_length_m": float(
                trajectory.metadata.get("planner_geodesic_length_m", np.nan)
            ),
        },
    )


def _distribution(values: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "minimum": float(np.min(array)),
        "mean": float(np.mean(array)),
        "maximum": float(np.max(array)),
    }
