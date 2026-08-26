from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike

from multi_view_world_dataset.cameras.calibration import backproject_depth, project_world_points
from multi_view_world_dataset.errors import GeometryError


def _directed_overlap(
    source_depth: np.ndarray,
    source_intrinsics: np.ndarray,
    source_camera_to_world: np.ndarray,
    target_depth: np.ndarray,
    target_intrinsics: np.ndarray,
    target_camera_to_world: np.ndarray,
    stride: int,
    tolerance_m: float,
) -> float:
    points = backproject_depth(source_depth, source_intrinsics, source_camera_to_world, stride=stride)
    if len(points) == 0:
        return 0.0
    pixels, projected_depth = project_world_points(points, target_intrinsics, target_camera_to_world)
    finite_projection = np.isfinite(pixels).all(axis=1) & np.isfinite(projected_depth)
    u = np.zeros(len(points), dtype=np.int64)
    v = np.zeros(len(points), dtype=np.int64)
    u[finite_projection] = np.rint(pixels[finite_projection, 0]).astype(np.int64)
    v[finite_projection] = np.rint(pixels[finite_projection, 1]).astype(np.int64)
    inside = (
        finite_projection
        & (projected_depth > 0)
        & (u >= 0)
        & (u < target_depth.shape[1])
        & (v >= 0)
        & (v < target_depth.shape[0])
    )
    if not inside.any():
        return 0.0
    observed = np.full(len(points), np.nan, dtype=np.float64)
    observed[inside] = target_depth[v[inside], u[inside]]
    shared = inside & np.isfinite(observed) & (observed > 0) & (np.abs(observed - projected_depth) <= tolerance_m)
    return float(shared.sum() / len(points))


def pairwise_visible_surface_overlap(
    depth_a: ArrayLike,
    intrinsics_a: ArrayLike,
    camera_a_to_world: ArrayLike,
    depth_b: ArrayLike,
    intrinsics_b: ArrayLike,
    camera_b_to_world: ArrayLike,
    *,
    stride: int = 8,
    tolerance_m: float = 0.08,
) -> float:
    """Symmetric GT-depth overlap: mean fraction of each view geometrically verified in the other."""
    a = np.asarray(depth_a, dtype=np.float64)
    b = np.asarray(depth_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or stride < 1 or tolerance_m <= 0:
        raise GeometryError("Invalid depth overlap inputs")
    ka, kb = np.asarray(intrinsics_a, dtype=np.float64), np.asarray(intrinsics_b, dtype=np.float64)
    ta, tb = np.asarray(camera_a_to_world, dtype=np.float64), np.asarray(camera_b_to_world, dtype=np.float64)
    ab = _directed_overlap(a, ka, ta, b, kb, tb, stride, tolerance_m)
    ba = _directed_overlap(b, kb, tb, a, ka, ta, stride, tolerance_m)
    return 0.5 * (ab + ba)


@dataclass(frozen=True)
class OverlapGraph:
    robot_ids: tuple[str, ...]
    overlaps: dict[tuple[str, str], float]
    edges: tuple[tuple[str, str], ...]
    connected: bool
    near_duplicate_pairs: tuple[tuple[str, str], ...]


def _connected(nodes: tuple[str, ...], edges: tuple[tuple[str, str], ...]) -> bool:
    if not nodes:
        return False
    adjacency = {node: set() for node in nodes}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    visited, frontier = set(), [nodes[0]]
    while frontier:
        node = frontier.pop()
        if node in visited:
            continue
        visited.add(node)
        frontier.extend(adjacency[node] - visited)
    return len(visited) == len(nodes)


def build_overlap_graph(
    robot_ids: list[str] | tuple[str, ...],
    depths: dict[str, ArrayLike],
    intrinsics: dict[str, ArrayLike],
    camera_to_world: dict[str, ArrayLike],
    *,
    edge_threshold: float = 0.20,
    near_duplicate_threshold: float = 0.90,
    stride: int = 8,
    tolerance_m: float = 0.08,
) -> OverlapGraph:
    ids = tuple(robot_ids)
    overlaps: dict[tuple[str, str], float] = {}
    for left, right in combinations(ids, 2):
        overlaps[(left, right)] = pairwise_visible_surface_overlap(
            depths[left], intrinsics[left], camera_to_world[left],
            depths[right], intrinsics[right], camera_to_world[right],
            stride=stride, tolerance_m=tolerance_m,
        )
    edges = tuple(pair for pair, value in overlaps.items() if value >= edge_threshold)
    duplicates = tuple(pair for pair, value in overlaps.items() if value >= near_duplicate_threshold)
    return OverlapGraph(ids, overlaps, edges, _connected(ids, edges), duplicates)

