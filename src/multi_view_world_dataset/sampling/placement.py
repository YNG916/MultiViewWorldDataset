from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from multi_view_world_dataset.errors import SampleRejected


def sample_clustered_positions(
    candidates_xyz: Sequence[np.ndarray],
    rng: np.random.Generator,
    *,
    count: int = 3,
    cluster_radius_m: float = 3.0,
    minimum_pairwise_distance_m: float = 0.6,
    is_valid: Callable[[np.ndarray, tuple[np.ndarray, ...]], bool] | None = None,
    maximum_attempts: int = 200,
) -> tuple[np.ndarray, ...]:
    if len(candidates_xyz) < count:
        raise SampleRejected("insufficient_traversable_candidates", {"available": len(candidates_xyz)})
    center = np.asarray(candidates_xyz[int(rng.integers(len(candidates_xyz)))], dtype=np.float64)
    nearby = [np.asarray(point, dtype=np.float64) for point in candidates_xyz if np.linalg.norm(point[:2] - center[:2]) <= cluster_radius_m]
    for _ in range(maximum_attempts):
        selection: list[np.ndarray] = []
        for index in rng.permutation(len(nearby)):
            point = nearby[int(index)]
            if any(np.linalg.norm(point[:2] - other[:2]) < minimum_pairwise_distance_m for other in selection):
                continue
            if is_valid is not None and not is_valid(point, tuple(selection)):
                continue
            selection.append(point)
            if len(selection) == count:
                return tuple(selection)
    raise SampleRejected("robot_cluster_placement_failed", {"nearby_candidates": len(nearby)})

