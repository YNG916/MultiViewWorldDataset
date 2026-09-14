from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from multi_view_world_dataset.errors import SampleRejected


def select_local_traversable_heading(
    source_xy: np.ndarray,
    candidates_xy: Sequence[np.ndarray],
    desired_yaw: float,
    is_path_traversable: Callable[[np.ndarray], bool],
    *,
    minimum_probe_m: float,
    maximum_probe_m: float,
    validation_spacing_m: float,
) -> tuple[float, float]:
    """Project a view-direction prior onto a collision-free local exit ray.

    A distant goal bearing is not an initial path tangent when a wall forces the
    native planner to detour. Restricting this projection to short, directly
    traversable rays makes the stored start heading compatible with the actual
    outgoing navigation direction without imposing a shared heading.
    """
    source = np.asarray(source_xy, dtype=np.float64)
    candidates = np.asarray(candidates_xy, dtype=np.float64)
    if (
        source.shape != (2,)
        or candidates.ndim != 2
        or candidates.shape[1] != 2
        or not 0.0 < minimum_probe_m <= maximum_probe_m
        or validation_spacing_m <= 0.0
    ):
        raise ValueError("invalid local heading projection inputs")
    offsets = candidates - source
    distances = np.linalg.norm(offsets, axis=1)
    eligible = np.flatnonzero(
        (distances >= minimum_probe_m) & (distances <= maximum_probe_m)
    )
    if not len(eligible):
        raise SampleRejected("initial_heading_no_local_exit")
    bearings = np.arctan2(offsets[eligible, 1], offsets[eligible, 0])
    errors = np.abs((bearings - desired_yaw + np.pi) % (2.0 * np.pi) - np.pi)
    # Deterministic secondary key prefers a shorter validation ray.
    order = np.lexsort((distances[eligible], errors))
    for local_index in order:
        candidate_index = int(eligible[int(local_index)])
        sample_count = max(
            2, int(np.ceil(distances[candidate_index] / validation_spacing_m)) + 1
        )
        ray = np.linspace(source, candidates[candidate_index], sample_count)
        if is_path_traversable(ray):
            return float(bearings[int(local_index)]), float(errors[int(local_index)])
    raise SampleRejected("initial_heading_no_traversable_local_exit")


def soft_anchor_candidate_order(
    indices: Sequence[int],
    candidates_xy: Sequence[np.ndarray],
    anchor_xy: np.ndarray,
    scale_m: float,
    rng: np.random.Generator,
    *,
    probability_floor: float = 0.05,
) -> np.ndarray:
    """Return a complete weighted permutation around an anchor.

    The non-zero probability floor makes this a soft spatial prior: distant
    candidates remain eligible and there is no hidden cluster-radius cutoff.
    """
    candidate_indices = np.asarray(indices, dtype=np.int64)
    points = np.asarray(candidates_xy, dtype=np.float64)
    anchor = np.asarray(anchor_xy, dtype=np.float64)
    if scale_m <= 0.0 or not 0.0 < probability_floor <= 1.0:
        raise ValueError("invalid soft spatial-prior parameters")
    if candidate_indices.ndim != 1 or anchor.shape != (2,):
        raise ValueError("indices must be 1D and anchor_xy must have shape [2]")
    if not len(candidate_indices):
        return candidate_indices
    distances = np.linalg.norm(points[candidate_indices, :2] - anchor, axis=1)
    weights = probability_floor + np.exp(-0.5 * (distances / scale_m) ** 2)
    uniforms = np.maximum(rng.random(len(candidate_indices)), np.finfo(float).tiny)
    race = -np.log(uniforms) / weights
    return candidate_indices[np.argsort(race)]


def sample_region_balanced_positions(
    candidates_xyz: Sequence[np.ndarray],
    region_ids: Sequence[str],
    region_weights: Mapping[str, float],
    rng: np.random.Generator,
    *,
    count: int = 3,
    minimum_pairwise_distance_m: float = 0.6,
    is_valid: Callable[[np.ndarray, tuple[np.ndarray, ...]], bool] | None = None,
    maximum_attempts: int = 200,
) -> tuple[np.ndarray, ...]:
    """Sample separated positions without a compactness or shared-focus objective."""
    candidates = np.asarray(candidates_xyz, dtype=np.float64)
    labels = np.asarray(region_ids, dtype=object)
    if len(candidates) != len(labels) or len(candidates) < count:
        raise SampleRejected("insufficient_traversable_candidates", {"available": len(candidates)})
    regions = tuple(sorted(set(map(str, labels))))
    probabilities = np.asarray([float(region_weights.get(name, 1.0)) for name in regions])
    probabilities /= probabilities.sum()
    for _ in range(maximum_attempts):
        selection: list[np.ndarray] = []
        for _robot in range(count):
            region = str(rng.choice(regions, p=probabilities))
            indices = np.flatnonzero(labels == region)
            accepted = None
            for index in rng.permutation(indices):
                point = candidates[int(index)]
                if any(np.linalg.norm(point[:2] - other[:2]) < minimum_pairwise_distance_m for other in selection):
                    continue
                if is_valid is not None and not is_valid(point, tuple(selection)):
                    continue
                accepted = point
                break
            if accepted is None:
                break
            selection.append(accepted)
        if len(selection) == count:
            return tuple(selection)
    raise SampleRejected("region_balanced_robot_placement_failed", {"region_count": len(regions)})
