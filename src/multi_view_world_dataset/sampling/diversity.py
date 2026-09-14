from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def stable_seed(root_seed: int, *identifiers: object) -> int:
    """Return a stable uint32 seed independent of enumeration order."""
    payload = "\x1f".join([str(int(root_seed)), *(str(value) for value in identifiers)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def choose_weighted_label(weights: Mapping[str, float], rng: np.random.Generator) -> str:
    labels = tuple(sorted(weights))
    probabilities = np.asarray([float(weights[label]) for label in labels], dtype=np.float64)
    if not labels or np.any(probabilities < 0) or probabilities.sum() <= 0:
        raise ValueError("weights must be a non-empty non-negative distribution")
    probabilities /= probabilities.sum()
    return str(rng.choice(labels, p=probabilities))


def _wrap(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def joint_trajectory_metrics(trajectories: Sequence[Any]) -> dict[str, Any]:
    positions = np.stack([item.base_to_world[:, :2, 3] for item in trajectories])
    yaws = np.stack([
        np.unwrap(np.arctan2(item.base_to_world[:, 1, 0], item.base_to_world[:, 0, 0]))
        for item in trajectories
    ])
    prior_errors = np.asarray(
        [float(item.metadata.get("initial_heading_prior_error_rad", np.nan)) for item in trajectories],
        dtype=np.float64,
    )
    valid_prior = np.isfinite(prior_errors)
    mean_prior_error = float(np.mean(prior_errors[valid_prior])) if np.any(valid_prior) else 0.0
    mean_prior_alignment = (
        float(np.mean(0.5 * (1.0 + np.cos(prior_errors[valid_prior]))))
        if np.any(valid_prior) else 0.0
    )
    pair_distances: list[np.ndarray] = []
    heading_differences: list[np.ndarray] = []
    path_similarities: list[float] = []
    velocity_correlations: list[float] = []
    for left in range(len(trajectories)):
        for right in range(left + 1, len(trajectories)):
            pair_distances.append(np.linalg.norm(positions[left] - positions[right], axis=1))
            heading_differences.append(np.abs(_wrap(yaws[left] - yaws[right])))
            left_delta = np.diff(positions[left], axis=0)
            right_delta = np.diff(positions[right], axis=0)
            left_norm = np.linalg.norm(left_delta, axis=1)
            right_norm = np.linalg.norm(right_delta, axis=1)
            valid = (left_norm > 1e-9) & (right_norm > 1e-9)
            cosine = np.sum(left_delta[valid] * right_delta[valid], axis=1) / (
                left_norm[valid] * right_norm[valid]
            ) if np.any(valid) else np.asarray([1.0])
            path_similarities.append(float(np.mean(cosine)))
            if len(left_norm) > 1 and np.std(left_norm) > 1e-9 and np.std(right_norm) > 1e-9:
                velocity_correlations.append(float(np.corrcoef(left_norm, right_norm)[0, 1]))
            else:
                velocity_correlations.append(float(np.mean(cosine)))
    distances = np.stack(pair_distances)
    headings = np.stack(heading_differences)
    all_xy = positions.reshape(-1, 2)
    extent = np.ptp(all_xy, axis=0)
    return {
        "minimum_inter_robot_distance_m": float(distances.min()),
        "mean_inter_robot_distance_m": float(distances.mean()),
        "maximum_inter_robot_distance_m": float(distances.max()),
        "mean_pairwise_heading_difference_rad": float(headings.mean()),
        "maximum_pairwise_heading_difference_rad": float(headings.max()),
        "mean_initial_heading_prior_error_rad": mean_prior_error,
        "mean_initial_heading_prior_alignment": mean_prior_alignment,
        "mean_path_direction_similarity": float(np.mean(path_similarities)),
        "maximum_path_direction_similarity": float(np.max(path_similarities)),
        "mean_velocity_profile_correlation": float(np.mean(velocity_correlations)),
        "spatial_coverage_bbox_area_m2": float(extent[0] * extent[1]),
        "spatial_coverage_trace_m": float(sum(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in positions)),
    }


def regime_trajectory_soft_score(
    metrics: Mapping[str, Any],
    regime: str,
    coverage_saturation_m2: Mapping[str, float],
    *,
    jitter: float = 0.0,
    heading_prior_weights: Mapping[str, float] | None = None,
) -> float:
    """Score useful spatial diversity without rewarding unbounded dispersion."""
    if regime not in coverage_saturation_m2:
        raise ValueError(f"missing coverage saturation for regime {regime}")
    saturation = float(coverage_saturation_m2[regime])
    if not np.isfinite(saturation) or saturation <= 0.0:
        raise ValueError("coverage saturation values must be finite and positive")
    coverage = float(metrics["spatial_coverage_bbox_area_m2"])
    normalized_coverage = min(max(coverage, 0.0) / saturation, 1.0)
    score = normalized_coverage
    prior_weights = heading_prior_weights or {}
    prior_weight = float(prior_weights.get(regime, 0.0))
    if not np.isfinite(prior_weight) or prior_weight < 0.0:
        raise ValueError("heading-prior soft-score weights must be finite and non-negative")
    score += prior_weight * float(metrics.get("mean_initial_heading_prior_alignment", 0.0))
    if regime == "exploratory":
        separation = max(float(metrics["mean_inter_robot_distance_m"]), 0.0)
        score += 0.15 * min(separation / np.sqrt(saturation), 1.0)
    return float(score + jitter)


def formation_degenerate(metrics: Mapping[str, Any], limits: Mapping[str, float]) -> bool:
    """Distribution-control rejection for near-parallel duplicate formations."""
    return bool(
        float(metrics["mean_pairwise_heading_difference_rad"])
        < np.deg2rad(float(limits["minimum_mean_heading_difference_deg"]))
        and float(metrics["mean_path_direction_similarity"])
        > float(limits["maximum_mean_path_similarity"])
        and float(metrics["spatial_coverage_bbox_area_m2"])
        < float(limits["minimum_spatial_coverage_m2"])
    )


def temporal_overlap_acceptance(
    robot_ids: Sequence[str],
    keyframes: Sequence[Mapping[str, Any]],
    *,
    regime: str,
    regime_connected_fraction_target: Mapping[str, float],
    regime_shared_keyframe_fraction_target: Mapping[str, float],
    regime_maximum_consecutive_isolated_keyframes: Mapping[str, int],
) -> dict[str, Any]:
    """Evaluate hard episode connectivity and report soft overlap-regime targets.

    G_union connectivity, robot participation, non-degenerate views, and
    regime-aware long-term isolation are hard constraints. Per-frame graph
    connectivity is a target / QA metric rather than a universal hard gate.
    """
    ids = tuple(robot_ids)
    union_edges: set[tuple[str, str]] = set()
    participation = {robot_id: 0 for robot_id in ids}
    isolation_runs = {robot_id: 0 for robot_id in ids}
    maximum_runs = {robot_id: 0 for robot_id in ids}
    connected_count = 0
    meaningful_shared_count = 0
    near_duplicate_count = 0
    for frame in keyframes:
        edges = {tuple(sorted(map(str, edge))) for edge in frame.get("edges", ())}
        union_edges.update(edges)
        connected_count += int(bool(frame.get("connected", False)))
        meaningful_shared_count += int(bool(edges))
        near_duplicate_count += len(frame.get("near_duplicate_pairs", ()))
        incident = {robot_id: False for robot_id in ids}
        for left, right in edges:
            incident[left] = True
            incident[right] = True
        for robot_id in ids:
            participation[robot_id] += int(incident[robot_id])
            isolation_runs[robot_id] = 0 if incident[robot_id] else isolation_runs[robot_id] + 1
            maximum_runs[robot_id] = max(maximum_runs[robot_id], isolation_runs[robot_id])
    reached = {ids[0]} if ids else set()
    changed = True
    while changed:
        changed = False
        for left, right in union_edges:
            if left in reached and right not in reached:
                reached.add(right); changed = True
            if right in reached and left not in reached:
                reached.add(left); changed = True
    count = max(1, len(keyframes))
    connected_fraction = connected_count / count
    shared_keyframe_fraction = meaningful_shared_count / count
    union_is_tree = len(union_edges) == max(0, len(ids) - 1)
    if connected_fraction >= float(regime_connected_fraction_target["dense_shared"]):
        realized_regime = "dense_shared"
    elif (
        union_is_tree
        and shared_keyframe_fraction
        >= float(regime_shared_keyframe_fraction_target["partial_chain"])
    ):
        realized_regime = "partial_chain"
    else:
        realized_regime = "exploratory"
    allowed_isolation = int(
        regime_maximum_consecutive_isolated_keyframes[realized_regime]
    )
    connected_target = float(regime_connected_fraction_target[regime])
    shared_target = float(regime_shared_keyframe_fraction_target[regime])
    checks = {
        "union_graph_connected": len(reached) == len(ids),
        "meaningful_shared_moment": meaningful_shared_count > 0,
        "every_robot_participates": all(value > 0 for value in participation.values()),
        "no_severe_isolation": all(value <= allowed_isolation for value in maximum_runs.values()),
        "no_near_duplicate_views": near_duplicate_count == 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "union_edges": [list(edge) for edge in sorted(union_edges)],
        "connected_keyframe_count": connected_count,
        "keyframe_count": len(keyframes),
        "connected_fraction": connected_fraction,
        "connected_fraction_target": connected_target,
        "connected_fraction_target_met": connected_fraction >= connected_target,
        "meaningful_shared_keyframe_count": meaningful_shared_count,
        "shared_keyframe_fraction": shared_keyframe_fraction,
        "shared_keyframe_fraction_target": shared_target,
        "shared_keyframe_fraction_target_met": shared_keyframe_fraction >= shared_target,
        "participating_keyframe_count": participation,
        "maximum_consecutive_isolated_keyframes": maximum_runs,
        "allowed_consecutive_isolated_keyframes": allowed_isolation,
        "near_duplicate_keyframe_pair_count": near_duplicate_count,
        "requested_regime": regime,
        "realized_regime": realized_regime,
        "regime_target_match": regime == realized_regime,
        "regime": realized_regime,
    }

def complementary_hybrid_trajectory_sets(
    candidate_sets: Sequence[tuple[Sequence[Any], Mapping[str, Any]]],
    candidate_failures: Sequence[Mapping[str, Any]],
    *,
    minimum_pairwise_distance_m: float,
    minimum_waypoint_trajectories: int,
    formation_degeneracy_limits: Mapping[str, float],
    coverage_saturation_m2: Mapping[str, float],
    heading_prior_weights: Mapping[str, float],
    maximum_candidates: int,
) -> tuple[tuple[tuple[Any, ...], dict[str, int], dict[str, Any]], ...]:
    """Recombine only candidate pairs whose measured overlap edges are complementary."""
    if maximum_candidates <= 0 or len(candidate_sets) < 2:
        return ()
    robot_ids = tuple(sorted(item.robot_id for item in candidate_sets[0][0]))
    robot_id_set = set(robot_ids)

    def _edges(failure: Mapping[str, Any]) -> set[tuple[str, str]]:
        if failure.get("reason") != "trajectory_temporal_overlap_failed":
            return set()
        return {
            tuple(sorted((str(edge[0]), str(edge[1]))))
            for edge in failure.get("details", {}).get("union_edges", ())
            if len(edge) == 2 and set(map(str, edge)).issubset(robot_id_set)
        }

    def _connected(edges: set[tuple[str, str]]) -> bool:
        if not robot_ids:
            return False
        reached = {robot_ids[0]}
        changed = True
        while changed:
            changed = False
            for left, right in edges:
                if left in reached and right not in reached:
                    reached.add(right)
                    changed = True
                if right in reached and left not in reached:
                    reached.add(left)
                    changed = True
        return len(reached) == len(robot_ids)

    failure_by_rank = {
        int(item["candidate_rank"]): item
        for item in candidate_failures
        if isinstance(item.get("candidate_rank"), int)
    }
    pools = []
    for trajectories, metrics in candidate_sets:
        by_robot = {item.robot_id: item for item in trajectories}
        if tuple(sorted(by_robot)) != robot_ids:
            raise ValueError("all trajectory sets must contain identical robot IDs")
        pools.append((by_robot, metrics))

    ranked: list[
        tuple[float, tuple[int, ...], tuple[Any, ...], dict[str, int], dict[str, Any]]
    ] = []
    seen_source_assignments: set[tuple[int, ...]] = set()
    for left_index in range(len(pools)):
        for right_index in range(left_index + 1, len(pools)):
            left_edges = _edges(failure_by_rank.get(left_index, {}))
            right_edges = _edges(failure_by_rank.get(right_index, {}))
            if not _connected(left_edges | right_edges):
                continue
            for mask in range(1, (1 << len(robot_ids)) - 1):
                source_indices = tuple(
                    right_index if mask & (1 << robot_index) else left_index
                    for robot_index in range(len(robot_ids))
                )
                if source_indices in seen_source_assignments:
                    continue
                seen_source_assignments.add(source_indices)
                trajectories = tuple(
                    pools[source_index][0][robot_id]
                    for robot_id, source_index in zip(robot_ids, source_indices)
                )
                if sum(item.path_family != "direct" for item in trajectories) < minimum_waypoint_trajectories:
                    continue
                metrics = joint_trajectory_metrics(trajectories)
                if float(metrics["minimum_inter_robot_distance_m"]) < minimum_pairwise_distance_m:
                    continue
                if formation_degenerate(metrics, formation_degeneracy_limits):
                    continue
                regime = str(pools[source_indices[0]][1]["observation_regime"])
                if any(
                    str(pools[source_index][1]["observation_regime"]) != regime
                    for source_index in source_indices
                ):
                    continue
                predicted_edges = {
                    edge for edge in left_edges
                    if all(source_indices[robot_ids.index(robot_id)] == left_index for robot_id in edge)
                } | {
                    edge for edge in right_edges
                    if all(source_indices[robot_ids.index(robot_id)] == right_index for robot_id in edge)
                }
                participating = {robot_id for edge in predicted_edges for robot_id in edge}
                priority = (
                    4.0 * float(_connected(predicted_edges))
                    + float(len(predicted_edges))
                    + float(len(participating)) / max(1, len(robot_ids))
                    + regime_trajectory_soft_score(
                        metrics,
                        regime,
                        coverage_saturation_m2,
                        heading_prior_weights=heading_prior_weights,
                    )
                )
                source_by_robot = dict(zip(robot_ids, source_indices))
                metrics["formation_degenerate"] = False
                metrics["complementary_hybrid"] = {
                    "source_candidate_by_robot": source_by_robot,
                    "source_candidate_pair": [left_index, right_index],
                    "source_union_edges": [list(edge) for edge in sorted(left_edges | right_edges)],
                    "predicted_preserved_edges": [list(edge) for edge in sorted(predicted_edges)],
                    "soft_priority": float(priority),
                }
                ranked.append(
                    (priority, source_indices, trajectories, source_by_robot, metrics)
                )
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return tuple(
        (trajectories, source_by_robot, metrics)
        for _, _, trajectories, source_by_robot, metrics in ranked[:maximum_candidates]
    )

