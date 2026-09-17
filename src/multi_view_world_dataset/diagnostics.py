from __future__ import annotations

from collections import Counter
from itertools import combinations
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from multi_view_world_dataset.adapters.omnigibson import OmniGibsonAdapter
from multi_view_world_dataset.adapters.navigation import footprint_physx_diagnostic
from multi_view_world_dataset.errors import SampleRejected
from multi_view_world_dataset.generator import _temporal_overlap_preflight
from multi_view_world_dataset.rendering.inspection import save_trajectory_inspection
from multi_view_world_dataset.sampling.diversity import stable_seed
from multi_view_world_dataset.utils.runtime import RuntimePaths
from multi_view_world_dataset.utils.serialization import dump_json


def run_sampling_diagnostics(
    runtime: RuntimePaths,
    config: dict[str, Any],
    *,
    scene_id: str | None = None,
    samples: int | None = None,
    include_overlap_preflight: bool | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Sample placements and geodesic paths without dense episode rendering."""
    adapter = OmniGibsonAdapter(runtime, config)
    requested = int(samples or config["sampling_diagnostics"]["sample_count"])
    regimes: Counter[str] = Counter()
    path_families: Counter[str] = Counter()
    regions: Counter[str] = Counter()
    rejects: Counter[str] = Counter()
    rejection_details: list[dict[str, Any]] = []
    lengths: list[float] = []
    tortuosities: list[float] = []
    coverages: list[float] = []
    start_distances: list[float] = []
    displacements: list[float] = []
    yaw_changes: list[float] = []
    heading_differences: list[float] = []
    heading_prior_errors: list[float] = []
    heading_prior_alignments: list[float] = []
    path_similarities: list[float] = []
    velocity_correlations: list[float] = []
    view_connectivity_proxies: list[float] = []
    minimum_separations: list[float] = []
    traversed_regions: Counter[str] = Counter()
    overlap_values: list[float] = []
    overlap_topologies: Counter[str] = Counter()
    connected_fractions: list[float] = []
    union_connected = 0
    overlap_evaluated = 0
    overlap_passed = 0
    near_duplicate_keyframes = 0
    run_overlap = (
        bool(config["sampling_diagnostics"].get("include_overlap_preflight", False))
        if include_overlap_preflight is None
        else bool(include_overlap_preflight)
    )
    accepted = 0
    inspection_trajectories = None
    inspection_overlap = None
    inspection_floor_index = None
    footprint_report = None
    try:
        adapter.start()
        selected_scene = scene_id or adapter.discover_scenes()[0]
        adapter.load_scene(
            selected_scene,
            robot_count=3,
            development_robot=(
                config["robot"]["final_model"]
                if config["robot"]["use_final_robot"]
                else config["robot"]["development_model"]
            ),
        )
        snapshot = adapter.dump_snapshot()
        catalog = adapter.object_catalog_with_relations()
        adapter.prepare_navigation_context(
            "sampling-diagnostics",
            stable_seed(int(config["seed"]), selected_scene, "navigation-context"),
            force=True,
        )
        navigation_report = adapter.navigation_context_metadata()
        for index in range(requested):
            adapter.load_snapshot(snapshot)
            try:
                _, candidate_sets = adapter.sample_route_first_trajectory_sets(
                    stable_seed(
                        int(config["seed"]), selected_scene,
                        "diagnostic-route-first", index,
                    )
                )
                trajectories, metrics = candidate_sets[0]
                if footprint_report is None:
                    footprint_report = footprint_physx_diagnostic(
                        adapter,
                        int(metrics["floor_index"]),
                        stable_seed(
                            int(config["seed"]), selected_scene,
                            "footprint-physx-diagnostic",
                        ),
                    )
                overlap = None
                if run_overlap:
                    selected = None
                    for candidate_rank, (
                        candidate_trajectories,
                        candidate_metrics,
                    ) in enumerate(candidate_sets):
                        overlap_evaluated += 1
                        adapter.place_robots_at_trajectory_frame(
                            candidate_trajectories, 0
                        )
                        try:
                            candidate_overlap = _temporal_overlap_preflight(
                                adapter,
                                config,
                                candidate_trajectories,
                                catalog=catalog,
                            )
                        except SampleRejected as error:
                            rejects[error.reason] += 1
                            rejection_details.append({
                                "sample_index": index,
                                "candidate_rank": candidate_rank,
                                "route_ids": candidate_metrics["route_ids"],
                                "reason": error.reason,
                                "details": error.details,
                            })
                            if inspection_trajectories is None:
                                inspection_trajectories = candidate_trajectories
                                inspection_overlap = (
                                    error.details
                                    if error.reason
                                    == "trajectory_temporal_overlap_failed"
                                    else None
                                )
                                inspection_floor_index = int(
                                    candidate_metrics["floor_index"]
                                )
                            continue
                        selected = (
                            candidate_trajectories,
                            candidate_metrics,
                            candidate_overlap,
                        )
                        break
                    if selected is None:
                        accepted += 1
                        continue
                    trajectories, metrics, overlap = selected
                    overlap_passed += 1
                    inspection_trajectories = trajectories
                    inspection_overlap = overlap
                    inspection_floor_index = int(metrics["floor_index"])
            except SampleRejected as error:
                rejects[error.reason] += 1
                rejection_details.append({
                    "sample_index": index,
                    "reason": error.reason,
                    "details": error.details,
                })
                continue
            accepted += 1
            regions.update(map(str, metrics["start_region_ids"]))
            joint = metrics["joint_diversity"]
            coverages.append(float(joint["spatial_coverage_bbox_area_m2"]))
            heading_differences.append(float(joint["mean_pairwise_heading_difference_rad"]))
            heading_prior_errors.append(float(joint["mean_initial_heading_prior_error_rad"]))
            heading_prior_alignments.append(
                float(joint["mean_initial_heading_prior_alignment"])
            )
            path_similarities.append(float(joint["mean_path_direction_similarity"]))
            velocity_correlations.append(float(joint["mean_velocity_profile_correlation"]))
            view_connectivity_proxies.append(float(
                joint["temporal_camera_view_connectivity_proxy"]
            ))
            minimum_separations.append(float(joint["minimum_inter_robot_distance_m"]))
            positions = [trajectory.base_to_world[0, :2, 3] for trajectory in trajectories]
            start_distances.extend(
                float(np.linalg.norm(positions[left] - positions[right]))
                for left, right in combinations(range(len(positions)), 2)
            )
            for values in metrics["traversed_region_ids"].values():
                traversed_regions.update(map(str, values))
            for trajectory in trajectories:
                path_families[trajectory.path_family] += 1
                robot = metrics["robots"][trajectory.robot_id]
                lengths.append(float(robot["arc_path_length_m"]))
                displacements.append(float(robot["start_end_displacement_m"]))
                tortuosities.append(float(robot["tortuosity"]))
                yaw_changes.append(float(robot["cumulative_absolute_yaw_change_rad"]))
            if run_overlap:
                assert overlap is not None
                connected_fractions.append(float(overlap["connected_fraction"]))
                union_connected += int(bool(overlap["checks"]["union_graph_connected"]))
                near_duplicate_keyframes += int(overlap["near_duplicate_keyframe_pair_count"])
                for keyframe in overlap["keyframes"]:
                    edges = len(keyframe["edges"])
                    topology = {0: "sparse", 1: "single_edge", 2: "chain", 3: "triangle"}.get(
                        edges, f"edges_{edges}"
                    )
                    overlap_topologies[topology] += 1
                    overlap_values.extend(float(value) for value in keyframe["overlaps"].values())
                regimes[str(overlap["realized_regime"])] += 1
            else:
                regimes[str(metrics["observation_regime"])] += 1
            if inspection_trajectories is None:
                inspection_trajectories = trajectories
                inspection_overlap = overlap if run_overlap else None
                inspection_floor_index = int(metrics["floor_index"])

        movable = [obj for obj in catalog if obj.movable and not obj.structural]
        report = {
            "mode": (
                "sparse_gt_depth_overlap_preflight_no_dense_rgb_rollout"
                if run_overlap else "cheap_sampling_only_no_dense_rgb_rollout"
            ),
            "scene_id": selected_scene,
            "requested_samples": requested,
            "accepted_samples": (
                overlap_passed if run_overlap else accepted
            ),
            "accepted_sample_definition": (
                "physical_and_temporal_overlap_eligible"
                if run_overlap else "physically_valid_proposal"
            ),
            "physically_valid_proposal_samples": accepted,
            "physical_proposal_acceptance_rate": accepted / max(1, requested),
            "episode_eligible_samples": overlap_passed if run_overlap else accepted,
            "acceptance_rate": (
                overlap_passed if run_overlap else accepted
            ) / max(1, requested),
            "rejection_reasons": dict(rejects),
            "rejection_details": rejection_details,
            "observation_regimes": dict(regimes),
            "path_families": dict(path_families),
            "start_regions": dict(regions),
            "traversed_regions": dict(traversed_regions),
            "pairwise_start_distance_m": _summary(start_distances),
            "arc_path_length_m": _summary(lengths),
            "start_end_displacement_m": _summary(displacements),
            "tortuosity": _summary(tortuosities),
            "cumulative_absolute_yaw_change_rad": _summary(yaw_changes),
            "mean_pairwise_heading_difference_rad": _summary(heading_differences),
            "mean_initial_heading_prior_error_rad": _summary(heading_prior_errors),
            "mean_initial_heading_prior_alignment": _summary(heading_prior_alignments),
            "mean_path_direction_similarity": _summary(path_similarities),
            "mean_velocity_profile_correlation": _summary(velocity_correlations),
            "temporal_camera_view_connectivity_proxy": _summary(
                view_connectivity_proxies
            ),
            "spatial_coverage_m2": _summary(coverages),
            "minimum_inter_robot_distance_m": _summary(minimum_separations),
            "navigation_context": navigation_report,
            "footprint_physx_diagnostic": footprint_report,
            "configuration_candidates": {
                "movable_non_structural": len(movable),
                "categories": dict(Counter(obj.category for obj in movable)),
                "rooms": dict(Counter(obj.room_id or "unknown" for obj in movable)),
            },
            "intervention_candidates": {
                "rigid_relocation": sum(obj.movable for obj in movable),
                "articulation": sum(obj.articulated for obj in movable),
                "state_change": sum(bool(obj.available_states) for obj in movable),
            },
            "temporal_overlap": {
                "enabled": run_overlap,
                "evaluated_samples": overlap_evaluated,
                "passed_samples": overlap_passed,
                "union_connected_samples": union_connected,
                "connected_fraction": _summary(connected_fractions),
                "pairwise_omega": _summary(overlap_values),
                "topology_keyframe_counts": dict(overlap_topologies),
                "near_duplicate_keyframe_pair_count": near_duplicate_keyframes,
            },
        }
        diagnostics_config = config["sampling_diagnostics"]
        collapse_threshold = float(
            diagnostics_config["collapse_warning_rate"]
        )
        thresholds = diagnostics_config["collapse_thresholds"]
        minimum_warning_samples = int(
            diagnostics_config["minimum_samples_for_distribution_warnings"]
        )
        warnings = []
        if report["acceptance_rate"] < collapse_threshold:
            warnings.append("sampling_acceptance_collapse")
        if accepted and requested >= minimum_warning_samples:
            if len(regimes) < 2:
                warnings.append("observation_regime_collapse")
            if len(path_families) < 2:
                warnings.append("path_family_collapse")
            if (
                max(regions.values(), default=0) / max(1, sum(regions.values()))
                > float(thresholds["dominant_start_region_fraction_max"])
            ):
                warnings.append("dominant_start_region")
            if (
                path_families.get("direct", 0)
                / max(1, sum(path_families.values()))
                > float(thresholds["direct_path_fraction_max"])
            ):
                warnings.append("direct_path_collapse")
            parallel_min = float(
                diagnostics_config[
                    "parallel_path_direction_similarity_min"
                ]
            )
            if (
                sum(value >= parallel_min for value in path_similarities)
                / max(1, len(path_similarities))
                > float(thresholds["parallel_episode_fraction_max"])
            ):
                warnings.append("parallel_motion_collapse")
            if run_overlap and (
                union_connected / max(1, overlap_evaluated)
                < float(
                    thresholds[
                        "temporal_union_connected_fraction_min"
                    ]
                )
            ):
                warnings.append("temporal_union_connectivity_collapse")
            if run_overlap and (
                overlap_topologies.get("triangle", 0)
                / max(1, sum(overlap_topologies.values()))
                > float(
                    thresholds[
                        "complete_triangle_keyframe_fraction_max"
                    ]
                )
            ):
                warnings.append("complete_triangle_overlap_collapse")
        report["collapse_warnings"] = warnings
        report["collapse_warning_rate"] = collapse_threshold
        report["collapse_thresholds"] = thresholds
        report["collapse_warning_evaluation_deferred"] = (
            requested < minimum_warning_samples
        )
        root = runtime.require_output()
        if inspection_trajectories is not None and inspection_floor_index is not None:
            inspection_name = "navigation_diagnostic.png"
            save_trajectory_inspection(
                root / inspection_name,
                adapter.trajectory_traversability_inspection(
                    inspection_floor_index
                ),
                inspection_trajectories,
                inspection_overlap,
            )
            report["inspection_image"] = inspection_name
        path = root / str(config["sampling_diagnostics"]["output_name"])
        dump_json(path, report)
        (root / "sampling_diagnostics_failure.json").unlink(missing_ok=True)
        return path, report
    except BaseException as error:
        root = runtime.require_output()
        dump_json(
            root / "sampling_diagnostics_failure.json",
            {
                "error_type": type(error).__name__,
                "message": str(error),
                "reason": getattr(error, "reason", None),
                "details": getattr(error, "details", {}),
                "runtime_findings": adapter.runtime_report(),
            },
        )
        traceback.print_exc()
        raise
    finally:
        adapter.close()


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values), "minimum": float(array.min()),
        "mean": float(array.mean()), "maximum": float(array.max()),
        "p10": float(np.quantile(array, 0.1)), "p90": float(np.quantile(array, 0.9)),
    }
