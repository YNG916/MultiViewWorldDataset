from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from multi_view_world_dataset.adapters.omnigibson import OmniGibsonAdapter
from multi_view_world_dataset.cameras.calibration import PinholeCalibration
from multi_view_world_dataset.cameras.overlap import build_overlap_graph
from multi_view_world_dataset.cameras.transforms import invert_transform
from multi_view_world_dataset.errors import ConfigurationError, SampleRejected
from multi_view_world_dataset.pipeline import _bev_geometry_metrics
from multi_view_world_dataset.qa.checks import check_bev_pair, check_paired_trajectories, require_all
from multi_view_world_dataset.rendering.inspection import save_rgb, save_trajectory_inspection, write_html_summary
from multi_view_world_dataset.sampling.configurations import near_duplicate_configuration
from multi_view_world_dataset.sampling.splits import assign_scene_family_splits
from multi_view_world_dataset.schema.records import (
    CameraState,
    DynamicConfiguration,
    ObjectState,
    Observation,
    QAResult,
    RobotState,
    WorldEpisode,
    WorldState,
)
from multi_view_world_dataset.storage.writer import DatasetWriter
from multi_view_world_dataset.utils.runtime import RuntimePaths, generator_git_commit
from multi_view_world_dataset.utils.serialization import dump_json


def _write_status(root: Path, **values: Any) -> None:
    dump_json(root / "generation_status.json", values)


def _render_environment_floors(
    adapter: OmniGibsonAdapter,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[int, Any]]:
    bev_config = config["bev"]
    arrays: dict[str, np.ndarray] = {}
    calibrations: dict[int, Any] = {}
    floor_count = len(adapter.scene_record("unassigned").floor_ids)
    for floor_index in range(floor_count):
        calibration = adapter.calibrated_floor_bounds(
            floor_index,
            float(bev_config["environment_meters_per_pixel"]),
            float(bev_config["bounds_margin_m"]),
        )
        render = adapter.render_floor_bev(
            floor_index,
            calibration,
            include_robots=False,
            modalities=tuple(bev_config["modalities"]),
        )
        geometry_ok, geometry = _bev_geometry_metrics(render)
        if not geometry_ok or render.projection_token != "orthographic":
            raise SampleRejected(
                "environment_bev_geometry_failed",
                {"floor_index": floor_index, "geometry": geometry},
            )
        instance_info = render.metadata.get("segmentation_info", {}).get("seg_instance_id", {})
        robot_raw_ids = [
            int(raw_id)
            for raw_id, label in instance_info.items()
            if any(f"robot_{index:02d}" in str(label) for index in range(3))
        ]
        labels = np.asarray(render.modalities["instance_id"]).squeeze()
        robot_pixels = int(np.isin(labels, robot_raw_ids).sum())
        if robot_pixels:
            raise SampleRejected(
                "environment_bev_contains_robot",
                {"floor_index": floor_index, "robot_pixels": robot_pixels},
            )
        prefix = f"floor_{floor_index:02d}"
        for name, value in render.modalities.items():
            arrays[f"{prefix}/{name}"] = np.asarray(value)
        arrays[f"{prefix}/calibration_world_bounds"] = np.asarray(calibration.world_bounds)
        arrays[f"{prefix}/calibration_pixel_to_world"] = calibration.pixel_to_world_transform
        arrays[f"{prefix}/calibration_meters_per_pixel"] = np.asarray(
            calibration.meters_per_pixel
        )
        arrays[f"{prefix}/calibration_floor_z"] = np.asarray(calibration.floor_z)
        calibrations[floor_index] = calibration
    return arrays, calibrations


def _initial_overlap(
    adapter: OmniGibsonAdapter,
    config: dict[str, Any],
) -> tuple[Any, dict[str, dict[str, Any]]]:
    observations = adapter.robot_observations()
    camera_config = config["camera"]
    calibration = PinholeCalibration(
        int(camera_config["geometry_width"]),
        int(camera_config["geometry_height"]),
        float(camera_config["hfov_deg"]),
        float(camera_config["near_m"]),
        float(camera_config["far_m"]),
    )
    depths = {
        robot_id: np.asarray(record["modalities"]["depth_linear"]).squeeze()[::2, ::2]
        for robot_id, record in observations.items()
    }
    overlap_config = config["overlap"]
    graph = build_overlap_graph(
        tuple(sorted(observations)),
        depths,
        {robot_id: calibration.pixel_intrinsics for robot_id in observations},
        {robot_id: record["camera_to_world"] for robot_id, record in observations.items()},
        edge_threshold=float(overlap_config["edge_threshold"]),
        near_duplicate_threshold=float(overlap_config["near_duplicate_threshold"]),
        stride=int(overlap_config["depth_sample_stride"]),
        tolerance_m=float(overlap_config["reprojection_tolerance_m"]),
    )
    if not graph.connected or graph.near_duplicate_pairs:
        raise SampleRejected(
            "initial_overlap_failed",
            {
                "connected": graph.connected,
                "near_duplicate_pairs": graph.near_duplicate_pairs,
                "overlaps": graph.overlaps,
            },
        )
    return graph, observations



def _temporal_overlap_preflight(
    adapter: OmniGibsonAdapter,
    config: dict[str, Any],
    trajectories: tuple[Any, ...],
) -> dict[str, Any]:
    """Validate sparse synchronized GT-depth overlap before full rollout capture."""
    preflight = config["trajectory"]["overlap_preflight"]
    frame_count = trajectories[0].frames
    keyframe_indices = np.unique(
        np.rint(np.linspace(0, frame_count - 1, int(preflight["keyframe_count"]))).astype(int)
    )
    width = int(preflight["geometry_width"])
    height = int(preflight["geometry_height"])
    camera_config = config["camera"]
    calibration = PinholeCalibration(
        width,
        height,
        float(camera_config["hfov_deg"]),
        float(camera_config["near_m"]),
        float(camera_config["far_m"]),
    )
    overlap_config = config["overlap"]
    robot_ids = tuple(sorted(trajectory.robot_id for trajectory in trajectories))
    connected_count = 0
    isolation_runs = {robot_id: 0 for robot_id in robot_ids}
    maximum_isolation_runs = {robot_id: 0 for robot_id in robot_ids}
    keyframes: list[dict[str, Any]] = []
    try:
        for frame_index in keyframe_indices:
            adapter.place_robots_at_trajectory_frame(trajectories, int(frame_index))
            observations = adapter.robot_observations()
            depths = {}
            for robot_id, record in observations.items():
                depth = np.asarray(record["modalities"]["depth_linear"]).squeeze()
                rows = np.rint(np.linspace(0, depth.shape[0] - 1, height)).astype(np.int64)
                columns = np.rint(np.linspace(0, depth.shape[1] - 1, width)).astype(np.int64)
                depths[robot_id] = depth[rows[:, None], columns[None, :]]
            graph = build_overlap_graph(
                robot_ids,
                depths,
                {robot_id: calibration.pixel_intrinsics for robot_id in robot_ids},
                {robot_id: observations[robot_id]["camera_to_world"] for robot_id in robot_ids},
                edge_threshold=float(overlap_config["edge_threshold"]),
                near_duplicate_threshold=float(overlap_config["near_duplicate_threshold"]),
                stride=int(preflight["depth_sample_stride"]),
                tolerance_m=float(overlap_config["reprojection_tolerance_m"]),
            )
            connected_count += int(graph.connected)
            incident = {robot_id: False for robot_id in robot_ids}
            for left, right in graph.edges:
                incident[left] = True
                incident[right] = True
            isolated = []
            for robot_id in robot_ids:
                isolation_runs[robot_id] = 0 if incident[robot_id] else isolation_runs[robot_id] + 1
                maximum_isolation_runs[robot_id] = max(
                    maximum_isolation_runs[robot_id], isolation_runs[robot_id]
                )
                if not incident[robot_id]:
                    isolated.append(robot_id)
            keyframes.append(
                {
                    "frame_index": int(frame_index),
                    "connected": bool(graph.connected),
                    "edges": [list(edge) for edge in graph.edges],
                    "isolated_robot_ids": isolated,
                    "overlaps": {
                        f"{left}|{right}": float(value)
                        for (left, right), value in graph.overlaps.items()
                    },
                }
            )
    finally:
        adapter.place_robots_at_trajectory_frame(trajectories, 0)
    connected_fraction = connected_count / len(keyframe_indices)
    maximum_allowed_isolation = int(preflight["maximum_consecutive_isolated_keyframes"])
    passed = (
        connected_fraction >= float(preflight["connected_fraction_min"])
        and all(value <= maximum_allowed_isolation for value in maximum_isolation_runs.values())
    )
    metrics = {
        "keyframe_indices": keyframe_indices.tolist(),
        "connected_keyframe_count": connected_count,
        "keyframe_count": len(keyframe_indices),
        "connected_fraction": connected_fraction,
        "required_connected_fraction": float(preflight["connected_fraction_min"]),
        "maximum_consecutive_isolated_keyframes": maximum_isolation_runs,
        "allowed_consecutive_isolated_keyframes": maximum_allowed_isolation,
        "geometry_resolution": [width, height],
        "keyframes": keyframes,
    }
    if not passed:
        raise SampleRejected("trajectory_temporal_overlap_failed", metrics)
    return metrics
def _robot_states(
    config: dict[str, Any],
    heights: dict[str, float],
    trajectories: tuple[Any, ...],
) -> tuple[RobotState, ...]:
    by_id = {trajectory.robot_id: trajectory for trajectory in trajectories}
    model = (
        config["robot"]["final_model"]
        if config["robot"]["use_final_robot"]
        else config["robot"]["development_model"]
    )
    return tuple(
        RobotState(
            robot_id=robot_id,
            model=model,
            base_to_world=by_id[robot_id].base_to_world[0],
            camera_height_m=float(heights[robot_id]),
        )
        for robot_id in sorted(by_id)
    )


def _observation_records(
    config: dict[str, Any],
    trajectories: tuple[Any, ...],
    branch: str,
    view_refs: dict[str, str],
) -> tuple[Observation, ...]:
    camera_config = config["camera"]
    calibration = PinholeCalibration(
        int(camera_config["rgb_width"]),
        int(camera_config["rgb_height"]),
        float(camera_config["hfov_deg"]),
        float(camera_config["near_m"]),
        float(camera_config["far_m"]),
    )
    records: list[Observation] = []
    for trajectory in trajectories:
        for frame_index in range(trajectory.frames):
            base_to_world = trajectory.base_to_world[frame_index]
            camera_to_world = trajectory.camera_to_world[frame_index]
            camera_to_base = invert_transform(base_to_world) @ camera_to_world
            modality_refs = {
                modality: f"{view_refs[trajectory.robot_id]}::{modality}[{frame_index}]"
                for modality in ("rgb", "depth_linear", "semantic", "instance", "normal")
            }
            records.append(
                Observation(
                    robot_id=trajectory.robot_id,
                    physical_time_index=frame_index,
                    camera=CameraState(
                        camera_id=f"{trajectory.robot_id}_{branch}",
                        robot_id=trajectory.robot_id,
                        width=int(camera_config["rgb_width"]),
                        height=int(camera_config["rgb_height"]),
                        pixel_intrinsics=calibration.pixel_intrinsics,
                        normalized_intrinsics=calibration.normalized_intrinsics,
                        camera_to_world=camera_to_world,
                        world_to_camera=invert_transform(camera_to_world),
                        robot_base_to_world=base_to_world,
                        camera_to_robot_base=camera_to_base,
                        near_m=float(camera_config["near_m"]),
                        far_m=float(camera_config["far_m"]),
                        camera_height_m=float(camera_to_base[2, 3]),
                    ),
                    modality_refs=modality_refs,
                )
            )
    return tuple(records)


def _existing_event_targets(root: Path, scene_id: str, configuration_id: str) -> tuple[str, ...]:
    episode_root = root / "episodes" / scene_id / configuration_id
    if not episode_root.is_dir():
        return ()
    targets = []
    for path in sorted(episode_root.glob("*/events.json")):
        try:
            events = json.loads(path.read_text(encoding="utf-8"))
            if events:
                targets.append(str(events[0]["target_instance_id"]))
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return tuple(targets)


def _read_configuration_catalog(path: Path) -> tuple[ObjectState, ...]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Resuming configuration generation requires pyarrow") from error
    return tuple(ObjectState(**row) for row in pq.read_table(path).to_pylist())


def generate_dataset(
    runtime: RuntimePaths,
    config: dict[str, Any],
    *,
    scene_id: str | None = None,
    allow_large: bool = False,
) -> tuple[Path, dict[str, Any]]:
    profile = str(config["profile"])
    if profile not in {"smoke", "integration"} and not allow_large:
        raise ConfigurationError(
            "Refusing large pilot/default generation without explicit --allow-large"
        )
    root = runtime.require_output()
    writer = DatasetWriter(root)
    writer.initialize(
        {
            "schema_version": config["dataset"]["schema_version"],
            "profile": profile,
            "seed": int(config["seed"]),
            "source_of_truth": "world_state+simulator_snapshot+trajectory+event_log",
        }
    )
    adapter = OmniGibsonAdapter(runtime, config)
    repository_root = Path(__file__).resolve().parents[2]
    commit = generator_git_commit(repository_root)
    accepted_configurations = 0
    accepted_episodes = 0
    try:
        _write_status(root, status="running", stage="launch_simulator", profile=profile)
        adapter.start()
        scenes = adapter.discover_scenes()
        if scene_id is not None:
            if scene_id not in scenes:
                raise ConfigurationError(f"Requested scene is not installed: {scene_id}")
            scenes = [scene_id]
        scene_limit = config["dataset"].get("scene_limit")
        if scene_limit is not None:
            scenes = scenes[: int(scene_limit)]
        splits = assign_scene_family_splits(
            adapter.discover_scenes(),
            config["dataset"]["splits"],
            int(config["dataset"]["scene_family_split_seed"]),
        )
        requested_configurations = int(config["dataset"]["accepted_configurations_per_scene"])
        requested_episodes = int(config["dataset"]["accepted_episodes_per_configuration"])
        for scene_position, selected_scene in enumerate(scenes):
            _write_status(
                root,
                status="running",
                stage="load_scene",
                scene_id=selected_scene,
                scene_position=scene_position,
                accepted_configurations=accepted_configurations,
                accepted_episodes=accepted_episodes,
            )
            adapter.load_scene(
                selected_scene,
                robot_count=3,
                development_robot=(
                    config["robot"]["final_model"]
                    if config["robot"]["use_final_robot"]
                    else config["robot"]["development_model"]
                ),
            )
            base_snapshot = adapter.dump_snapshot()
            base_catalog = adapter.object_catalog_with_relations()
            scene_root = root / "scenes" / selected_scene
            if not (scene_root / "scene_meta.json").is_file():
                writer.write_scene(
                    selected_scene,
                    adapter.scene_record(splits[selected_scene]),
                    base_catalog,
                )
            completed_configurations = set(writer.completed_configuration_ids(selected_scene))
            accepted_catalogs: list[tuple[Any, ...]] = []
            accepted_hashes: set[str] = set()
            for configuration_id in completed_configurations:
                meta_path = (
                    root
                    / "configurations"
                    / selected_scene
                    / configuration_id
                    / "config_meta.json"
                )
                try:
                    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                    accepted_hashes.add(str(metadata["exact_state_hash"]))
                    accepted_catalogs.append(
                        _read_configuration_catalog(meta_path.with_name("object_catalog.parquet"))
                    )
                except (OSError, KeyError, json.JSONDecodeError):
                    pass
            expected_configuration_ids = {
                f"config_{index:03d}" for index in range(requested_configurations)
            }
            accepted_configurations += len(
                completed_configurations & expected_configuration_ids
            )
            for configuration_index in range(requested_configurations):
                configuration_id = f"config_{configuration_index:03d}"
                configuration_root = (
                    root / "configurations" / selected_scene / configuration_id
                )
                if configuration_id not in completed_configurations:
                    accepted = None
                    for attempt in range(int(config["generation"]["maximum_configuration_attempts"])):
                        seed = (
                            int(config["seed"])
                            + scene_position * 10_000_000
                            + configuration_index * 100_000
                            + attempt
                        )
                        _write_status(
                            root,
                            status="running",
                            stage="sample_configuration",
                            scene_id=selected_scene,
                            configuration_id=configuration_id,
                            attempt=attempt,
                            accepted_configurations=accepted_configurations,
                            accepted_episodes=accepted_episodes,
                        )
                        adapter.load_snapshot(base_snapshot)
                        try:
                            candidate = adapter.randomize_relation_preserving_configuration(seed)
                            if candidate["exact_state_hash"] in accepted_hashes:
                                raise SampleRejected("exact_duplicate_configuration")
                            if any(
                                near_duplicate_configuration(
                                    candidate["catalog"],
                                    catalog,
                                    translation_threshold_m=float(
                                        config["generation"]["near_duplicate_translation_m"]
                                    ),
                                    rotation_threshold_deg=float(
                                        config["generation"]["near_duplicate_rotation_deg"]
                                    ),
                                )
                                for catalog in accepted_catalogs
                            ):
                                raise SampleRejected("near_duplicate_configuration")
                            environment_arrays, _ = _render_environment_floors(adapter, config)
                            world_state = WorldState(
                                scene_id=selected_scene,
                                configuration_id=configuration_id,
                                physical_time_index=None,
                                objects=candidate["catalog"],
                                simulator_snapshot_ref="simulator_state.npy",
                            )
                            configuration = DynamicConfiguration(
                                configuration_id=configuration_id,
                                scene_id=selected_scene,
                                seed=seed,
                                exact_state_hash=candidate["exact_state_hash"],
                                world_state=world_state,
                                environment_bev_ref="bev/environment_base.npz",
                                simulator_snapshot_ref="simulator_state.npy",
                                accepted_attempt=attempt,
                            )
                            writer.write_configuration(
                                configuration,
                                candidate["catalog"],
                                snapshot=candidate["snapshot"],
                                environment_bev=environment_arrays,
                            )
                            accepted = candidate
                            accepted_catalogs.append(candidate["catalog"])
                            accepted_hashes.add(candidate["exact_state_hash"])
                            accepted_configurations += 1
                            break
                        except SampleRejected as error:
                            writer.record_reject(
                                f"configuration:{selected_scene}/{configuration_id}",
                                error.reason,
                                {"attempt": attempt, **error.details},
                            )
                    if accepted is None:
                        raise SampleRejected(
                            "configuration_attempts_exhausted",
                            {"scene_id": selected_scene, "configuration_id": configuration_id},
                        )
                snapshot_path = configuration_root / "simulator_state.npy"
                configuration_snapshot = np.load(snapshot_path, allow_pickle=False)
                existing_episodes = set(
                    writer.completed_episode_ids(selected_scene, configuration_id)
                )
                accepted_episodes += len(existing_episodes)
                used_targets = set(
                    _existing_event_targets(root, selected_scene, configuration_id)
                )
                for episode_index in range(requested_episodes):
                    episode_id = f"episode_{episode_index:03d}"
                    if episode_id in existing_episodes:
                        continue
                    episode_seed = (
                        int(config["seed"])
                        + scene_position * 10_000_000
                        + configuration_index * 100_000
                        + episode_index * 1_000
                    )
                    before = None
                    graph = None
                    heights = None
                    trajectories = None
                    w0_snapshot = None
                    w0_catalog = None
                    trajectory_metrics = None
                    temporal_overlap_metrics = None
                    for placement_attempt in range(int(config["placement"]["maximum_attempts"])):
                        _write_status(
                            root,
                            status="running",
                            stage="sample_episode_before",
                            scene_id=selected_scene,
                            configuration_id=configuration_id,
                            episode_id=episode_id,
                            attempt=placement_attempt,
                            accepted_configurations=accepted_configurations,
                            accepted_episodes=accepted_episodes,
                        )
                        adapter.load_snapshot(configuration_snapshot)
                        try:
                            heights = adapter.place_development_robots(
                                episode_seed + placement_attempt
                            )
                            graph, _ = _initial_overlap(adapter, config)
                            trajectories, trajectory_metrics = adapter.sample_robot_trajectories(
                                episode_seed + placement_attempt + 101
                            )
                            w0_snapshot = adapter.dump_snapshot()
                            temporal_overlap_metrics = _temporal_overlap_preflight(
                                adapter, config, trajectories
                            )
                            trajectory_metrics["temporal_overlap"] = temporal_overlap_metrics
                            w0_catalog = adapter.object_catalog_with_relations()
                            world_calibration = adapter.calibrated_floor_bounds(
                                int(trajectory_metrics["floor_index"]),
                                float(config["bev"]["world_meters_per_pixel"]),
                                float(config["bev"]["bounds_margin_m"]),
                            )
                            before = adapter.playback_trajectories(
                                trajectories,
                                int(trajectory_metrics["floor_index"]),
                                world_calibration,
                            )
                            break
                        except SampleRejected as error:
                            writer.record_reject(
                                f"episode-before:{selected_scene}/{configuration_id}/{episode_id}",
                                error.reason,
                                {"attempt": placement_attempt, **error.details},
                            )
                    if before is None:
                        raise SampleRejected(
                            "episode_before_attempts_exhausted",
                            {"scene_id": selected_scene, "configuration_id": configuration_id},
                        )
                    after = None
                    intervention = None
                    environment_after = None
                    after_catalog = None
                    after_snapshot = None
                    qa_results = None
                    for event_attempt in range(int(config["intervention"]["maximum_attempts"])):
                        _write_status(
                            root,
                            status="running",
                            stage="sample_episode_intervention",
                            scene_id=selected_scene,
                            configuration_id=configuration_id,
                            episode_id=episode_id,
                            attempt=event_attempt,
                            accepted_configurations=accepted_configurations,
                            accepted_episodes=accepted_episodes,
                        )
                        adapter.load_snapshot(w0_snapshot)
                        try:
                            intervention = adapter.apply_atomic_intervention(
                                episode_seed + 500 + event_attempt,
                                excluded_target_ids=tuple(sorted(used_targets)),
                            )
                            environment_after, _ = _render_environment_floors(adapter, config)
                            after = adapter.playback_trajectories(
                                trajectories,
                                int(trajectory_metrics["floor_index"]),
                                world_calibration,
                            )
                            paired = check_paired_trajectories(
                                before["actual_trajectories"],
                                after["actual_trajectories"],
                                position_atol_m=float(
                                    config["trajectory"]["validation_position_tolerance_m"]
                                ),
                                matrix_atol=float(
                                    config["trajectory"]["validation_rotation_tolerance_rad"]
                                ),
                            )
                            bev_pair = check_bev_pair(
                                world_calibration,
                                world_calibration,
                                True,
                                True,
                            )
                            qa_results = (
                                QAResult(
                                    "initial_overlap",
                                    bool(graph.connected and not graph.near_duplicate_pairs),
                                    metrics={
                                        "minimum_overlap": float(min(graph.overlaps.values())),
                                        "maximum_overlap": float(max(graph.overlaps.values())),
                                    },
                                ),
                                QAResult(
                                    "temporal_overlap_connectivity",
                                    True,
                                    metrics={
                                        "connected_fraction": float(
                                            temporal_overlap_metrics["connected_fraction"]
                                        ),
                                        "maximum_isolated_run": max(
                                            temporal_overlap_metrics["maximum_consecutive_isolated_keyframes"].values()
                                        ),
                                    },
                                ),
                                QAResult(
                                    "before_rollout",
                                    all(before["checks"].values()),
                                    metrics=before["metrics"],
                                ),
                                QAResult(
                                    "after_rollout",
                                    all(after["checks"].values()),
                                    metrics=after["metrics"],
                                ),
                                paired,
                                bev_pair,
                                QAResult(
                                    "atomic_intervention",
                                    all(intervention["checks"].values()),
                                    metrics={
                                        "changed_object_count": len(
                                            intervention["changed_instance_ids"]
                                        )
                                    },
                                ),
                            )
                            require_all(qa_results)
                            after_catalog = intervention["catalog"]
                            after_snapshot = intervention["snapshot"]
                            break
                        except SampleRejected as error:
                            writer.record_reject(
                                f"event:{selected_scene}/{configuration_id}/{episode_id}",
                                error.reason,
                                {"attempt": event_attempt, **error.details},
                            )
                    if after is None or intervention is None:
                        raise SampleRejected(
                            "intervention_attempts_exhausted",
                            {
                                "scene_id": selected_scene,
                                "configuration_id": configuration_id,
                                "episode_id": episode_id,
                            },
                        )
                    robot_states = _robot_states(config, heights, trajectories)
                    state_before = WorldState(
                        scene_id=selected_scene,
                        configuration_id=configuration_id,
                        physical_time_index=None,
                        objects=w0_catalog,
                        robots=robot_states,
                        simulator_snapshot_ref="simulator_before.npy",
                    )
                    state_after = WorldState(
                        scene_id=selected_scene,
                        configuration_id=configuration_id,
                        physical_time_index=None,
                        objects=after_catalog,
                        robots=robot_states,
                        simulator_snapshot_ref="simulator_after.npy",
                    )
                    with writer.begin_episode(
                        selected_scene, configuration_id, episode_id
                    ) as transaction:
                        before_world_ref = transaction.write_dense_group(
                            "bev/world_before", before["world_bev"]
                        )
                        after_world_ref = transaction.write_dense_group(
                            "bev/world_after", after["world_bev"]
                        )
                        environment_after_ref = transaction.write_dense_group(
                            "bev/environment_after", environment_after
                        )
                        before_view_refs = {
                            robot_id: transaction.write_dense_group(
                                f"robot_views/before/{robot_id}", modalities
                            )
                            for robot_id, modalities in before["robot_views"].items()
                        }
                        after_view_refs = {
                            robot_id: transaction.write_dense_group(
                                f"robot_views/after/{robot_id}", modalities
                            )
                            for robot_id, modalities in after["robot_views"].items()
                        }
                        np.save(
                            transaction.staging / "simulator_before.npy",
                            np.asarray(w0_snapshot),
                            allow_pickle=False,
                        )
                        np.save(
                            transaction.staging / "simulator_after.npy",
                            np.asarray(after_snapshot),
                            allow_pickle=False,
                        )
                        episode = WorldEpisode(
                            episode_id=episode_id,
                            scene_id=selected_scene,
                            configuration_id=configuration_id,
                            seed=episode_seed,
                            simulator_versions=adapter.runtime_report().get("versions", {}),
                            generator_git_commit=commit,
                            trajectories=trajectories,
                            intervention=intervention["event"],
                            state_before=state_before,
                            state_after=state_after,
                            environment_after_bev_ref=environment_after_ref,
                            world_before_bev_ref=before_world_ref,
                            world_after_bev_ref=after_world_ref,
                            observations_before=_observation_records(
                                config,
                                before["actual_trajectories"],
                                "before",
                                before_view_refs,
                            ),
                            observations_after=_observation_records(
                                config,
                                after["actual_trajectories"],
                                "after",
                                after_view_refs,
                            ),
                            qa=qa_results,
                        )
                        writer.write_episode_metadata(transaction, episode)
                        transaction.write_json(
                            "generation_metrics.json",
                            {
                                "overlap": graph,
                                "trajectory": trajectory_metrics,
                                "before": before["metrics"],
                                "after": after["metrics"],
                                "intervention_attempt": intervention["attempt"],
                            },
                        )
                        inspection_root = transaction.staging / "inspection"
                        image_names = []
                        save_rgb(
                            inspection_root / "world_before_t000.png",
                            before["world_bev"]["rgb"][0],
                        )
                        save_rgb(
                            inspection_root / "world_after_t000.png",
                            after["world_bev"]["rgb"][0],
                        )
                        image_names.extend(
                            ["world_before_t000.png", "world_after_t000.png"]
                        )
                        trajectory_image_name = "trajectory_inspection.png"
                        save_trajectory_inspection(
                            inspection_root / trajectory_image_name,
                            adapter.trajectory_traversability_inspection(
                                int(trajectory_metrics["floor_index"])
                            ),
                            trajectories,
                            temporal_overlap_metrics,
                        )
                        image_names.append(trajectory_image_name)
                        for robot_id in sorted(before["robot_views"]):
                            name = f"{robot_id}_before_t000.png"
                            save_rgb(
                                inspection_root / name,
                                before["robot_views"][robot_id]["rgb"][0],
                            )
                            image_names.append(name)
                        write_html_summary(
                            inspection_root,
                            f"{selected_scene}/{configuration_id}/{episode_id}",
                            {
                                "qa": qa_results,
                                "event": intervention["event"],
                                "trajectory": trajectory_metrics,
                            },
                            image_names,
                        )
                        transaction.finalize()
                    used_targets.add(intervention["event"].target_instance_id)
                    accepted_episodes += 1
                    _write_status(
                        root,
                        status="running",
                        stage="episode_finalized",
                        scene_id=selected_scene,
                        configuration_id=configuration_id,
                        episode_id=episode_id,
                        accepted_configurations=accepted_configurations,
                        accepted_episodes=accepted_episodes,
                    )
        result = {
            "status": "pass",
            "profile": profile,
            "scenes": scenes,
            "accepted_configurations": accepted_configurations,
            "accepted_episodes": accepted_episodes,
            "scope": "development profiles only; full generation not started",
        }
        dump_json(root / "generation_result.json", result)
        (root / "generation_failure.json").unlink(missing_ok=True)
        _write_status(root, **result, stage="complete")
        return root, result
    except BaseException as error:
        failure = {
            "status": "error",
            "profile": profile,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "accepted_configurations": accepted_configurations,
            "accepted_episodes": accepted_episodes,
        }
        dump_json(root / "generation_failure.json", failure)
        _write_status(root, **failure, stage="failed")
        raise
    finally:
        adapter.close()

