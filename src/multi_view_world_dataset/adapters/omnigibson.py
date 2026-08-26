from __future__ import annotations

import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from multi_view_world_dataset.adapters.base import BEVRender, BaseSimulatorAdapter
from multi_view_world_dataset.cameras.transforms import rotation_angle, validate_transform
from multi_view_world_dataset.errors import GeometryError, SampleRejected, SimulatorUnavailableError
from multi_view_world_dataset.rendering.bev import BEVCalibration
from multi_view_world_dataset.sampling.splits import infer_scene_family
from multi_view_world_dataset.schema.records import BaseSceneRecord, ObjectState
from multi_view_world_dataset.utils.runtime import RuntimePaths, installed_versions


class OmniGibsonAdapter(BaseSimulatorAdapter):
    """OmniGibson 3.9 adapter. Imports Kit only when :meth:`start` is called."""

    def __init__(self, runtime: RuntimePaths, dataset_config: dict[str, Any]):
        self.runtime = runtime
        self.config = dataset_config
        self._og: Any = None
        self._lazy: Any = None
        self._th: Any = None
        self._transform_utils: Any = None
        self._asset_utils: Any = None
        self._vision_sensor_type: Any = None
        self._on_top_type: Any = None
        self._inside_type: Any = None
        self._rigid_contact_api: Any = None
        self._development_camera_mounts: dict[str, np.ndarray] = {}
        self._env: Any = None
        self._scene_id: str | None = None
        self._started = False
        self._runtime_findings: dict[str, Any] = {}

    def start(self) -> None:
        if self._started:
            return
        try:
            import torch as th
            import omnigibson as og
            import omnigibson.lazy as lazy
            import omnigibson.utils.transform_utils as transform_utils
            from omnigibson.object_states import Inside, OnTop
            from omnigibson.sensors.vision_sensor import VisionSensor
            from omnigibson.utils import asset_utils
            from omnigibson.utils.usd_utils import RigidContactAPI
        except Exception as error:
            raise SimulatorUnavailableError(f"Failed to launch/import OmniGibson: {error}") from error
        self._og, self._lazy, self._th = og, lazy, th
        self._transform_utils, self._asset_utils = transform_utils, asset_utils
        self._vision_sensor_type = VisionSensor
        self._on_top_type, self._inside_type, self._rigid_contact_api = OnTop, Inside, RigidContactAPI
        if og.sim is None:
            og.launch()
        self._started = True
        self._runtime_findings.update(
            versions=installed_versions(),
            headless=bool(og.gm.HEADLESS),
            device=str(og.sim.device),
            scene_discovery_api="omnigibson.utils.asset_utils.get_available_behavior_1k_scenes",
            snapshot_api="og.sim.dump_state/load_state(serialized=True)",
            sensor_class="omnigibson.sensors.vision_sensor.VisionSensor",
        )

    def close(self) -> None:
        if not self._started:
            return
        self._env = None
        # OmniGibson cleanup can race an asynchronous USD temp writer and raise
        # "Directory not empty", which otherwise prevents SimulationApp.close().
        try:
            self._og.cleanup()
        except OSError as error:
            self._runtime_findings["cleanup_warning"] = str(error)
        if self._og.sim is not None:
            self._og.sim._disable_usd_guard()
        self._og.app.close()
        self._started = False

    def _require_started(self) -> None:
        if not self._started:
            raise SimulatorUnavailableError("Adapter has not been started")

    def _require_scene(self) -> Any:
        self._require_started()
        if self._env is None:
            raise SimulatorUnavailableError("No scene has been loaded")
        return self._env.scene

    def discover_scenes(self) -> list[str]:
        self._require_started()
        scenes = list(self._asset_utils.get_available_behavior_1k_scenes())
        self._runtime_findings["available_scene_count"] = len(scenes)
        return scenes

    def load_scene(self, scene_id: str, *, robot_count: int = 0, development_robot: str = "turtlebot") -> None:
        self._require_started()
        if self._env is not None:
            self._og.clear()
        self._development_camera_mounts.clear()
        if scene_id not in self.discover_scenes():
            raise SimulatorUnavailableError(f"Scene is not installed: {scene_id}")
        camera = self.config["camera"]
        aperture = 20.995
        focal = aperture / (2.0 * np.tan(np.deg2rad(camera["hfov_deg"]) / 2.0))
        robots = []
        for index in range(robot_count):
            robots.append(
                {
                    "model": development_robot.lower(),
                    "name": f"robot_{index:02d}",
                    "obs_modalities": ["rgb", "depth_linear", "normal", "seg_semantic", "seg_instance"],
                    "sensor_config": {
                        "VisionSensor": {
                            "sensor_kwargs": {
                                "image_width": camera["rgb_width"],
                                "image_height": camera["rgb_height"],
                                "focal_length": float(focal),
                                "horizontal_aperture": aperture,
                                "clipping_range": [camera["near_m"], camera["far_m"]],
                            }
                        }
                    },
                }
            )
        environment_config = {
            "scene": {
                "type": "InteractiveTraversableScene",
                "scene_model": scene_id,
                "trav_map_with_objects": True,
            },
            "robots": robots,
        }
        self._env = self._og.Environment(configs=environment_config)
        self._scene_id = scene_id
        self._og.sim.step()
        self._runtime_findings["loaded_scene"] = scene_id
        self._runtime_findings["device"] = str(self._og.sim.device)
        self._runtime_findings["loaded_robot_count"] = len(self._env.robots)

    def _floor_heights(self) -> tuple[float, ...]:
        scene = self._require_scene()
        values = getattr(scene, "floor_heights", [scene.get_floor_height(0)])
        return tuple(float(value) for value in values)

    def scene_record(self, split: str) -> BaseSceneRecord:
        if self._scene_id is None:
            self._require_scene()
        heights = self._floor_heights()
        return BaseSceneRecord(
            scene_id=str(self._scene_id),
            scene_family=infer_scene_family(str(self._scene_id)),
            simulator_scene_model=str(self._scene_id),
            floor_ids=tuple(f"floor_{index:02d}" for index in range(len(heights))),
            floor_heights_m=heights,
            split=split,
        )

    def _pose_matrix(self, obj: Any) -> np.ndarray:
        position, orientation = obj.get_position_orientation()
        orientation_values = self._native_value(orientation).astype(np.float64)
        quaternion_norm = float(np.linalg.norm(orientation_values))
        norm_drift = abs(quaternion_norm - 1.0)
        if not np.isfinite(quaternion_norm) or quaternion_norm < 1.0e-8 or norm_drift > 1.0e-3:
            raise GeometryError(f"Pose quaternion norm drift is unsafe: {quaternion_norm}")
        if hasattr(orientation, "detach"):
            orientation = orientation / quaternion_norm
        else:
            orientation = orientation_values / quaternion_norm
        max_drift = float(self._runtime_findings.get("maximum_pose_quaternion_norm_drift", 0.0))
        self._runtime_findings["maximum_pose_quaternion_norm_drift"] = max(
            max_drift, norm_drift
        )
        matrix = self._transform_utils.pose2mat((position, orientation))
        native_matrix = self._native_value(matrix).astype(np.float64).copy()
        raw_rotation = native_matrix[:3, :3]
        orthonormality_error = float(np.linalg.norm(raw_rotation.T @ raw_rotation - np.eye(3), ord="fro"))
        if orthonormality_error > 1.0e-3:
            raise GeometryError(f"Pose rotation projection would be unsafe: {orthonormality_error}")
        left, _, right = np.linalg.svd(raw_rotation)
        rotation = left @ right
        if np.linalg.det(rotation) < 0:
            left[:, -1] *= -1
            rotation = left @ right
        native_matrix[:3, :3] = rotation
        max_error = float(self._runtime_findings.get("maximum_pose_rotation_projection_error", 0.0))
        self._runtime_findings["maximum_pose_rotation_projection_error"] = max(
            max_error, orthonormality_error
        )
        return validate_transform(native_matrix)

    @staticmethod
    def _native_value(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def object_catalog(self) -> tuple[ObjectState, ...]:
        scene = self._require_scene()
        floor_heights = np.asarray(self._floor_heights())
        structural_categories = {"floors", "walls", "ceilings", "roof", "stairs"}
        native_objects = sorted(scene.objects, key=lambda item: (str(getattr(item, "category", "")), item.name))
        category_ordinals: dict[tuple[str, str], int] = {}
        catalog: list[ObjectState] = []
        for obj in native_objects:
            category = str(getattr(obj, "category", "unknown"))
            asset_uid = str(getattr(obj, "model", getattr(obj, "usd_path", "unknown")))
            ordinal_key = (category, asset_uid)
            ordinal = category_ordinals.get(ordinal_key, 0)
            category_ordinals[ordinal_key] = ordinal + 1
            instance_id = "obj_" + uuid.uuid5(
                uuid.NAMESPACE_URL, f"multi-view-world-dataset:{self._scene_id}:{category}:{asset_uid}:{ordinal}"
            ).hex
            transform = self._pose_matrix(obj)
            try:
                bbox_min, bbox_max = (self._native_value(value) for value in obj.aabb)
            except Exception:
                bbox_min = bbox_max = transform[:3, 3]
            joints = list(getattr(obj, "joints", {}).values())
            joint_names, joint_limits, joint_values = [], [], []
            for joint in joints:
                joint_names.append(str(joint.name))
                try:
                    lower = float(self._native_value(joint.lower_limit).reshape(-1)[0])
                    upper = float(self._native_value(joint.upper_limit).reshape(-1)[0])
                    value = float(self._native_value(joint.get_state()[0]).reshape(-1)[0])
                except Exception:
                    lower, upper, value = float("-inf"), float("inf"), 0.0
                joint_limits.append((lower, upper))
                joint_values.append(value)
            available_states = tuple(sorted(state_type.__name__ for state_type in getattr(obj, "states", {})))
            rooms = getattr(obj, "in_rooms", None) or ()
            floor_index = int(np.argmin(np.abs(floor_heights - transform[2, 3]))) if len(floor_heights) else 0
            fixed_base = bool(getattr(obj, "fixed_base", False))
            scale = self._native_value(getattr(obj, "scale", [1, 1, 1])).astype(float).reshape(-1)[:3]
            catalog.append(
                ObjectState(
                    instance_id=instance_id,
                    asset_uid=asset_uid,
                    category=category,
                    native_path=str(obj.prim_path),
                    structural=category in structural_categories,
                    movable=not fixed_base and category not in structural_categories,
                    articulated=bool(joints),
                    available_states=available_states,
                    object_to_world=transform,
                    bbox_min_world=tuple(float(x) for x in bbox_min),
                    bbox_max_world=tuple(float(x) for x in bbox_max),
                    scale=tuple(float(x) for x in scale),
                    joint_names=tuple(joint_names),
                    joint_limits=tuple(joint_limits),
                    joint_values=tuple(joint_values),
                    room_id=str(rooms[0]) if rooms else None,
                    floor_id=f"floor_{floor_index:02d}",
                )
            )
        return tuple(catalog)

    def _native_objects_by_path(self) -> dict[str, Any]:
        return {str(obj.prim_path): obj for obj in self._require_scene().objects}

    @staticmethod
    def _attach_relations(
        catalog: tuple[ObjectState, ...], relations: tuple[dict[str, Any], ...]
    ) -> tuple[ObjectState, ...]:
        by_target: dict[str, list[dict[str, Any]]] = {}
        for relation in relations:
            by_target.setdefault(str(relation["target_instance_id"]), []).append(
                {
                    "predicate": str(relation["predicate"]),
                    "reference_instance_id": str(relation["reference_instance_id"]),
                    "reference_category": str(relation["reference_category"]),
                }
            )
        return tuple(
            replace(
                obj,
                relations=tuple(
                    sorted(
                        by_target.get(obj.instance_id, ()),
                        key=lambda item: (item["predicate"], item["reference_instance_id"]),
                    )
                ),
            )
            for obj in catalog
        )

    def relation_candidates(
        self, catalog: tuple[ObjectState, ...] | None = None
    ) -> tuple[dict[str, Any], ...]:
        """Return current OnTop / Inside relations suitable for semantic resampling.

        AABB filtering keeps this query local; the final answer always comes from
        OmniGibson's actual object-state predicates rather than geometry heuristics.
        """
        catalog = self.object_catalog() if catalog is None else catalog
        by_path = self._native_objects_by_path()
        records: list[dict[str, Any]] = []
        for target in catalog:
            native_target = by_path.get(target.native_path)
            if not target.movable or target.structural or native_target is None:
                continue
            target_low = np.asarray(target.bbox_min_world)
            target_high = np.asarray(target.bbox_max_world)
            target_center = 0.5 * (target_low + target_high)
            for reference in catalog:
                native_reference = by_path.get(reference.native_path)
                if reference.instance_id == target.instance_id or native_reference is None:
                    continue
                reference_low = np.asarray(reference.bbox_min_world)
                reference_high = np.asarray(reference.bbox_max_world)
                state_specs: list[tuple[str, Any]] = []
                if self._inside_type in native_target.states:
                    contained = np.all(target_center >= reference_low - 0.05) and np.all(
                        target_center <= reference_high + 0.05
                    )
                    if contained:
                        state_specs.append(("Inside", self._inside_type))
                if self._on_top_type in native_target.states:
                    overlap_xy = np.minimum(target_high[:2], reference_high[:2]) - np.maximum(
                        target_low[:2], reference_low[:2]
                    )
                    vertically_close = abs(float(target_low[2] - reference_high[2])) <= 0.25
                    if np.all(overlap_xy > 0.0) and vertically_close:
                        state_specs.append(("OnTop", self._on_top_type))
                for predicate, state_type in state_specs:
                    try:
                        active = bool(native_target.states[state_type].get_value(native_reference))
                    except Exception:
                        active = False
                    if active:
                        records.append(
                            {
                                "predicate": predicate,
                                "target_instance_id": target.instance_id,
                                "target_category": target.category,
                                "reference_instance_id": reference.instance_id,
                                "reference_category": reference.category,
                                "target_native_path": target.native_path,
                                "reference_native_path": reference.native_path,
                            }
                        )
        return tuple(
            sorted(
                records,
                key=lambda item: (
                    item["target_instance_id"],
                    item["predicate"],
                    item["reference_instance_id"],
                ),
            )
        )

    def object_catalog_with_relations(self) -> tuple[ObjectState, ...]:
        catalog = self.object_catalog()
        return self._attach_relations(catalog, self.relation_candidates(catalog))

    def _free_traversable_candidate_count(self, floor_index: int) -> int:
        scene = self._require_scene()
        floor_map = self._th.clone(scene.trav_map.floor_map[floor_index])
        robot = self._env.robots[0] if self._env.robots else None
        eroded = scene.trav_map._erode_trav_map(floor_map, robot=robot)
        return int(self._th.count_nonzero(eroded == 255).item())

    def randomize_relation_preserving_configuration(self, seed: int) -> dict[str, Any]:
        """Create one native-validated dynamic configuration without random XYZ poses."""
        from multi_view_world_dataset.sampling.configurations import exact_state_hash

        baseline_snapshot = self.dump_snapshot()
        baseline_raw_catalog = self.object_catalog()
        relations = self.relation_candidates(baseline_raw_catalog)
        baseline_catalog = self._attach_relations(baseline_raw_catalog, relations)
        if not relations:
            raise SampleRejected("no_relation_preserving_configuration_candidate")
        rng = np.random.default_rng(seed)
        failures: list[dict[str, Any]] = []
        generation = self.config["generation"]
        translation_threshold = float(generation["near_duplicate_translation_m"])
        rotation_threshold = float(np.deg2rad(generation["near_duplicate_rotation_deg"]))
        baseline_by_id = {obj.instance_id: obj for obj in baseline_catalog}
        random_tiebreakers = rng.random(len(relations))
        relation_order = sorted(
            range(len(relations)),
            key=lambda index: (
                relations[index]["reference_category"] != "floors",
                np.prod(
                    np.asarray(baseline_by_id[relations[index]["target_instance_id"]].bbox_max_world)
                    - np.asarray(baseline_by_id[relations[index]["target_instance_id"]].bbox_min_world)
                ),
                random_tiebreakers[index],
            ),
        )
        for attempt_index, relation_index in enumerate(relation_order, start=1):
            relation = relations[relation_index]
            self.load_snapshot(baseline_snapshot)
            native_by_path = self._native_objects_by_path()
            target_native = native_by_path[relation["target_native_path"]]
            reference_native = native_by_path[relation["reference_native_path"]]
            state_type = self._on_top_type if relation["predicate"] == "OnTop" else self._inside_type
            self._th.manual_seed(int(seed + attempt_index))
            try:
                sampled = bool(
                    target_native.states[state_type].set_value(
                        reference_native,
                        True,
                        reset_before_sampling=True,
                        use_trav_map=(
                            relation["predicate"] == "OnTop" and relation["reference_category"] == "floors"
                        ),
                    )
                )
            except Exception as error:
                failures.append({"reason": "relation_sampler_error", "error": str(error), **relation})
                continue
            if not sampled:
                failures.append({"reason": "relation_sampler_failed", **relation})
                continue
            for _ in range(int(generation["settle_steps"])):
                for robot in self._env.robots:
                    robot.keep_still()
                self._og.sim.step_physics()
            self._og.sim.step()
            if not bool(target_native.states[state_type].get_value(reference_native)):
                failures.append({"reason": "relation_lost_after_settle", **relation})
                continue
            raw_catalog = self.object_catalog()
            catalog = self._attach_relations(raw_catalog, (relation,))
            after_by_id = {obj.instance_id: obj for obj in catalog}
            before_target = baseline_by_id[relation["target_instance_id"]]
            after_target = after_by_id[relation["target_instance_id"]]
            translation = float(
                np.linalg.norm(after_target.object_to_world[:3, 3] - before_target.object_to_world[:3, 3])
            )
            rotation = float(rotation_angle(before_target.object_to_world, after_target.object_to_world))
            same_identity = set(after_by_id) == set(baseline_by_id)
            same_floor = after_target.floor_id == before_target.floor_id
            same_room = before_target.room_id is None or after_target.room_id == before_target.room_id
            extra_collision = bool(
                self._rigid_contact_api.is_in_contact(
                    scene_idx=target_native.scene.idx,
                    query_set=[target_native],
                    with_set=None,
                    ignore_set=[reference_native],
                    current_only=True,
                )
            )
            floor_index = int((after_target.floor_id or "floor_00").split("_")[-1])
            free_candidates = self._free_traversable_candidate_count(floor_index)
            intervention_targets = sum(obj.movable and not obj.structural for obj in catalog)
            diverse = translation > translation_threshold or rotation > rotation_threshold
            checks = {
                "stable_instance_ids": same_identity,
                "relation_preserved": True,
                "same_floor": same_floor,
                "same_room_when_known": same_room,
                "no_extra_collision": not extra_collision,
                "configuration_diverse": diverse,
                "robot_free_space": free_candidates >= 3,
                "intervention_candidate_available": intervention_targets > 0,
            }
            if not all(checks.values()):
                failures.append({"reason": "configuration_qa_failed", "checks": checks, **relation})
                continue
            accepted_snapshot = self.dump_snapshot()
            accepted_hash = exact_state_hash(catalog, decimals=int(generation["exact_hash_decimals"]))
            self.load_snapshot(accepted_snapshot)
            restored_catalog = self._attach_relations(self.object_catalog(), (relation,))
            restored_hash = exact_state_hash(restored_catalog, decimals=int(generation["exact_hash_decimals"]))
            if restored_hash != accepted_hash:
                failures.append({"reason": "configuration_snapshot_restore_mismatch", **relation})
                continue
            return {
                "catalog": restored_catalog,
                "snapshot": accepted_snapshot,
                "exact_state_hash": accepted_hash,
                "baseline_exact_state_hash": exact_state_hash(
                    baseline_catalog, decimals=int(generation["exact_hash_decimals"])
                ),
                "accepted_attempt": attempt_index,
                "relation": relation,
                "checks": checks,
                "translation_m": translation,
                "rotation_deg": float(np.rad2deg(rotation)),
                "free_traversable_candidates": free_candidates,
                "intervention_target_count": intervention_targets,
            }
        self.load_snapshot(baseline_snapshot)
        raise SampleRejected(
            "configuration_relation_sampling_failed",
            {"candidate_count": len(relations), "failures": failures[:20]},
        )

    def dump_snapshot(self) -> np.ndarray:
        self._require_scene()
        state = self._og.sim.dump_state(serialized=True)
        return self._native_value(state).copy()

    def load_snapshot(self, snapshot: np.ndarray) -> None:
        self._require_scene()
        state = self._th.as_tensor(snapshot, dtype=self._th.float32, device=self._og.sim.device)
        self._og.sim.load_state(state, serialized=True)

    def calibrated_floor_bounds(self, floor_index: int, meters_per_pixel: float, margin_m: float) -> BEVCalibration:
        catalog = self.object_catalog()
        floor_id = f"floor_{floor_index:02d}"
        on_floor = [obj for obj in catalog if obj.floor_id == floor_id]
        if not on_floor:
            raise GeometryError(f"No catalog objects assigned to {floor_id}")
        xmin = min(obj.bbox_min_world[0] for obj in on_floor)
        ymin = min(obj.bbox_min_world[1] for obj in on_floor)
        xmax = max(obj.bbox_max_world[0] for obj in on_floor)
        ymax = max(obj.bbox_max_world[1] for obj in on_floor)
        return BEVCalibration.from_bounds(
            floor_id, self._floor_heights()[floor_index], meters_per_pixel, (xmin, ymin, xmax, ymax), margin_m
        )

    def place_development_robots(self, seed: int) -> dict[str, float]:
        """Place three robots on one floor and mount their development sensors at frozen v1 camera poses."""
        scene = self._require_scene()
        robots = list(self._env.robots)
        if len(robots) != 3:
            raise SimulatorUnavailableError("Development placement requires exactly three loaded robots")
        rng = np.random.default_rng(seed)
        placement = self.config["placement"]
        heights = self.config["camera"]["heights_m"]
        # Installed OG 3.9.2 calls torch.randint without a size when floor=None.
        floor_index = int(rng.integers(int(scene.n_floors)))
        # Sampling repeatedly with reference_point is uniform over the entire connected
        # component, not over a neighborhood of the reference point. On large components
        # that makes the probability of landing inside a 3 m cluster needlessly tiny.
        # Build the local candidate pool from the installed traversability map instead.
        trav_map = scene.trav_map
        eroded = trav_map._erode_trav_map(
            self._th.clone(trav_map.floor_map[floor_index]), robot=robots[0]
        )
        pixels = self._th.stack(self._th.where(eroded == 255), dim=1)
        if pixels.shape[0] < 3:
            raise SimulatorUnavailableError(f"Floor {floor_index} has fewer than three traversable pixels")
        world_xy = self._native_value(trav_map.map_to_world(pixels)).astype(float)
        z = float(scene.get_floor_height(floor_index))
        candidates = np.column_stack((world_xy, np.full(len(world_xy), z)))
        cluster_radius = float(placement["cluster_radius_m"])
        effective_radius = min(cluster_radius, float(placement.get("preferred_cluster_radius_m", cluster_radius)))
        minimum_distance = float(placement["minimum_pairwise_distance_m"])
        maximum_attempts = int(placement["maximum_attempts"])
        selected: list[np.ndarray] = []
        attempts = 0
        for center_index in rng.permutation(len(candidates))[:maximum_attempts]:
            attempts += 1
            center = candidates[int(center_index)]
            local = candidates[np.linalg.norm(candidates[:, :2] - center[:2], axis=1) <= effective_radius]
            if len(local) < 3:
                continue
            selection = [local[int(rng.integers(len(local)))]]
            # Greedy farthest-point selection is deterministic under the seeded tie order
            # and avoids rejection loops when the connected component is large.
            tie_order = rng.permutation(len(local))
            while len(selection) < 3:
                distances = np.stack(
                    [np.linalg.norm(local[:, :2] - point[:2], axis=1) for point in selection], axis=1
                ).min(axis=1)
                distances[distances < minimum_distance] = -1.0
                best_distance = float(distances.max())
                if best_distance < minimum_distance:
                    break
                best = tie_order[np.argmax(distances[tie_order])]
                selection.append(local[int(best)])
            if len(selection) == 3:
                selected = selection
                break
        if len(selected) != 3:
            raise SimulatorUnavailableError(
                f"Failed clustered placement after {attempts} candidate centers on floor {floor_index}"
            )
        sampled_heights: dict[str, float] = {}
        pitch = np.deg2rad(float(self.config["camera"]["pitch_deg"]))
        cosine, sine = np.cos(pitch), np.sin(pitch)
        # Columns are OpenCV camera right, down, forward expressed in robot base coordinates.
        camera_rotation_base = np.array([[0.0, sine, cosine], [-1.0, 0.0, 0.0], [0.0, -cosine, sine]])
        cv_to_usd = np.diag([1.0, -1.0, -1.0, 1.0])
        shared_yaw = float(rng.uniform(-np.pi, np.pi))
        heading_jitter = np.deg2rad(float(placement.get("heading_jitter_deg", 5.0)))
        heading_offsets = np.linspace(-heading_jitter, heading_jitter, len(robots))
        for robot, point, heading_offset in zip(robots, selected, heading_offsets, strict=True):
            yaw = shared_yaw + float(heading_offset)
            orientation = self._transform_utils.euler2quat(self._th.tensor([0.0, 0.0, yaw]))
            robot.set_position_orientation(
                position=self._th.as_tensor(point, dtype=self._th.float32), orientation=orientation
            )
            height = float(rng.choice(heights))
            sampled_heights[robot.name] = height
            base_to_world = self._pose_matrix(robot)
            camera_to_base = np.eye(4)
            camera_to_base[:3, :3] = camera_rotation_base
            camera_to_base[:3, 3] = [0.0, 0.0, height]
            cv_camera_to_world = base_to_world @ camera_to_base
            usd_camera_to_world = cv_camera_to_world @ cv_to_usd
            camera_position, camera_orientation = self._transform_utils.mat2pose(
                self._th.as_tensor(usd_camera_to_world, dtype=self._th.float32)
            )
            vision_sensors = [
                sensor for sensor in robot.sensors.values() if isinstance(sensor, self._vision_sensor_type)
            ]
            if len(vision_sensors) != 1:
                raise SimulatorUnavailableError(
                    f"Expected one VisionSensor on {robot.name}, found {len(vision_sensors)}"
                )
            sensor = vision_sensors[0]
            sensor.set_position_orientation(position=camera_position, orientation=camera_orientation)
            self._development_camera_mounts[robot.name] = camera_to_base.copy()
        settle_steps = min(30, int(self.config["generation"]["settle_steps"]))
        for _ in range(settle_steps):
            for robot in robots:
                robot.keep_still()
            self._og.sim.step()
        for _ in range(4):
            self._og.sim.render()
        self._runtime_findings["development_camera_heights_m"] = sampled_heights
        self._runtime_findings["development_camera_settle_steps"] = settle_steps
        self._runtime_findings["placement_attempts"] = attempts
        self._runtime_findings["placement_candidate_count"] = len(candidates)
        self._runtime_findings["placement_shared_yaw_rad"] = shared_yaw
        self._runtime_findings["placement_source"] = "robot-eroded traversability map"
        self._runtime_findings["placement_effective_radius_m"] = effective_radius
        return sampled_heights

    def robot_observations(self) -> dict[str, dict[str, Any]]:
        self._require_scene()
        result: dict[str, dict[str, Any]] = {}
        for robot in self._env.robots:
            sensors = [
                sensor for sensor in robot.sensors.values() if isinstance(sensor, self._vision_sensor_type)
            ]
            if len(sensors) != 1:
                raise SimulatorUnavailableError(f"Expected one VisionSensor on {robot.name}")
            sensor = sensors[0]
            observation, info = sensor.get_obs()
            usd_camera_to_world = self._pose_matrix(sensor)
            camera_to_world = validate_transform(usd_camera_to_world @ np.diag([1.0, -1.0, -1.0, 1.0]))
            base_to_world = self._pose_matrix(robot)
            camera_to_base = np.linalg.inv(base_to_world) @ camera_to_world
            mount_orthonormality_error = float(
                np.linalg.norm(camera_to_base[:3, :3].T @ camera_to_base[:3, :3] - np.eye(3), ord="fro")
            )
            expected_mount = self._development_camera_mounts.get(robot.name)
            translation_error = float("inf")
            rotation_error = float("inf")
            if expected_mount is not None:
                translation_error = float(np.linalg.norm(camera_to_base[:3, 3] - expected_mount[:3, 3]))
                relative_rotation = expected_mount[:3, :3].T @ camera_to_base[:3, :3]
                rotation_error = float(np.arccos(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)))
            sensor_prim_path = str(sensor.prim_path)
            robot_prim_path = str(robot.prim_path)
            result[robot.name] = {
                "modalities": {name: self._native_value(value) for name, value in observation.items()},
                "info": info,
                "camera_to_world": camera_to_world,
                "base_to_world": base_to_world,
                "camera_to_base": camera_to_base,
                "expected_camera_to_base": expected_mount,
                "mount_translation_error_m": translation_error,
                "mount_rotation_error_rad": rotation_error,
                "mount_orthonormality_error": mount_orthonormality_error,
                "sensor_prim_path": sensor_prim_path,
                "sensor_attached_to_robot": sensor_prim_path.startswith(robot_prim_path + "/"),
            }
        return result

    def _set_visible(self, objects: list[Any], visible: bool) -> list[tuple[Any, bool]]:
        prior: list[tuple[Any, bool]] = []
        for obj in objects:
            try:
                prior.append((obj, bool(obj.visible)))
                obj.visible = visible
            except Exception:
                continue
        return prior

    def render_floor_bev(
        self, floor_index: int, calibration: BEVCalibration, *, include_robots: bool, modalities: tuple[str, ...]
    ) -> BEVRender:
        scene = self._require_scene()
        if calibration.floor_id != f"floor_{floor_index:02d}":
            raise GeometryError("BEV floor index and calibration floor_id disagree")
        sensor_modalities = set(modalities) & {
            "rgb", "depth_linear", "normal", "semantic", "instance", "instance_id"
        }
        backend_names = {
            "semantic": "seg_semantic",
            "instance": "seg_instance",
            "instance_id": "seg_instance_id",
        }
        sensor_names = [backend_names.get(name, name) for name in sensor_modalities]
        if "height" in modalities:
            sensor_names.append("depth_linear")
        sensor_names = sorted(set(sensor_names))
        maximum_dimension = 16384
        if calibration.width > maximum_dimension or calibration.height > maximum_dimension:
            raise GeometryError(
                f"BEV {calibration.width}x{calibration.height} exceeds untiled renderer limit {maximum_dimension}"
            )
        camera = self._vision_sensor_type(
            relative_prim_path=f"/mvwd_bev_camera_{floor_index}",
            name=f"mvwd_bev_camera_{floor_index}",
            modalities=sensor_names,
            image_width=calibration.width,
            image_height=calibration.height,
            clipping_range=(0.01, 100.0),
        )
        hidden: list[tuple[Any, bool]] = []
        try:
            camera.load(None)
            with self._og.sim.editing_usd():
                usd_camera = self._lazy.pxr.UsdGeom.Camera(camera.prim)
                usd_camera.GetProjectionAttr().Set(self._lazy.pxr.UsdGeom.Tokens.orthographic)
                # Isaac Sim expresses USD camera aperture in tenths of a world unit.
                # A world-space span of W metres therefore requires aperture 10 * W.
                world_width = float(calibration.world_bounds[2] - calibration.world_bounds[0])
                world_height = float(calibration.world_bounds[3] - calibration.world_bounds[1])
                usd_camera.GetHorizontalApertureAttr().Set(10.0 * world_width)
                usd_camera.GetVerticalApertureAttr().Set(10.0 * world_height)
            projection = str(self._lazy.pxr.UsdGeom.Camera(camera.prim).GetProjectionAttr().Get())
            if projection != "orthographic":
                raise GeometryError(f"BEV camera projection is {projection!r}, not orthographic")
            catalog = self.object_catalog()
            top_z = max((obj.bbox_max_world[2] for obj in catalog), default=calibration.floor_z + 3.0)
            camera_z = top_z + 2.0
            xmin, ymin, xmax, ymax = calibration.world_bounds
            camera.set_position_orientation(
                position=self._th.tensor([(xmin + xmax) / 2, (ymin + ymax) / 2, camera_z]),
                orientation=self._th.tensor([0.0, 0.0, 0.0, 1.0]),
            )
            if not include_robots:
                hidden.extend(self._set_visible(list(self._env.robots), False))
            ceilings = [obj for obj in scene.objects if str(getattr(obj, "category", "")) in {"ceilings", "roof"}]
            hidden.extend(self._set_visible(ceilings, False))
            camera.initialize()
            for _ in range(4):
                self._og.sim.render()
            observation, info = camera.get_obs()
            arrays: dict[str, np.ndarray] = {}
            for public_name in sensor_modalities:
                backend_name = backend_names.get(public_name, public_name)
                arrays[public_name] = self._native_value(observation[backend_name])
            if "height" in modalities:
                depth = self._native_value(observation["depth_linear"]).squeeze()
                arrays["height"] = np.where(np.isfinite(depth), camera_z - depth - calibration.floor_z, np.nan).astype(np.float32)
            if "occupancy" in modalities:
                height = arrays.get("height")
                if height is None:
                    depth = self._native_value(observation["depth_linear"]).squeeze()
                    height = np.where(np.isfinite(depth), camera_z - depth - calibration.floor_z, np.nan)
                arrays["occupancy"] = (np.isfinite(height) & (height > 0.10)).astype(np.uint8)
            return BEVRender(
                calibration,
                arrays,
                projection,
                include_robots,
                metadata={
                    "segmentation_info": info,
                    "horizontal_aperture": 10.0 * world_width,
                    "vertical_aperture": 10.0 * world_height,
                    "camera_z": camera_z,
                },
            )
        finally:
            for obj, was_visible in reversed(hidden):
                obj.visible = was_visible
            if camera.loaded:
                camera.remove()
            self._og.sim.render()

    def resolve_nova_carter_asset(self, *, verify: bool = True) -> dict[str, Any]:
        self._require_started()
        # OmniGibson 3.9.2's lazy_isaacsim namespace does not expose storage.native.
        # The official Isaac Sim 5.1 extension imports this module directly after Kit has launched.
        self._lazy.isaacsim.core.utils.extensions.enable_extension("isaacsim.storage.native")
        self._og.app.update()
        from isaacsim.storage.native import get_assets_root_path

        root = get_assets_root_path(skip_check=not verify)
        if not root:
            result = {"asset_root": None, "uri": None, "exists": False, "reason": "Isaac asset root unresolved"}
            self._runtime_findings["nova_carter"] = result
            return result
        uri = root.rstrip("/") + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
        exists, status_name = None, "not_checked"
        if verify:
            status, _ = self._lazy.omni.client.stat(uri)
            status_name = str(status)
            exists = status == self._lazy.omni.client.Result.OK
        result = {"asset_root": root, "uri": uri, "exists": exists, "status": status_name}
        self._runtime_findings["nova_carter"] = result
        return result

    def runtime_report(self) -> dict[str, Any]:
        return dict(self._runtime_findings)

