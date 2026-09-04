from __future__ import annotations

import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from multi_view_world_dataset.adapters.base import BEVRender, BaseSimulatorAdapter
from multi_view_world_dataset.assets import materialize_mobile_sensor_robot
from multi_view_world_dataset.cameras.transforms import rotation_angle, validate_transform
from multi_view_world_dataset.errors import GeometryError, SampleRejected, SimulatorUnavailableError
from multi_view_world_dataset.rendering.bev import BEVCalibration
from multi_view_world_dataset.sampling.interventions import (
    choose_intervention_type,
    eligible_intervention_targets,
    propose_articulation,
    propose_rigid_relocation,
    propose_state_change,
)
from multi_view_world_dataset.sampling.splits import infer_scene_family
from multi_view_world_dataset.sampling.trajectories import (
    sample_geodesic_trajectory_set,
    trajectory_kinematic_metrics,
)
from multi_view_world_dataset.schema.records import (
    BaseSceneRecord, InterventionEvent, InterventionType, ObjectState, Trajectory,
)
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
        self._object_state_utils: Any = None
        self._development_camera_mounts: dict[str, np.ndarray] = {}
        self._relation_cache: tuple[dict[str, Any], ...] | None = None
        self._env: Any = None
        self._scene_id: str | None = None
        self._using_final_robot = False
        self._development_bev_sensor: Any = None
        self._final_robot_capture_sensor: Any = None
        self._final_robot_fast_instance_annotator: Any = None
        self._syntheticdata_helpers: Any = None
        self._started = False
        self._runtime_findings: dict[str, Any] = {}

    def _configured_bev_sensor_names(self) -> list[str]:
        """Return all Replicator modalities needed by configured BEV captures."""
        public_modalities = set(self.config["bev"].get("modalities", ())) | set(
            self.config["bev"].get("world_modalities", ())
        )
        backend_names = {
            "semantic": "seg_semantic",
            "instance": "seg_instance_id" if self._using_final_robot else "seg_instance",
            "instance_id": "seg_instance_id",
        }
        sensor_names = {
            backend_names.get(name, name)
            for name in public_modalities
            if name
            in {
                "rgb",
                "depth_linear",
                "normal",
                "semantic",
                "instance",
                "instance_id",
            }
        }
        if {"height", "occupancy"} & public_modalities:
            sensor_names.add("depth_linear")
        if self._using_final_robot:
            # Final-robot segmentation is derived from the raw renderer-ID
            # AOV. Letting VisionSensor own any segmentation modality also
            # installs SemanticSegmentation / InstanceMapping graphs, which
            # are unsafe after articulation motion in Isaac Sim 5.0.
            sensor_names.difference_update(
                {"seg_semantic", "seg_instance", "seg_instance_id"}
            )
        return sorted(sensor_names)

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
            from omnigibson.utils import object_state_utils
            from omnigibson.utils.usd_utils import RigidContactAPI
        except Exception as error:
            raise SimulatorUnavailableError(f"Failed to launch/import OmniGibson: {error}") from error
        self._og, self._lazy, self._th = og, lazy, th
        self._transform_utils, self._asset_utils = transform_utils, asset_utils
        self._vision_sensor_type = VisionSensor
        self._on_top_type, self._inside_type, self._rigid_contact_api = OnTop, Inside, RigidContactAPI
        self._object_state_utils = object_state_utils
        if og.sim is None:
            og.launch()
        from omni.syntheticdata import helpers as syntheticdata_helpers

        self._syntheticdata_helpers = syntheticdata_helpers
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
        self._development_bev_sensor = None
        self._final_robot_capture_sensor = None
        self._development_camera_mounts.clear()
        self._relation_cache = None
        if scene_id not in self.discover_scenes():
            raise SimulatorUnavailableError(f"Scene is not installed: {scene_id}")
        camera = self.config["camera"]
        aperture = 20.995
        focal = aperture / (2.0 * np.tan(np.deg2rad(camera["hfov_deg"]) / 2.0))
        model_name = development_robot.lower()
        final_model = str(self.config["robot"]["final_model"]).lower()
        self._using_final_robot = model_name == final_model
        robot_asset_root: Path | None = None
        robot_module: Any = None
        if self._using_final_robot:
            if self.runtime.cache_root is None:
                raise SimulatorUnavailableError(
                    "Final robot materialization requires --cache-root or DATASET_CACHE_ROOT"
                )
            nova = self.resolve_nova_carter_asset(verify=True)
            if nova.get("exists") is not True or not nova.get("uri"):
                raise SimulatorUnavailableError(
                    f"Installed Nova Carter asset could not be verified: {nova}"
                )
            repository_root = Path(__file__).resolve().parents[3]
            robot_asset_root = (
                self.runtime.cache_root / "multi_view_world_dataset" / "robot_assets"
            )
            asset = materialize_mobile_sensor_robot(
                repository_root / "assets" / "robots" / final_model,
                robot_asset_root,
                str(nova["uri"]),
            )
            stage = self._lazy.pxr.Usd.Stage.Open(str(asset.usd_path))
            if stage is None:
                raise SimulatorUnavailableError(f"Failed to open final robot overlay: {asset.usd_path}")
            instance_paths: list[str] = []
            for _ in range(16):
                instances = [prim for prim in stage.TraverseAll() if prim.IsInstanceable()]
                if not instances:
                    break
                for prim in instances:
                    path = str(prim.GetPath())
                    if path not in instance_paths:
                        instance_paths.append(path)
                    prim.SetInstanceable(False)
                stage.GetRootLayer().Save()
                stage = self._lazy.pxr.Usd.Stage.Open(str(asset.usd_path))
                if stage is None:
                    raise SimulatorUnavailableError(
                        f"Failed to reopen final robot overlay: {asset.usd_path}"
                    )
            else:
                raise SimulatorUnavailableError("Final robot overlay contains recursively nested USD instances")
            mast_path = f"/{model_name}/mast_carriage"
            mast_prim = stage.GetPrimAtPath(mast_path)
            if not mast_prim.IsValid():
                raise SimulatorUnavailableError(f"Final robot mast link is missing: {mast_path}")
            mast_xform = self._lazy.pxr.UsdGeom.Xformable(mast_prim)
            authored_ops: list[str] = []
            if not mast_prim.GetAttribute("xformOp:orient").IsValid():
                mast_xform.AddOrientOp().Set(self._lazy.pxr.Gf.Quatf(1.0))
                authored_ops.append("xformOp:orient")
            if not mast_prim.GetAttribute("xformOp:scale").IsValid():
                mast_xform.AddScaleOp().Set(self._lazy.pxr.Gf.Vec3d(1.0))
                authored_ops.append("xformOp:scale")
            mast_rigid_body_api = self._lazy.pxr.PhysxSchema.PhysxRigidBodyAPI.Apply(mast_prim)
            mast_rigid_body_api.CreateDisableGravityAttr().Set(True)
            controlled_joint_names = {"joint_wheel_left", "joint_wheel_right"}
            disabled_drives: dict[str, list[str]] = {}
            for prim in stage.TraverseAll():
                drive_axes = [
                    axis
                    for axis in ("angular", "linear")
                    if prim.HasAPI(self._lazy.pxr.UsdPhysics.DriveAPI, axis)
                ]
                if drive_axes and prim.GetName() not in controlled_joint_names:
                    for axis in drive_axes:
                        prim.RemoveAPI(self._lazy.pxr.UsdPhysics.DriveAPI, axis)
                    disabled_drives[str(prim.GetPath())] = drive_axes
            stage.GetRootLayer().Save()
            self._runtime_findings["final_robot_disabled_unused_drives"] = disabled_drives
            self._runtime_findings["final_robot_authored_xform_ops"] = {
                mast_path: authored_ops,
            }
            self._runtime_findings["final_robot_deinstanced_prim_paths"] = instance_paths
            del stage

            from omnigibson.robots import REGISTERED_ROBOTS
            import omnigibson.robots.robot as robot_module

            if model_name not in REGISTERED_ROBOTS:
                REGISTERED_ROBOTS.append(model_name)
            self._runtime_findings["final_robot_asset"] = {
                "model": model_name,
                "usd_path": str(asset.usd_path),
                "definition_path": str(asset.definition_path),
                "nova_carter_uri": asset.nova_carter_uri,
            }
        robots = []
        for index in range(robot_count):
            robot_config = {
                "model": model_name,
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
            if self._using_final_robot:
                robot_config["include_sensor_names"] = ["dataset_camera"]
                robot_config["obs_modalities"] = ["rgb", "depth_linear"]
            robots.append(robot_config)
        environment_config = {
            "scene": {
                "type": "InteractiveTraversableScene",
                "scene_model": scene_id,
                "trav_map_with_objects": True,
            },
            "robots": robots,
        }
        original_get_dataset_path = None
        original_xform_get_attribute = None
        xform_prim_module = None
        if self._using_final_robot:
            import omnigibson.prims.xform_prim as xform_prim_module

            original_get_dataset_path = robot_module.get_dataset_path
            original_xform_get_attribute = xform_prim_module.XFormPrim.get_attribute

            def project_robot_dataset_path(dataset_name: str) -> str:
                if dataset_name == "omnigibson-robot-assets":
                    return str(robot_asset_root)
                return original_get_dataset_path(dataset_name)

            def project_xform_get_attribute(prim: Any, attr: str) -> Any:
                value = original_xform_get_attribute(prim, attr)
                if attr == "xformOp:scale" and value is None:
                    # Nova instance-proxy geometry may omit an authored scale. Unit
                    # scale is the USD-defined default and requires no stage edit.
                    count = int(self._runtime_findings.get("missing_scale_fallback_count", 0))
                    self._runtime_findings["missing_scale_fallback_count"] = count + 1
                    return [1.0, 1.0, 1.0]
                return value

            robot_module.get_dataset_path = project_robot_dataset_path
            xform_prim_module.XFormPrim.get_attribute = project_xform_get_attribute
            self._runtime_findings["final_robot_missing_scale_fallback"] = "unit_scale"
        try:
            self._env = self._og.Environment(configs=environment_config)
        finally:
            if original_get_dataset_path is not None:
                robot_module.get_dataset_path = original_get_dataset_path
            if original_xform_get_attribute is not None:
                xform_prim_module.XFormPrim.get_attribute = original_xform_get_attribute
        self._scene_id = scene_id
        self._og.sim.step()
        if not self._using_final_robot:
            # Keep the visible semantic instance set stable for the lifetime of
            # the persistent BEV graph. Toggling ceilings between paired
            # rollouts invalidates SyntheticData's instance-mapping graph.
            ceilings = [
                obj
                for obj in self._require_scene().objects
                if str(getattr(obj, "category", "")) in {"ceilings", "roof"}
            ]
            for ceiling in ceilings:
                ceiling.visible = False
            self._runtime_findings["development_hidden_ceilings"] = sorted(
                ceiling.name for ceiling in ceilings
            )
            self._initialize_development_bev_sensor()
        if self._using_final_robot:
            mast_qa = {}
            for robot in self._env.robots:
                joint_names = [
                    name for name in robot.joints if name.endswith("mvwd_mast_joint")
                ]
                carriage_links = [
                    name for name in robot.links if name.endswith("mast_carriage")
                ]
                sensors = [
                    sensor
                    for sensor in robot.sensors.values()
                    if isinstance(sensor, self._vision_sensor_type)
                ]
                if len(joint_names) != 1 or len(carriage_links) != 1 or len(sensors) != 1:
                    raise SimulatorUnavailableError(
                        f"Final robot {robot.name} mast/sensor topology is invalid: "
                        f"joints={joint_names}, carriage_links={carriage_links}, sensors={len(sensors)}"
                    )
                joint = robot.joints[joint_names[0]]
                lower, upper = float(joint.lower_limit), float(joint.upper_limit)
                if lower > 1.0e-6 or upper < 0.6 - 1.0e-6:
                    raise SimulatorUnavailableError(
                        f"Final robot {robot.name} mast limits are [{lower}, {upper}], expected [0, 0.6]"
                    )
                mast_qa[robot.name] = {
                    "joint_name": joint_names[0],
                    "carriage_link": carriage_links[0],
                    "lower_limit_m": lower,
                    "upper_limit_m": upper,
                    "sensor_prim_path": str(sensors[0].prim_path),
                }
            self._runtime_findings["final_robot_mast_qa"] = mast_qa
            self._initialize_final_robot_capture_sensor()
            self._release_final_robot_bev_modalities(self._final_robot_capture_sensor)
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
        robot_paths = {str(robot.prim_path) for robot in self._env.robots}
        native_objects = sorted(
            (obj for obj in scene.objects if str(obj.prim_path) not in robot_paths),
            key=lambda item: (str(getattr(item, "category", "")), item.name),
        )
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
            native_states = getattr(obj, "states", {})
            available_states = tuple(sorted(state_type.__name__ for state_type in native_states))
            meaningful_boolean_states = {
                "Open", "ToggledOn", "Cooked", "Burnt", "Frozen", "Heated", "OnFire"
            }
            semantic_states: dict[str, bool] = {}
            for state_type, state in native_states.items():
                state_name = state_type.__name__
                if state_name not in meaningful_boolean_states:
                    continue
                try:
                    semantic_states[state_name] = bool(state.get_value())
                except Exception:
                    continue
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
                    semantic_states=semantic_states,
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
        if self._relation_cache is None:
            self._relation_cache = self.relation_candidates(catalog)
        return self._attach_relations(catalog, self._relation_cache)

    @staticmethod
    def _catalog_restore_metrics(
        before: tuple[ObjectState, ...],
        after: tuple[ObjectState, ...],
    ) -> tuple[float, bool]:
        before_by_id = {obj.instance_id: obj for obj in before}
        after_by_id = {obj.instance_id: obj for obj in after}
        if set(before_by_id) != set(after_by_id):
            return float("inf"), False
        maximum_error = 0.0
        discrete_state_equal = True
        for instance_id, before_object in before_by_id.items():
            after_object = after_by_id[instance_id]
            maximum_error = max(
                maximum_error,
                float(
                    np.max(
                        np.abs(
                            before_object.object_to_world
                            - after_object.object_to_world
                        )
                    )
                ),
            )
            if len(before_object.joint_values) != len(after_object.joint_values):
                return float("inf"), False
            maximum_error = max(
                maximum_error,
                max(
                    (
                        abs(float(a) - float(b))
                        for a, b in zip(
                            before_object.joint_values,
                            after_object.joint_values,
                            strict=True,
                        )
                    ),
                    default=0.0,
                ),
            )
            discrete_state_equal = discrete_state_equal and (
                before_object.semantic_states == after_object.semantic_states
                and before_object.relations == after_object.relations
            )
        return maximum_error, discrete_state_equal


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
        if self._relation_cache is None:
            self._relation_cache = self.relation_candidates(baseline_raw_catalog)
        relations = self._relation_cache
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
            sampler_macros = self._object_state_utils.m
            previous_high = int(sampler_macros.DEFAULT_HIGH_LEVEL_SAMPLING_ATTEMPTS)
            previous_low = int(sampler_macros.DEFAULT_LOW_LEVEL_SAMPLING_ATTEMPTS)
            with sampler_macros.unlocked():
                sampler_macros.DEFAULT_HIGH_LEVEL_SAMPLING_ATTEMPTS = int(
                    generation["native_relation_high_level_attempts"]
                )
                sampler_macros.DEFAULT_LOW_LEVEL_SAMPLING_ATTEMPTS = int(
                    generation["native_relation_low_level_attempts"]
                )
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
            finally:
                with sampler_macros.unlocked():
                    sampler_macros.DEFAULT_HIGH_LEVEL_SAMPLING_ATTEMPTS = previous_high
                    sampler_macros.DEFAULT_LOW_LEVEL_SAMPLING_ATTEMPTS = previous_low
            if not sampled:
                failures.append({"reason": "relation_sampler_failed", **relation})
                continue
            for _ in range(int(generation["settle_steps"])):
                for robot in self._env.robots:
                    robot.keep_still()
                self._og.sim.step_physics()
            self._og.sim.step_physics()
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
            self.load_snapshot(accepted_snapshot)
            restored_catalog = self._attach_relations(self.object_catalog(), (relation,))
            maximum_restore_error, restored_discrete_state = self._catalog_restore_metrics(
                catalog,
                restored_catalog,
            )
            restored_native_by_path = self._native_objects_by_path()
            relation_restored = bool(
                restored_native_by_path[relation["target_native_path"]]
                .states[state_type]
                .get_value(
                    restored_native_by_path[relation["reference_native_path"]]
                )
            )
            restore_tolerance = float(generation["snapshot_restore_tolerance"])
            restored_hash = exact_state_hash(
                restored_catalog,
                decimals=int(generation["exact_hash_decimals"]),
            )
            if (
                maximum_restore_error > restore_tolerance
                or not restored_discrete_state
                or not relation_restored
            ):
                failures.append(
                    {
                        "reason": "configuration_snapshot_restore_mismatch",
                        "maximum_restore_error": maximum_restore_error,
                        "restore_tolerance": restore_tolerance,
                        "restored_discrete_state": restored_discrete_state,
                        "relation_restored": relation_restored,
                        **relation,
                    }
                )
                continue
            checks["snapshot_restored"] = True
            return {
                "catalog": restored_catalog,
                "snapshot": accepted_snapshot,
                "exact_state_hash": restored_hash,
                "maximum_snapshot_restore_error": maximum_restore_error,
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

    def apply_atomic_intervention(
        self,
        seed: int,
        *,
        forced_type: InterventionType | None = None,
        excluded_target_ids: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Apply and verify exactly one v1 intervention in the currently loaded W0."""
        baseline_snapshot = self.dump_snapshot()
        baseline_catalog = self.object_catalog_with_relations()
        baseline_by_id = {obj.instance_id: obj for obj in baseline_catalog}
        native_by_path = self._native_objects_by_path()
        rng = np.random.default_rng(seed)
        weights = self.config["intervention"]["type_weights"]
        preferred = forced_type or choose_intervention_type(weights, rng)
        type_order = [preferred] + [
            intervention_type
            for intervention_type in InterventionType
            if intervention_type is not preferred and float(weights.get(intervention_type.value, 0.0)) > 0
        ]
        maximum_attempts = int(self.config["intervention"]["maximum_attempts"])
        failures: list[dict[str, Any]] = []
        attempt = 0
        for intervention_type in type_order:
            candidates = [
                obj
                for obj in eligible_intervention_targets(baseline_catalog, intervention_type)
                if obj.instance_id not in excluded_target_ids
            ]
            if intervention_type is InterventionType.RIGID_RELOCATION:
                candidates = [
                    obj
                    for obj in candidates
                    if any(
                        relation.get("predicate") == "OnTop"
                        and relation.get("reference_category") == "floors"
                        for relation in obj.relations
                    )
                ]
            if not candidates:
                failures.append({"reason": "no_eligible_targets", "type": intervention_type.value})
                continue
            for target_index in rng.permutation(len(candidates)):
                if attempt >= maximum_attempts:
                    break
                attempt += 1
                self.load_snapshot(baseline_snapshot)
                target = candidates[int(target_index)]
                native_target = native_by_path[target.native_path]
                atomic_baseline_catalog: tuple[ObjectState, ...] | None = None
                try:
                    if intervention_type is InterventionType.RIGID_RELOCATION:
                        event = propose_rigid_relocation(
                            target,
                            rng,
                            translation_range_m=(
                                float(self.config["intervention"]["translation_min_m"]),
                                float(self.config["intervention"]["translation_max_m"]),
                            ),
                            rotation_range_deg=(
                                float(self.config["intervention"]["rotation_min_deg"]),
                                float(self.config["intervention"]["rotation_max_deg"]),
                            ),
                        )
                        relation = next(
                            relation
                            for relation in target.relations
                            if relation.get("predicate") == "OnTop"
                            and relation.get("reference_category") == "floors"
                        )
                        reference = baseline_by_id[str(relation["reference_instance_id"])]
                        native_reference = native_by_path[reference.native_path]
                        delta_xy = np.asarray(event.parameters["translation_xy_m"], dtype=np.float64)
                        yaw_delta = float(event.parameters["yaw_delta_rad"])
                        candidate_transform = target.object_to_world.copy()
                        candidate_transform[:2, 3] += delta_xy
                        cosine, sine = np.cos(yaw_delta), np.sin(yaw_delta)
                        yaw_rotation = np.asarray(
                            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
                        )
                        candidate_transform[:3, :3] = yaw_rotation @ candidate_transform[:3, :3]
                        position, orientation = self._transform_utils.mat2pose(
                            self._th.as_tensor(candidate_transform, dtype=self._th.float32)
                        )
                        native_target.set_position_orientation(position=position, orientation=orientation)
                        for native_object in list(self._require_scene().objects):
                            if hasattr(native_object, "keep_still"):
                                native_object.keep_still()
                        self._og.sim.step_physics()
                        relation_valid = bool(
                            native_target.states[self._on_top_type].get_value(native_reference)
                        )
                        extra_collision = bool(
                            self._rigid_contact_api.is_in_contact(
                                scene_idx=native_target.scene.idx,
                                query_set=[native_target],
                                with_set=None,
                                ignore_set=[native_reference, native_target],
                                current_only=True,
                            )
                        )
                        if not relation_valid or extra_collision:
                            failures.append(
                                {
                                    "reason": "rigid_physics_validation_failed",
                                    "relation_valid": relation_valid,
                                    "extra_collision": extra_collision,
                                    "target": target.instance_id,
                                }
                            )
                            continue
                        # Physics validation is intentionally performed in the live
                        # world above, but even a single step can advance unrelated
                        # dynamic objects. Preserve the validated target pose, restore
                        # W0, and replay only that pose so the stored intervention is
                        # genuinely atomic rather than a one-step world transition.
                        validated_position, validated_orientation = (
                            value.detach().clone()
                            for value in native_target.get_position_orientation()
                        )
                        self.load_snapshot(baseline_snapshot)
                        atomic_baseline_catalog = self.object_catalog_with_relations()
                        native_target.set_position_orientation(
                            position=validated_position,
                            orientation=validated_orientation,
                        )
                        native_target.keep_still()
                    elif intervention_type is InterventionType.ARTICULATION:
                        event = propose_articulation(target, rng)
                        desired = np.asarray(event.after_object_state.joint_values, dtype=np.float32)
                        native_target.set_joint_positions(
                            self._th.as_tensor(desired, device=self._og.sim.device),
                            drive=False,
                        )
                        native_target.keep_still()
                        self._og.sim.step_physics()
                        validated_joint_positions = (
                            native_target.get_joint_positions().detach().clone()
                        )
                        self.load_snapshot(baseline_snapshot)
                        atomic_baseline_catalog = self.object_catalog_with_relations()
                        native_target.set_joint_positions(
                            validated_joint_positions,
                            drive=False,
                        )
                        native_target.keep_still()
                    else:
                        event = propose_state_change(target, rng)
                        state_name = str(event.parameters["state_name"])
                        state_type = next(
                            state_type
                            for state_type in native_target.states
                            if state_type.__name__ == state_name
                        )
                        applied = bool(
                            native_target.states[state_type].set_value(
                                bool(event.parameters["value_after"])
                            )
                        )
                        self._og.sim.step_physics()
                        if not applied or bool(native_target.states[state_type].get_value()) != bool(
                            event.parameters["value_after"]
                        ):
                            failures.append(
                                {
                                    "reason": "semantic_state_set_failed",
                                    "state_name": state_name,
                                    "target": target.instance_id,
                                }
                            )
                            continue
                        self.load_snapshot(baseline_snapshot)
                        atomic_baseline_catalog = self.object_catalog_with_relations()
                        reapplied = bool(
                            native_target.states[state_type].set_value(
                                bool(event.parameters["value_after"])
                            )
                        )
                        if not reapplied or bool(
                            native_target.states[state_type].get_value()
                        ) != bool(event.parameters["value_after"]):
                            failures.append(
                                {
                                    "reason": "semantic_state_replay_failed",
                                    "state_name": state_name,
                                    "target": target.instance_id,
                                }
                            )
                            continue
                except Exception as error:
                    failures.append(
                        {
                            "reason": "intervention_application_error",
                            "type": intervention_type.value,
                            "target": target.instance_id,
                            "error": str(error),
                        }
                    )
                    continue
                if atomic_baseline_catalog is None:
                    raise SimulatorUnavailableError(
                        "Intervention replay did not capture its restored atomic baseline"
                    )
                atomic_baseline_by_id = {obj.instance_id: obj for obj in atomic_baseline_catalog}
                after_catalog = self.object_catalog_with_relations()
                after_by_id = {obj.instance_id: obj for obj in after_catalog}
                after_target = after_by_id[target.instance_id]
                if intervention_type is InterventionType.RIGID_RELOCATION:
                    translation = float(
                        np.linalg.norm(
                            after_target.object_to_world[:3, 3] - target.object_to_world[:3, 3]
                        )
                    )
                    rotation_deg = float(
                        np.rad2deg(rotation_angle(target.object_to_world, after_target.object_to_world))
                    )
                    valid_range = (
                        float(self.config["intervention"]["translation_min_m"]) <= translation
                        <= float(self.config["intervention"]["translation_max_m"]) + 0.05
                        and float(self.config["intervention"]["rotation_min_deg"]) <= rotation_deg
                        <= float(self.config["intervention"]["rotation_max_deg"]) + 1.0
                    )
                    if not valid_range:
                        failures.append(
                            {
                                "reason": "rigid_delta_out_of_range",
                                "translation_m": translation,
                                "rotation_deg": rotation_deg,
                                "target": target.instance_id,
                            }
                        )
                        continue
                changed_ids = []
                for instance_id, before in atomic_baseline_by_id.items():
                    after = after_by_id[instance_id]
                    changed = (
                        np.linalg.norm(
                            before.object_to_world[:3, 3] - after.object_to_world[:3, 3]
                        )
                        > 1.0e-4
                        or rotation_angle(before.object_to_world, after.object_to_world) > 1.0e-4
                        or before.joint_values != after.joint_values
                        or before.semantic_states != after.semantic_states
                    )
                    if changed:
                        changed_ids.append(instance_id)
                if changed_ids != [target.instance_id]:
                    failures.append(
                        {
                            "reason": "non_atomic_environment_change",
                            "target": target.instance_id,
                            "changed_ids": changed_ids,
                        }
                    )
                    continue
                event = replace(event, after_object_state=after_target)
                return {
                    "event": event,
                    "catalog": after_catalog,
                    "snapshot": self.dump_snapshot(),
                    "attempt": attempt,
                    "changed_instance_ids": changed_ids,
                    "checks": {
                        "physically_valid": True,
                        "atomic_single_target": True,
                        "same_floor": after_target.floor_id == target.floor_id,
                        "same_room_when_known": target.room_id is None
                        or after_target.room_id == target.room_id,
                    },
                }
        self.load_snapshot(baseline_snapshot)
        raise SampleRejected(
            "atomic_intervention_sampling_failed",
            {"attempts": attempt, "failures": failures[:50]},
        )

    def dump_snapshot(self) -> np.ndarray:
        self._require_scene()
        state = self._og.sim.dump_state(serialized=True)
        return self._native_value(state).copy()

    def load_snapshot(self, snapshot: np.ndarray) -> None:
        self._require_scene()
        state = self._th.as_tensor(snapshot, dtype=self._th.float32, device=self._og.sim.device)
        simulator_state, consumed = self._og.sim.deserialize(state)
        if consumed != len(state):
            raise SimulatorUnavailableError(
                "OmniGibson snapshot deserialization consumed "
                f"{consumed} values out of {len(state)}"
            )
        # Simulator.load_state() always reapplies the scene-root transform,
        # even though MVWD never moves it. That USD transform edit invalidates
        # persistent SyntheticData render products and can segfault either in
        # set_world_pose() or on the following render. Restore only registry
        # entries whose serialized state actually changed; calling load_state
        # on every unchanged furniture object also invalidates SyntheticData.
        requires_handle_refresh = False
        for index, scene in enumerate(self._og.sim.scenes):
            scene_state = simulator_state[index]
            target_position = scene_state.get("pos")
            target_orientation = scene_state.get("ori")
            if target_position is not None:
                current_position, current_orientation = scene.get_position_orientation()
                if not (
                    self._th.allclose(current_position, target_position, atol=1.0e-6)
                    and self._th.allclose(current_orientation, target_orientation, atol=1.0e-6)
                ):
                    raise SimulatorUnavailableError(
                        "MVWD snapshots cannot restore a changed scene-root transform "
                        "while persistent render products are active"
                    )
            target_registry_state = scene_state.get("registry", scene_state)
            registries = (
                ("object_registry", scene.object_registry),
                ("system_registry", scene.system_registry),
            )
            previous_filters: list[tuple[Any, Any]] = []
            changed_by_registry: dict[str, list[str]] = {}
            development_robot_names = {robot.name for robot in self._env.robots}
            lightweight_robot_restores: set[str] = set()
            for registry_name, registry in registries:
                target_subregistry_state = target_registry_state.get(registry_name)
                if target_subregistry_state is None:
                    raise SimulatorUnavailableError(
                        f"OmniGibson snapshot is missing {registry_name}"
                    )
                changed_names: set[str] = set()
                for obj in registry.objects:
                    target_object_state = target_subregistry_state.get(obj.name)
                    if target_object_state is None:
                        continue
                    current_serialized = obj.dump_state(serialized=True)
                    target_serialized = obj.serialize(target_object_state)
                    is_development_object = (
                        registry_name == "object_registry"
                        and not self._using_final_robot
                    )
                    if is_development_object and obj.name not in development_robot_names:
                        requires_restore = self._development_object_requires_restore(
                            obj,
                            current_serialized,
                            target_serialized,
                            target_object_state,
                        )
                    else:
                        requires_restore = (
                            current_serialized.shape != target_serialized.shape
                            or not self._th.allclose(
                                current_serialized,
                                target_serialized,
                                rtol=0.0,
                                atol=1.0e-7,
                                equal_nan=True,
                            )
                        )
                    if requires_restore:
                        if (
                            registry_name == "object_registry"
                            and obj.name in development_robot_names
                        ):
                            self._restore_robot_snapshot_lightweight(
                                obj,
                                target_object_state,
                            )
                            lightweight_robot_restores.add(obj.name)
                        else:
                            changed_names.add(obj.name)
                previous_filter = registry._load_filter
                previous_filters.append((registry, previous_filter))
                registry.set_load_filter(
                    lambda obj, allowed=changed_names, prior=previous_filter: (
                        prior(obj) and obj.name in allowed
                    )
                )
                changed_by_registry[registry_name] = sorted(changed_names)
                # Lightweight robot root / joint restores preserve existing
                # articulation and sensor handles. A complete furniture or
                # system restore is the only operation here that can require
                # OmniGibson to rebuild its tensor views.
                requires_handle_refresh = requires_handle_refresh or bool(changed_names)
            try:
                scene.load_state(state=target_registry_state, serialized=False)
            finally:
                for registry, previous_filter in previous_filters:
                    registry.set_load_filter(previous_filter)
            self._runtime_findings["snapshot_changed_registry_entries"] = (
                changed_by_registry
            )
            self._runtime_findings["snapshot_lightweight_robot_restores"] = sorted(
                lightweight_robot_restores
            )
        if self._using_final_robot:
            if requires_handle_refresh:
                self._rebuild_final_robot_capture_sensor()
                self._og.sim.update_handles()
            self._runtime_findings["final_robot_snapshot_sensor_lifecycle"] = (
                "scene_root_retained+robot_root_joint_restore+"
                "changed_registry_entries_restored+furniture_or_system_handle_refresh+"
                "capture_graph_rebuilt_before_handle_refresh"
            )
            self._runtime_findings["snapshot_handles_refreshed"] = (
                requires_handle_refresh
            )
        else:
            self._runtime_findings["development_snapshot_sensor_lifecycle"] = (
                "scene_root_retained+lightweight_robot_restore+changed_registry_entries_restored+graph_retained"
            )

    def _restore_robot_snapshot_lightweight(self, robot: Any, state: dict[str, Any]) -> None:
        """Restore robot physics state without reloading controllers or sensor graphs."""
        root_state = state["root_link"]
        robot.set_position_orientation(
            position=root_state["pos"],
            orientation=root_state["ori"],
        )
        robot.set_linear_velocity(root_state["lin_vel"])
        robot.set_angular_velocity(root_state["ang_vel"])
        if robot.n_joints > 0:
            robot.set_joint_positions(state["joint_pos"], drive=False)
            robot.set_joint_velocities(state["joint_vel"])

    def _development_object_requires_restore(
        self,
        obj: Any,
        current: Any,
        target: Any,
        target_state: dict[str, Any],
    ) -> bool:
        """Compare only object state represented in MVWD's public schema."""
        joint_count = int(obj.n_joints)
        root_state_size = int(
            obj.root_link.serialize(target_state["root_link"]).numel()
        )
        joint_position_start = 1 + root_state_size
        entity_state_size = joint_position_start + 2 * joint_count
        if current.shape != target.shape or current.numel() < entity_state_size:
            return True
        if not self._th.allclose(
            current[1:4], target[1:4], rtol=0.0, atol=1.0e-6
        ):
            return True
        orientation_delta = self._th.minimum(
            self._th.max(self._th.abs(current[4:8] - target[4:8])),
            self._th.max(self._th.abs(current[4:8] + target[4:8])),
        )
        if bool(orientation_delta > 1.0e-6):
            return True
        if joint_count > 0 and not self._th.allclose(
            current[joint_position_start : joint_position_start + joint_count],
            target[joint_position_start : joint_position_start + joint_count],
            rtol=0.0,
            atol=1.0e-6,
        ):
            return True
        meaningful_boolean_states = {
            "Open",
            "ToggledOn",
            "Cooked",
            "Burnt",
            "Frozen",
            "Heated",
            "OnFire",
        }
        target_non_kinematic = target_state.get("non_kin", {})
        for state_type, state_instance in getattr(obj, "states", {}).items():
            state_name = state_type.__name__
            if (
                state_name not in meaningful_boolean_states
                or not state_instance.stateful
                or state_name not in target_non_kinematic
            ):
                continue
            current_state = state_instance.dump_state(serialized=True)
            target_state_value = state_instance.serialize(
                target_non_kinematic[state_name]
            )
            if (
                current_state.shape != target_state_value.shape
                or not self._th.allclose(
                    current_state,
                    target_state_value,
                    rtol=0.0,
                    atol=1.0e-7,
                    equal_nan=True,
                )
            ):
                return True
        return False

    def _restore_final_robot_mast_mount(self, robot: Any) -> None:
        """Restore the passive mast after a rollout physics step."""
        if not self._using_final_robot:
            return
        expected_mount = self._development_camera_mounts.get(robot.name)
        if expected_mount is None:
            return
        mast_extension = float(expected_mount[2, 3]) - float(
            min(self.config["camera"]["heights_m"])
        )
        mast_joint = next(
            joint for name, joint in robot.joints.items()
            if name.endswith("mvwd_mast_joint")
        )
        mast_joint.set_pos(mast_extension, drive=False)

    def _robot_eroded_traversability(self, floor_index: int, robot: Any) -> Any:
        """Erode traversability by the robot footprint and one map-cell margin."""
        trav_map = self._require_scene().trav_map
        source = self._th.clone(trav_map.floor_map[floor_index])
        chassis_extent = self._native_value(
            robot.reset_joint_pos_aabb_extent[:2]
        ).astype(np.float64)
        footprint_radius_m = float(np.linalg.norm(chassis_extent) / 2.0)
        # Preserve one full traversability cell as a coarse-map safety margin.
        safety_margin_m = min(0.1, float(trav_map.map_resolution))
        clearance_m = footprint_radius_m + safety_margin_m
        radius_pixels = max(
            1,
            int(np.ceil(clearance_m / float(trav_map.map_resolution))),
        )
        # OG 3.9.2 hardcodes another 0.2 m and passes radius_pixels as the cv2
        # kernel width, which erodes by only about half that combined radius.
        # Use the actual circumscribed footprint plus the explicit margin above,
        # a true 2r+1 kernel, and treat the map boundary as occupied.
        obstacles = (source != 255).to(dtype=self._th.float32)[None, None]
        padded = self._th.nn.functional.pad(
            obstacles,
            (radius_pixels, radius_pixels, radius_pixels, radius_pixels),
            value=1.0,
        )
        blocked = self._th.nn.functional.max_pool2d(
            padded,
            kernel_size=2 * radius_pixels + 1,
            stride=1,
        )[0, 0] > 0
        eroded = self._th.where(
            blocked,
            self._th.zeros_like(source),
            self._th.full_like(source, 255),
        )
        self._runtime_findings["traversability_clearance"] = {
            "chassis_extent_xy_m": chassis_extent.tolist(),
            "footprint_radius_m": footprint_radius_m,
            "safety_margin_m": safety_margin_m,
            "clearance_m": clearance_m,
            "radius_pixels": radius_pixels,
            "map_resolution_m": float(trav_map.map_resolution),
        }
        return eroded

    def _external_robot_contact_pairs(
        self,
        robot: Any,
        floors: list[Any],
    ) -> list[list[str]]:
        contact_pairs = sorted(
            self._rigid_contact_api.get_contact_pairs(
                scene_idx=robot.scene.idx,
                query_set=[robot],
                with_set=None,
                current_only=True,
            )
        )
        ignored_prefixes = (
            str(robot.prim_path),
            *(str(floor.prim_path) for floor in floors),
        )
        return [
            [query_path, other_path]
            for query_path, other_path in contact_pairs
            if not any(
                other_path == prefix or other_path.startswith(prefix + "/")
                for prefix in ignored_prefixes
            )
        ]

    def _preflight_trajectory_contacts(
        self,
        by_id: dict[str, Trajectory],
        robots: dict[str, Any],
        floors: list[Any],
        frames: int,
    ) -> None:
        """Reject colliding trajectories before any expensive image capture."""
        for frame_index in range(frames):
            for robot_id, robot in robots.items():
                planned = by_id[robot_id].base_to_world[frame_index]
                position, orientation = self._transform_utils.mat2pose(
                    self._th.as_tensor(planned, dtype=self._th.float32)
                )
                robot.set_position_orientation(position=position, orientation=orientation)
                robot.keep_still()
            self._og.sim.step_physics()
            for robot_id, robot in robots.items():
                external_pairs = self._external_robot_contact_pairs(robot, floors)
                if external_pairs:
                    raise SampleRejected(
                        "trajectory_collision_detected",
                        {
                            "frame_index": frame_index,
                            "robot_id": robot_id,
                            "contact_pairs": external_pairs[:50],
                            "stage": "physics_preflight",
                        },
                    )
                self._restore_final_robot_mast_mount(robot)
                robot.keep_still()

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
        world_xy, is_path_traversable, _ = self._trajectory_traversability(floor_index, robots[0])
        if len(world_xy) < 3:
            raise SimulatorUnavailableError(f"Floor {floor_index} has fewer than three traversable pixels")
        z = float(scene.get_floor_height(floor_index))
        candidates = np.column_stack((world_xy, np.full(len(world_xy), z)))
        cluster_radius = float(placement["cluster_radius_m"])
        preferred_radius = min(
            cluster_radius,
            float(placement.get("preferred_cluster_radius_m", cluster_radius)),
        )
        trajectory_minimum_length = float(
            self.config["trajectory"]["path_length_min_m"]
        )
        trajectory_ready_radius = min(
            cluster_radius,
            preferred_radius + trajectory_minimum_length,
        )
        candidate_radii = [preferred_radius]
        for radius in (trajectory_ready_radius, cluster_radius):
            if radius > candidate_radii[-1] + 1.0e-9:
                candidate_radii.append(radius)
        minimum_distance = float(placement["minimum_pairwise_distance_m"])
        trajectory_headroom = float(placement["trajectory_separation_headroom_m"])
        selection_distance = minimum_distance + trajectory_headroom
        maximum_attempts = int(placement["maximum_attempts"])
        selected: list[np.ndarray] = []
        selected_focus_xy_by_robot: list[np.ndarray] = []
        selected_shared_heading: float | None = None
        heading_jitter = np.deg2rad(float(placement.get("heading_jitter_deg", 5.0)))
        heading_tolerance = np.deg2rad(
            float(self.config["trajectory"]["initial_heading_tolerance_deg"])
        )
        maximum_focus_bearing_error = max(0.0, heading_tolerance - heading_jitter)

        def has_safe_forward_line(source_xy: np.ndarray, target_xy: np.ndarray) -> bool:
            distance = float(np.linalg.norm(target_xy - source_xy))
            samples = max(2, int(np.ceil(distance / 0.025)) + 1)
            line = source_xy + np.linspace(0.0, 1.0, samples)[:, None] * (
                target_xy - source_xy
            )
            return is_path_traversable(line)

        attempts = 0
        attempts_by_radius: dict[str, int] = {}
        effective_radius = preferred_radius
        for radius in candidate_radii:
            radius_attempts = 0
            for center_index in rng.permutation(len(candidates))[:maximum_attempts]:
                attempts += 1
                radius_attempts += 1
                center = candidates[int(center_index)]
                local = candidates[np.linalg.norm(candidates[:, :2] - center[:2], axis=1) <= radius]
                if len(local) < 3:
                    continue
                selection = [local[int(rng.integers(len(local)))]]
                # Keep the group compact while respecting the hard separation.
                # Compact starts materially improve shared camera coverage.
                tie_order = rng.permutation(len(local))
                while len(selection) < 3:
                    pairwise = np.stack(
                        [np.linalg.norm(local[:, :2] - point[:2], axis=1) for point in selection], axis=1
                    )
                    nearest = pairwise.min(axis=1)
                    valid = nearest >= selection_distance
                    if not np.any(valid):
                        break
                    compactness = pairwise.max(axis=1)
                    score = np.where(valid, compactness, np.inf)
                    best = tie_order[np.argmin(score[tie_order])]
                    selection.append(local[int(best)])
                if len(selection) == 3:
                    positions_xy = np.stack(selection, axis=0)[:, :2]
                    cluster_center = positions_xy.mean(axis=0)
                    heading_targets = world_xy[
                        rng.permutation(len(world_xy))[: min(32, len(world_xy))]
                    ]
                    heading_candidates = np.arctan2(
                        heading_targets[:, 1] - cluster_center[1],
                        heading_targets[:, 0] - cluster_center[0],
                    )
                    for proposed_heading in heading_candidates:
                        per_robot_focus: list[np.ndarray] = []
                        for point in selection:
                            deltas = world_xy - point[:2]
                            distances = np.linalg.norm(deltas, axis=1)
                            bearings = np.arctan2(deltas[:, 1], deltas[:, 0])
                            bearing_errors = np.abs(
                                (bearings - proposed_heading + np.pi)
                                % (2.0 * np.pi)
                                - np.pi
                            )
                            forward_indices = np.flatnonzero(
                                (distances >= trajectory_minimum_length)
                                & (distances <= float(
                                    self.config["trajectory"]["path_length_max_m"]
                                ))
                                & (bearing_errors <= maximum_focus_bearing_error)
                            )
                            forward_order = forward_indices[
                                np.argsort(bearing_errors[forward_indices])
                            ]
                            safe_focus = next(
                                (
                                    world_xy[int(focus_index)]
                                    for focus_index in forward_order[:32]
                                    if has_safe_forward_line(
                                        point[:2], world_xy[int(focus_index)]
                                    )
                                ),
                                None,
                            )
                            if safe_focus is None:
                                break
                            per_robot_focus.append(safe_focus.copy())
                        if len(per_robot_focus) == len(selection):
                            selected = selection
                            selected_focus_xy_by_robot = per_robot_focus
                            selected_shared_heading = float(proposed_heading)
                            effective_radius = radius
                            break
                    if selected:
                        break
            attempts_by_radius[f"{radius:.6g}"] = radius_attempts
            if selected:
                break
        if len(selected) != 3:
            raise SimulatorUnavailableError(
                f"Failed clustered placement after {attempts} candidate centers "
                f"across radii {candidate_radii} on floor {floor_index}"
            )
        sampled_heights: dict[str, float] = {}
        pitch = np.deg2rad(float(self.config["camera"]["pitch_deg"]))
        cosine, sine = np.cos(pitch), np.sin(pitch)
        # Columns are OpenCV camera right, down, forward expressed in robot base coordinates.
        camera_rotation_base = np.array([[0.0, sine, cosine], [-1.0, 0.0, 0.0], [0.0, -cosine, sine]])
        cv_to_usd = np.diag([1.0, -1.0, -1.0, 1.0])
        cluster_center_xy = np.mean(np.stack(selected, axis=0)[:, :2], axis=0)
        assert selected_shared_heading is not None
        assert len(selected_focus_xy_by_robot) == len(selected)
        common_view_distance = float(placement["common_view_focus_distance_m"])
        common_view_focus_xy = cluster_center_xy + common_view_distance * np.array(
            [np.cos(selected_shared_heading), np.sin(selected_shared_heading)]
        )
        focus_distances_m = [
            float(np.linalg.norm(focus - point[:2]))
            for focus, point in zip(selected_focus_xy_by_robot, selected, strict=True)
        ]
        heading_offsets = np.linspace(-heading_jitter, heading_jitter, len(robots))
        placement_yaws: dict[str, float] = {}
        for robot, point, heading_offset in zip(robots, selected, heading_offsets, strict=True):
            delta_to_focus = common_view_focus_xy - point[:2]
            base_yaw = float(np.arctan2(delta_to_focus[1], delta_to_focus[0]))
            yaw = float((base_yaw + heading_offset + np.pi) % (2.0 * np.pi) - np.pi)
            placement_yaws[robot.name] = yaw
            orientation = self._transform_utils.euler2quat(self._th.tensor([0.0, 0.0, yaw]))
            robot.set_position_orientation(
                position=self._th.as_tensor(point, dtype=self._th.float32), orientation=orientation
            )
            height = float(rng.choice(heights))
            sampled_heights[robot.name] = height
            if self._using_final_robot:
                mast_joint = next(
                    joint for name, joint in robot.joints.items()
                    if name.endswith("mvwd_mast_joint")
                )
                mast_joint.set_pos(height - float(min(heights)), drive=False)
            base_to_world = self._pose_matrix(robot)
            camera_to_base = np.eye(4)
            camera_to_base[:3, :3] = camera_rotation_base
            camera_to_base[:3, 3] = [
                0.08 if self._using_final_robot else 0.0, 0.0, height
            ]
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
            if not self._using_final_robot:
                sensor.set_position_orientation(position=camera_position, orientation=camera_orientation)
            self._development_camera_mounts[robot.name] = camera_to_base.copy()
        settle_steps = min(30, int(self.config["generation"]["settle_steps"]))
        # Propagate articulated camera transforms into their stable RGB/depth
        # render products. Final-robot segmentation uses a separate raw AOV,
        # so no InstanceMapping graph is ticked here.
        rendering_settle_steps = min(2, settle_steps)
        for _ in range(settle_steps):
            for robot in robots:
                robot.keep_still()
            self._og.sim.step_physics()
        # Keep the sampled base poses exactly on their validated map pixels;
        # passive wheel settling must not move a trajectory start into an
        # adjacent non-traversable cell.
        for robot, point in zip(robots, selected, strict=True):
            yaw = placement_yaws[robot.name]
            orientation = self._transform_utils.euler2quat(
                self._th.tensor([0.0, 0.0, yaw])
            )
            robot.set_position_orientation(
                position=self._th.as_tensor(point, dtype=self._th.float32),
                orientation=orientation,
            )
            self._restore_final_robot_mast_mount(robot)
            robot.keep_still()
        # A full step is required for articulation-mounted VisionSensor render
        # products; bare render ticks can retain a stale camera transform.
        for _ in range(rendering_settle_steps):
            for robot in robots:
                robot.keep_still()
            self._og.sim.step()
        # The static traversability raster does not move with randomized
        # furniture. Reject such starts immediately, before camera-overlap and
        # geodesic candidate generation make the same frame-0 discovery much
        # more expensively in rollout preflight.
        floors = [
            obj
            for obj in scene.objects
            if str(getattr(obj, "category", "")) == "floors"
        ]
        for robot in robots:
            external_pairs = self._external_robot_contact_pairs(robot, floors)
            if external_pairs:
                raise SampleRejected(
                    "initial_robot_collision",
                    {"robot_id": robot.name, "contact_pairs": external_pairs[:50]},
                )
        self._runtime_findings["development_camera_heights_m"] = sampled_heights
        self._runtime_findings["development_camera_settle_steps"] = settle_steps
        self._runtime_findings["development_camera_rendering_settle_steps"] = rendering_settle_steps
        self._runtime_findings["placement_attempts"] = attempts
        self._runtime_findings["placement_attempts_by_radius_m"] = attempts_by_radius
        self._runtime_findings["placement_pairwise_minimum_m"] = minimum_distance
        self._runtime_findings["placement_pairwise_target_m"] = selection_distance
        self._runtime_findings["placement_candidate_count"] = len(candidates)
        self._runtime_findings["placement_common_view_focus_xy"] = (
            common_view_focus_xy.tolist()
        )
        self._runtime_findings["placement_heading_mode"] = (
            "distant_common_focus_with_independent_robot_eroded_los_corridors"
        )
        self._runtime_findings["placement_cluster_center_xy"] = cluster_center_xy.tolist()
        self._runtime_findings["placement_focus_xy_by_robot"] = {
            robot.name: focus.tolist()
            for robot, focus in zip(robots, selected_focus_xy_by_robot, strict=True)
        }
        self._runtime_findings["placement_focus_distances_m"] = focus_distances_m
        self._runtime_findings["placement_yaws_rad"] = placement_yaws
        self._runtime_findings["placement_source"] = "robot-eroded traversability map"
        self._runtime_findings["placement_effective_radius_m"] = effective_radius
        return sampled_heights

    def _trajectory_traversability(
        self,
        floor_index: int,
        robot: Any,
    ) -> tuple[
        np.ndarray, Callable[[np.ndarray], bool],
        Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, float] | None],
    ]:
        """Return strict robot-eroded candidates, validator, and OG planner."""
        scene = self._require_scene()
        trav_map = scene.trav_map
        eroded = trav_map._erode_trav_map(
            self._th.clone(trav_map.floor_map[floor_index]), robot=robot
        )
        pixels = self._th.stack(self._th.where(eroded == 255), dim=1)
        world_xy = self._native_value(trav_map.map_to_world(pixels)).astype(np.float64)

        eroded_native = self._native_value(eroded) == 255
        height, width = eroded_native.shape
        def is_path_traversable(path_xy: np.ndarray) -> bool:
            points = np.asarray(path_xy, dtype=np.float64)
            map_points = self._native_value(
                trav_map.world_to_map(
                    self._th.as_tensor(points, dtype=self._th.float32)
                )
            ).astype(int)
            rows = map_points[:, 0]
            columns = map_points[:, 1]
            inside = (
                (rows >= 0)
                & (rows < height)
                & (columns >= 0)
                & (columns < width)
            )
            return bool(
                np.all(inside)
                and np.all(eroded_native[rows, columns])
            )

        def plan_segment(
            source_xy: np.ndarray,
            target_xy: np.ndarray,
        ) -> tuple[np.ndarray, float] | None:
            original_waypoint_interval = int(trav_map.waypoint_interval)
            trav_map.waypoint_interval = 1
            try:
                path, distance = scene.get_shortest_path(
                    floor_index,
                    self._th.as_tensor(source_xy, dtype=self._th.float32),
                    self._th.as_tensor(target_xy, dtype=self._th.float32),
                    entire_path=True,
                    robot=robot,
                )
            finally:
                trav_map.waypoint_interval = original_waypoint_interval
            if path is None or distance is None:
                return None
            return (
                self._native_value(path).astype(np.float64),
                float(self._native_value(distance)),
            )

        self._runtime_findings["trajectory_planner"] = {
            "api": "scene.get_shortest_path",
            "entire_path": True,
            "planner_waypoint_interval": 1,
            "native_waypoint_interval": int(trav_map.waypoint_interval),
            "robot_eroded": True,
            "erosion_source": "installed_omnigibson_3.9.2",
        }
        return world_xy, is_path_traversable, plan_segment

    def sample_robot_trajectories(self, seed: int) -> tuple[tuple[Trajectory, ...], dict[str, Any]]:
        observations = self.robot_observations()
        starts = {robot_id: record["base_to_world"] for robot_id, record in observations.items()}
        # A serialized PhysX restore can leave the passive Nova mast a few
        # millimetres away from its calibrated joint position until the next
        # rollout step. That transient pose must not become the trajectory's
        # camera extrinsic: playback restores the calibrated mast on every
        # frame, and paired before/after trajectories require one frozen mount.
        mounts = {
            robot_id: self._development_camera_mounts.get(
                robot_id, record["camera_to_base"]
            ).copy()
            for robot_id, record in observations.items()
        }
        floor_heights = np.asarray(self._floor_heights())
        robot_floor_indices = {
            int(np.argmin(np.abs(floor_heights - transform[2, 3]))) for transform in starts.values()
        }
        if len(robot_floor_indices) != 1:
            raise SampleRejected("robots_span_multiple_floors", {"floor_indices": sorted(robot_floor_indices)})
        floor_index = robot_floor_indices.pop()
        world_xy, is_path_traversable, plan_segment = self._trajectory_traversability(
            floor_index, self._env.robots[0]
        )
        trajectory_config = self.config["trajectory"]
        trajectories = sample_geodesic_trajectory_set(
            starts,
            mounts,
            world_xy,
            float(floor_heights[floor_index]),
            np.random.default_rng(seed),
            frames=int(self.config["dataset"]["frames"]),
            fps=float(self.config["dataset"]["fps"]),
            path_length_range_m=(
                float(trajectory_config["path_length_min_m"]),
                float(trajectory_config["path_length_max_m"]),
            ),
            minimum_pairwise_distance_m=float(self.config["placement"]["minimum_pairwise_distance_m"]),
            maximum_linear_speed_mps=float(trajectory_config["maximum_linear_speed_mps"]),
            maximum_angular_speed_radps=float(trajectory_config["maximum_angular_speed_radps"]),
            maximum_acceleration_mps2=float(trajectory_config["maximum_acceleration_mps2"]),
            plan_segment=plan_segment,
            is_path_traversable=is_path_traversable,
            path_family_weights=trajectory_config["path_family_weights"],
            minimum_waypoint_trajectories=int(
                trajectory_config["minimum_waypoint_trajectories"]
            ),
            initial_heading_tolerance_rad=np.deg2rad(
                float(trajectory_config["initial_heading_tolerance_deg"])
            ),
            line_validation_spacing_m=float(trajectory_config["line_validation_spacing_m"]),
            smoothing_validation_spacing_m=float(trajectory_config["smoothing_validation_spacing_m"]),
            smoothing_strengths=trajectory_config["smoothing_strengths"],
            candidate_pool_size=int(trajectory_config["candidate_pool_size"]),
            maximum_attempts=int(trajectory_config["sampling_maximum_attempts"]),
            joint_pool_rounds=int(trajectory_config["joint_pool_rounds"]),
        )
        metrics = {
            "floor_index": floor_index,
            "traversable_candidate_count": int(len(world_xy)),
            "robots": {
                trajectory.robot_id: {
                    **trajectory_kinematic_metrics(trajectory),
                    "path_family": trajectory.path_family,
                    "control_waypoints_xy": trajectory.control_waypoints_xy.tolist(),
                    "planner_geodesic_length_m": float(
                        trajectory.metadata["planner_geodesic_length_m"]
                    ),
                }
                for trajectory in trajectories
            },
            "minimum_pairwise_distance_m": min(
                float(
                    np.linalg.norm(
                        trajectories[left].base_to_world[:, :2, 3]
                        - trajectories[right].base_to_world[:, :2, 3],
                        axis=1,
                    ).min()
                )
                for left in range(len(trajectories))
                for right in range(left + 1, len(trajectories))
            ),
        }
        return trajectories, metrics

    def trajectory_traversability_inspection(self, floor_index: int) -> dict[str, Any]:
        """Expose the exact robot-eroded planning raster for inspection only."""
        trav_map = self._require_scene().trav_map
        eroded = trav_map._erode_trav_map(
            self._th.clone(trav_map.floor_map[floor_index]),
            robot=self._env.robots[0],
        )
        return {
            "traversable": (self._native_value(eroded) == 255).astype(np.uint8),
            "map_resolution_m": float(trav_map.map_resolution),
            "map_size": int(trav_map.map_size),
        }

    def place_robots_at_trajectory_frame(
        self,
        trajectories: tuple[Trajectory, ...],
        frame_index: int,
    ) -> None:
        """Place all robots at one synchronized frame and refresh their sensors."""
        by_id = {trajectory.robot_id: trajectory for trajectory in trajectories}
        robots = {robot.name: robot for robot in self._env.robots}
        if by_id.keys() != robots.keys():
            raise SampleRejected(
                "trajectory_robot_identity_mismatch",
                {"trajectory_ids": sorted(by_id), "robot_ids": sorted(robots)},
            )
        for robot_id, robot in robots.items():
            planned = by_id[robot_id].base_to_world[frame_index]
            position, orientation = self._transform_utils.mat2pose(
                self._th.as_tensor(planned, dtype=self._th.float32)
            )
            robot.set_position_orientation(position=position, orientation=orientation)
            self._restore_final_robot_mast_mount(robot)
            robot.keep_still()
        self._og.sim.step_physics()
        for robot_id, robot in robots.items():
            planned = by_id[robot_id].base_to_world[frame_index]
            position, orientation = self._transform_utils.mat2pose(
                self._th.as_tensor(planned, dtype=self._th.float32)
            )
            robot.set_position_orientation(position=position, orientation=orientation)
            self._restore_final_robot_mast_mount(robot)
            robot.keep_still()
        for _ in range(2):
            self._og.sim.render()

    def _reshape_vision_observation(self, sensor: Any, modality: str, value: Any) -> np.ndarray:
        """Restore image axes when Replicator returns a flattened render variable."""
        array = np.asarray(self._native_value(value))
        height = int(sensor.image_height)
        width = int(sensor.image_width)
        if array.ndim >= 2 and array.shape[:2] == (height, width):
            return array
        pixels = height * width
        if pixels <= 0 or array.size == 0 or array.size % pixels:
            raise SampleRejected(
                "vision_observation_shape_invalid",
                {
                    "sensor": str(sensor.prim_path),
                    "modality": modality,
                    "shape": list(array.shape),
                    "expected_height": height,
                    "expected_width": width,
                },
            )
        channels = array.size // pixels
        reshaped = array.reshape(height, width, channels)
        if channels == 1:
            reshaped = reshaped[..., 0]
        self._runtime_findings.setdefault("reshaped_vision_modalities", {})[
            f"{sensor.name}:{modality}"
        ] = {
            "source_shape": list(array.shape),
            "output_shape": list(reshaped.shape),
        }
        return reshaped

    def _resample_vision_observation(
        self,
        sensor: Any,
        modality: str,
        value: Any,
        *,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Nearest-resample an existing render product without rebuilding its graph."""
        array = self._reshape_vision_observation(sensor, modality, value)
        source_height, source_width = array.shape[:2]
        if (source_height, source_width) == (height, width):
            return array
        rows = np.rint(np.linspace(0, source_height - 1, height)).astype(np.int64)
        columns = np.rint(np.linspace(0, source_width - 1, width)).astype(np.int64)
        return array[rows[:, None], columns[None, :]]

    def _bev_capture_spans(
        self,
        sensor: Any,
        calibration: BEVCalibration,
    ) -> tuple[float, float]:
        """Return aspect-correct world spans for an existing render product."""
        source_width = int(sensor.image_width)
        source_height = int(sensor.image_height)
        if source_width <= 0 or source_height <= 0:
            raise GeometryError(
                f"Invalid BEV capture resolution {source_width}x{source_height}"
            )
        capture_aspect = source_width / source_height
        world_width = float(
            calibration.world_bounds[2] - calibration.world_bounds[0]
        )
        world_height = float(
            calibration.world_bounds[3] - calibration.world_bounds[1]
        )
        capture_world_width = max(world_width, world_height * capture_aspect)
        capture_world_height = capture_world_width / capture_aspect
        self._runtime_findings["final_robot_bev_capture_geometry"] = {
            "source_resolution": [source_width, source_height],
            "target_resolution": [calibration.width, calibration.height],
            "requested_world_span_m": [world_width, world_height],
            "capture_world_span_m": [capture_world_width, capture_world_height],
        }
        return capture_world_width, capture_world_height

    def _configure_bev_camera(
        self,
        sensor: Any,
        calibration: BEVCalibration,
    ) -> tuple[float, float]:
        """Configure an orthographic camera without distorting its render product."""
        capture_width, capture_height = self._bev_capture_spans(sensor, calibration)
        with self._og.sim.editing_usd():
            usd_camera = self._lazy.pxr.UsdGeom.Camera(sensor.prim)
            usd_camera.GetProjectionAttr().Set(
                self._lazy.pxr.UsdGeom.Tokens.orthographic
            )
            # Isaac Sim expresses USD camera aperture in tenths of a world unit.
            usd_camera.GetHorizontalApertureAttr().Set(10.0 * capture_width)
            usd_camera.GetVerticalApertureAttr().Set(10.0 * capture_height)
        return capture_width, capture_height

    def _resample_bev_observation(
        self,
        sensor: Any,
        modality: str,
        value: Any,
        calibration: BEVCalibration,
    ) -> np.ndarray:
        """Center-crop an aspect-correct BEV capture and nearest-resample it."""
        array = self._reshape_vision_observation(sensor, modality, value)
        source_height, source_width = array.shape[:2]
        capture_width, capture_height = self._bev_capture_spans(sensor, calibration)
        world_width = float(calibration.world_bounds[2] - calibration.world_bounds[0])
        world_height = float(calibration.world_bounds[3] - calibration.world_bounds[1])
        columns = (source_width - 1) / 2.0 + (
            (np.arange(calibration.width, dtype=np.float64) + 0.5)
            / calibration.width
            - 0.5
        ) * source_width * world_width / capture_width
        rows = (source_height - 1) / 2.0 + (
            (np.arange(calibration.height, dtype=np.float64) + 0.5)
            / calibration.height
            - 0.5
        ) * source_height * world_height / capture_height
        columns = np.clip(np.rint(columns).astype(np.int64), 0, source_width - 1)
        rows = np.clip(np.rint(rows).astype(np.int64), 0, source_height - 1)
        return array[rows[:, None], columns[None, :]]

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
            requested_usd_camera_to_world = self._pose_matrix(sensor)
            usd_camera_to_world = requested_usd_camera_to_world
            observation_sensor = sensor
            capture_pose_translation_error = 0.0
            capture_pose_rotation_error = 0.0
            if self._using_final_robot:
                observation_sensor = self._final_robot_capture_sensor
                if observation_sensor is None:
                    raise SimulatorUnavailableError(
                        "Final-robot capture sensor is unavailable"
                    )
                self._configure_final_robot_ego_capture(observation_sensor)
                sensor_position, sensor_orientation = sensor.get_position_orientation()
                observation_sensor.set_position_orientation(
                    position=sensor_position, orientation=sensor_orientation
                )
                capture_settle_render_ticks = 4
                for _ in range(capture_settle_render_ticks):
                    self._og.sim.render()
                self._runtime_findings["final_robot_capture_settle_render_ticks"] = (
                    capture_settle_render_ticks
                )
            if self._using_final_robot:
                observation, info = self._get_final_robot_capture_observation(
                    observation_sensor
                )
                mounted_observation, mounted_info = sensor.get_obs()
                for mounted_modality in ("rgb", "depth_linear"):
                    if mounted_modality not in mounted_observation:
                        raise SimulatorUnavailableError(
                            "Final-robot mounted camera is missing modality: "
                            f"{mounted_modality}"
                        )
                    observation[mounted_modality] = mounted_observation[
                        mounted_modality
                    ]
                    info[mounted_modality] = mounted_info.get(mounted_modality, {})
                usd_camera_to_world = self._pose_matrix(observation_sensor)
                capture_pose_translation_error = float(
                    np.linalg.norm(
                        usd_camera_to_world[:3, 3]
                        - requested_usd_camera_to_world[:3, 3]
                    )
                )
                capture_relative_rotation = (
                    requested_usd_camera_to_world[:3, :3].T
                    @ usd_camera_to_world[:3, :3]
                )
                capture_pose_rotation_error = float(
                    np.arccos(
                        np.clip(
                            (np.trace(capture_relative_rotation) - 1.0) / 2.0,
                            -1.0,
                            1.0,
                        )
                    )
                )
            else:
                observation, info = observation_sensor.get_obs()
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
                "modalities": {
                    name: self._reshape_vision_observation(observation_sensor, name, value)
                    for name, value in observation.items()
                },
                "info": info,
                "camera_to_world": camera_to_world,
                "base_to_world": base_to_world,
                "camera_to_base": camera_to_base,
                "expected_camera_to_base": expected_mount,
                "mount_translation_error_m": translation_error,
                "mount_rotation_error_rad": rotation_error,
                "mount_orthonormality_error": mount_orthonormality_error,
                "capture_pose_translation_error_m": capture_pose_translation_error,
                "capture_pose_rotation_error_rad": capture_pose_rotation_error,
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

    def _refresh_physics_handles_after_sensor_edit(self) -> None:
        """Rebuild simulator and articulation views invalidated by a sensor stage edit."""
        simulator = self._og.sim
        if simulator.is_playing():
            simulator.update_handles()

    def _initialize_development_bev_sensor(self) -> None:
        """Create the development BEV render product before any physics snapshot."""
        if self._development_bev_sensor is not None:
            return
        calibration = self.calibrated_floor_bounds(
            0,
            float(self.config["bev"]["environment_meters_per_pixel"]),
            float(self.config["bev"]["bounds_margin_m"]),
        )
        sensor_names = self._configured_bev_sensor_names()
        camera = self._vision_sensor_type(
            relative_prim_path="/mvwd_persistent_development_bev_camera",
            name="mvwd_persistent_development_bev_camera",
            modalities=sensor_names,
            image_width=calibration.width,
            image_height=calibration.height,
            clipping_range=(0.01, 100.0),
        )
        camera.load(None)
        self._configure_bev_camera(camera, calibration)
        catalog = self.object_catalog()
        top_z = max(
            (obj.bbox_max_world[2] for obj in catalog),
            default=calibration.floor_z + 3.0,
        )
        xmin, ymin, xmax, ymax = calibration.world_bounds
        camera.set_position_orientation(
            position=self._th.tensor(
                [(xmin + xmax) / 2, (ymin + ymax) / 2, top_z + 2.0]
            ),
            orientation=self._th.tensor([0.0, 0.0, 0.0, 1.0]),
        )
        camera.initialize()
        self._refresh_physics_handles_after_sensor_edit()
        for _ in range(4):
            self._og.sim.render()
        self._development_bev_sensor = camera
        self._runtime_findings["development_bev_sensor"] = {
            "backend": "persistent_pre_snapshot_vision_sensor+numpy_nearest_resample",
            "resolution": [calibration.width, calibration.height],
            "modalities": sensor_names,
        }

    def _attach_final_robot_fast_instance_annotator(self, camera: Any) -> None:
        """Attach the raw renderer AOV without TokenMap or InstanceMapping."""
        registry = self._lazy.omni.replicator.core.AnnotatorRegistry
        annotator_name = "mvwd_raw_instance_segmentation"
        if annotator_name not in registry._annotators:
            registry.register_annotator_from_aov(
                aov="InstanceSegmentationSD",
                name=annotator_name,
                output_data_type=np.uint32,
                output_channels=1,
            )
        with self._og.sim.editing_usd():
            annotator = registry.get_annotator(annotator_name)
            annotator.attach([camera._render_product])
        self._final_robot_fast_instance_annotator = annotator
        self._runtime_findings["final_robot_raw_instance_backend"] = {
            "annotator": annotator_name,
            "aov": "InstanceSegmentationSD",
            "mapping_graph_dependencies": [],
        }

    def _detach_final_robot_fast_instance_annotator(self, camera: Any) -> None:
        annotator = self._final_robot_fast_instance_annotator
        if annotator is None:
            return
        with self._og.sim.editing_usd():
            annotator.detach(camera._render_product)
        self._final_robot_fast_instance_annotator = None

    def _final_robot_renderer_labels(
        self,
    ) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
        """Map leaf renderer IDs to semantic parent paths and classes."""
        if self._syntheticdata_helpers is None:
            raise SimulatorUnavailableError(
                "omni.syntheticdata helpers are unavailable"
            )
        mappings = self._syntheticdata_helpers.get_instance_mappings()
        rows = sorted(
            mappings,
            key=lambda row: str(row["name"]).count("/"),
            reverse=True,
        )
        id_to_path: dict[str, str] = {"0": "BACKGROUND"}
        id_to_semantic: dict[str, dict[str, str]] = {
            "0": {"class": "background"}
        }
        for row in rows:
            path = str(row["name"])
            semantic_label = str(row["semanticLabel"]).strip().lower()
            if not semantic_label:
                semantic_label = "unlabelled"
            instance_ids = row["instanceIds"]
            if instance_ids is None:
                continue
            for renderer_id in instance_ids:
                key = str(int(renderer_id))
                # The deepest semantically-labelled ancestor is the closest
                # object identity for a leaf renderer instance.
                if key not in id_to_path:
                    id_to_path[key] = path
                    id_to_semantic[key] = {"class": semantic_label}
        self._runtime_findings["final_robot_renderer_mapping"] = {
            "semantic_parent_count": int(len(mappings)),
            "renderer_id_count": len(id_to_path),
            "source": "omni.syntheticdata.helpers.get_instance_mappings",
        }
        return id_to_path, id_to_semantic

    def _final_robot_segmentation_observation(
        self, camera: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build OG-compatible semantic and instance masks from renderer IDs."""
        annotator = self._final_robot_fast_instance_annotator
        if annotator is None:
            raise SimulatorUnavailableError(
                "Final-robot fast instance annotator is unavailable"
            )
        renderer_ids = None
        renderer_id_count = 0
        maximum_read_attempts = 9
        for read_attempt in range(maximum_read_attempts):
            raw = annotator.get_data(device=self._og.sim.device)
            renderer_ids = raw["data"] if isinstance(raw, dict) else raw
            if self._og.sim.device == "cpu":
                renderer_ids = camera._preprocess_cpu_obs(
                    renderer_ids, "seg_instance_id"
                )
            else:
                renderer_ids = camera._preprocess_gpu_obs(
                    renderer_ids, "seg_instance_id"
                )
            renderer_id_count = (
                int(renderer_ids.numel())
                if hasattr(renderer_ids, "numel")
                else int(np.asarray(renderer_ids).size)
            )
            if renderer_id_count:
                self._runtime_findings["final_robot_raw_instance_read_attempts"] = (
                    read_attempt + 1
                )
                break
            self._og.sim.render()
        if renderer_ids is None or not renderer_id_count:
            raise SimulatorUnavailableError(
                "Final-robot raw instance AOV remained empty after "
                f"{maximum_read_attempts} render attempts"
            )
        id_to_path, id_to_semantic = self._final_robot_renderer_labels()
        for renderer_id in self._th.unique(renderer_ids).tolist():
            key = str(int(renderer_id))
            id_to_path.setdefault(key, "UNLABELLED")
            id_to_semantic.setdefault(key, {"class": "unlabelled"})
        instance, instance_info = camera._remap_instance_segmentation(
            renderer_ids.clone(), dict(id_to_path), id=True
        )
        semantic, semantic_info = camera._remap_semantic_segmentation(
            renderer_ids.clone(), id_to_semantic
        )
        return (
            {
                "seg_semantic": semantic,
                "seg_instance": instance,
                "seg_instance_id": instance,
            },
            {
                "seg_semantic": semantic_info,
                "seg_instance": instance_info,
                "seg_instance_id": instance_info,
            },
        )

    def _get_final_robot_capture_observation(
        self, camera: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        observation, info = camera.get_obs()
        segmentation, segmentation_info = (
            self._final_robot_segmentation_observation(camera)
        )
        observation.update(segmentation)
        info.update(segmentation_info)
        return observation, info

    def _initialize_final_robot_capture_sensor(self) -> None:
        """Create one stable world-space render graph before the first snapshot."""
        if not self._using_final_robot:
            return
        configured = [
            name
            for name in self._configured_bev_sensor_names()
            if not name.startswith("seg_")
        ]
        camera_config = self.config["camera"]
        aperture = 20.995
        focal = aperture / (
            2.0 * np.tan(np.deg2rad(float(camera_config["hfov_deg"])) / 2.0)
        )
        camera = self._vision_sensor_type(
            relative_prim_path="/mvwd_persistent_final_capture_camera",
            name="mvwd_persistent_final_capture_camera",
            modalities=configured,
            image_width=int(camera_config["rgb_width"]),
            image_height=int(camera_config["rgb_height"]),
            focal_length=float(focal),
            horizontal_aperture=aperture,
            clipping_range=(camera_config["near_m"], camera_config["far_m"]),
        )
        camera.load(None)
        attached = next(
            sensor
            for sensor in self._env.robots[0].sensors.values()
            if isinstance(sensor, self._vision_sensor_type)
        )
        position, orientation = attached.get_position_orientation()
        camera.set_position_orientation(position=position, orientation=orientation)
        camera.initialize()
        self._attach_final_robot_fast_instance_annotator(camera)
        warmup_ticks = 4
        for _ in range(warmup_ticks):
            self._og.sim.render()
        self._final_robot_capture_sensor = camera
        observation, _ = self._get_final_robot_capture_observation(camera)
        required = set(configured) | {
            "seg_semantic",
            "seg_instance",
            "seg_instance_id",
        }
        absent = sorted(required - set(observation))
        if absent:
            raise SimulatorUnavailableError(
                f"Final-robot capture warmup is missing modalities: {absent}"
            )
        self._runtime_findings["final_robot_capture_graph_initialization"] = {
            "stage": "pre_snapshot",
            "backend": "persistent_world_space_vision_sensor+renderer_id_fast",
            "modalities": sorted(required),
            "warmup_render_ticks": warmup_ticks,
            "update_handles_called": False,
        }

    def _rebuild_final_robot_capture_sensor(self) -> None:
        """Rebind Replicator before refreshing furniture/system physics handles."""
        camera = self._final_robot_capture_sensor
        if camera is None:
            return
        self._detach_final_robot_fast_instance_annotator(camera)
        camera.remove()
        self._final_robot_capture_sensor = None
        rebuild_count = int(
            self._runtime_findings.get("final_robot_capture_graph_rebuild_count", 0)
        )
        self._initialize_final_robot_capture_sensor()
        self._runtime_findings["final_robot_capture_graph_rebuild_count"] = (
            rebuild_count + 1
        )

    def _configure_final_robot_ego_capture(self, camera: Any) -> None:
        """Configure the persistent sensor to match the physical mast camera."""
        camera_config = self.config["camera"]
        aperture = 20.995
        focal = aperture / (
            2.0 * np.tan(np.deg2rad(float(camera_config["hfov_deg"])) / 2.0)
        )
        with self._og.sim.editing_usd():
            usd_camera = self._lazy.pxr.UsdGeom.Camera(camera.prim)
            usd_camera.GetProjectionAttr().Set(
                self._lazy.pxr.UsdGeom.Tokens.perspective
            )
            usd_camera.GetFocalLengthAttr().Set(float(focal))
            usd_camera.GetHorizontalApertureAttr().Set(aperture)
            usd_camera.GetClippingRangeAttr().Set(
                self._lazy.pxr.Gf.Vec2f(camera_config["near_m"], camera_config["far_m"])
            )

    def _prepare_bev_sensor(
        self,
        *,
        sensor_role: str,
        floor_index: int,
        sensor_names: list[str],
        width: int,
        height: int,
    ) -> tuple[Any, bool, bool]:
        """Return a BEV sensor and whether it was created or is simulator-owned.

        Nova Carter's articulated sensor stack is not safe when a separate
        Replicator render product is added and later a serialized physics state
        is restored: SyntheticData can retain an invalid on-demand graph and
        segfault on the next render. The simulator viewer camera predates all
        scene snapshots, so reuse its render product for final-robot BEV data.
        Development runs similarly reuse one sensor initialized before the
        first snapshot without changing its graph topology.
        """
        if self._using_final_robot:
            camera = self._final_robot_capture_sensor
            if camera is None:
                raise SimulatorUnavailableError(
                    "Final-robot capture sensor was not initialized before snapshot"
                )
            graph_modalities = set(sensor_names) - {
                "seg_semantic",
                "seg_instance",
                "seg_instance_id",
            }
            missing_modalities = sorted(
                graph_modalities - set(camera.modalities)
            )
            if missing_modalities:
                raise SimulatorUnavailableError(
                    "Final-robot capture graph is missing modalities: "
                    f"{missing_modalities}"
                )
            # VisionSensor.clipping_range toggles visibility and calls render().
            # Direct USD writes avoid hidden graph ticks while robots move.
            with self._og.sim.editing_usd():
                self._lazy.pxr.UsdGeom.Camera(camera.prim).GetClippingRangeAttr().Set(
                    self._lazy.pxr.Gf.Vec2f(0.01, 100.0)
                )
            self._runtime_findings["final_robot_bev_sensor_backend"] = (
                "persistent_world_space_vision_sensor+numpy_nearest_resample"
            )
            self._runtime_findings["final_robot_bev_capture_resolution"] = [
                int(camera.image_width),
                int(camera.image_height),
            ]
            self._runtime_findings["final_robot_bev_target_resolution"] = [width, height]
            return camera, False, True
        if self._development_bev_sensor is not None:
            camera = self._development_bev_sensor
            missing_modalities = sorted(set(sensor_names) - set(camera.modalities))
            if missing_modalities:
                raise SimulatorUnavailableError(
                    "Persistent development BEV sensor is missing modalities: "
                    f"{missing_modalities}"
                )
            self._runtime_findings["development_bev_target_resolution"] = [width, height]
            return camera, False, True
        camera = self._vision_sensor_type(
            relative_prim_path=f"/mvwd_{sensor_role}_bev_camera_{floor_index}",
            name=f"mvwd_{sensor_role}_bev_camera_{floor_index}",
            modalities=sensor_names,
            image_width=width,
            image_height=height,
            clipping_range=(0.01, 100.0),
        )
        return camera, True, False

    def _release_final_robot_bev_modalities(self, camera: Any) -> None:
        """Keep the pre-snapshot final capture graph intact for the process lifetime."""
        if not self._using_final_robot or camera is not self._final_robot_capture_sensor:
            return
        self._runtime_findings["final_robot_capture_modalities_retained"] = sorted(
            camera.modalities
        )

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
            "instance": "seg_instance_id" if self._using_final_robot else "seg_instance",
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
        sensor_role = "world" if include_robots else "environment"
        hidden: list[tuple[Any, bool]] = []
        displaced_robots: list[tuple[Any, Any, Any]] = []
        camera: Any = None
        created_camera = False
        simulator_owned_camera = False
        try:
            if not include_robots:
                xmin, ymin, xmax, ymax = calibration.world_bounds
                span = max(xmax - xmin, ymax - ymin)
                for robot_index, robot in enumerate(self._env.robots):
                    position, orientation = robot.get_position_orientation()
                    displaced_robots.append(
                        (robot, position.detach().clone(), orientation.detach().clone())
                    )
                    parking_offset = span + 10.0 + 5.0 * robot_index
                    robot.set_position_orientation(
                        position=self._th.tensor(
                            [xmax + parking_offset, ymax + parking_offset, float(position[2])],
                            dtype=position.dtype,
                            device=position.device,
                        ),
                        orientation=orientation,
                    )
                self._runtime_findings["environment_bev_robot_suppression"] = (
                    "out_of_frustum_pose+restored"
                )
            ceilings = [] if not self._using_final_robot else [
                obj
                for obj in scene.objects
                if str(getattr(obj, "category", "")) in {"ceilings", "roof"}
            ]
            hidden.extend(self._set_visible(ceilings, False))
            camera, created_camera, simulator_owned_camera = self._prepare_bev_sensor(
                sensor_role=sensor_role,
                floor_index=floor_index,
                sensor_names=sensor_names,
                width=calibration.width,
                height=calibration.height,
            )
            if created_camera:
                camera.load(None)
            capture_width, capture_height = self._configure_bev_camera(
                camera, calibration
            )
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
            if created_camera:
                camera.initialize()
                self._refresh_physics_handles_after_sensor_edit()
            # Flush enough frames for out-of-frustum robot poses to reach every
            # annotator. The visible instance set was stabilized at scene load,
            # so these ticks no longer invalidate instance mapping.
            render_ticks = 4
            for _ in range(render_ticks):
                self._og.sim.render()
            self._runtime_findings["bev_render_ticks_per_capture"] = render_ticks
            if self._using_final_robot:
                observation, info = self._get_final_robot_capture_observation(
                    camera
                )
            else:
                observation, info = camera.get_obs()
            arrays: dict[str, np.ndarray] = {}
            for public_name in sensor_modalities:
                backend_name = backend_names.get(public_name, public_name)
                arrays[public_name] = self._resample_bev_observation(
                    camera,
                    backend_name,
                    observation[backend_name],
                    calibration,
                )
            if "height" in modalities:
                depth = arrays.get("depth_linear")
                if depth is None:
                    depth = self._resample_bev_observation(
                        camera,
                        "depth_linear",
                        observation["depth_linear"],
                        calibration,
                    )
                depth = np.asarray(depth).squeeze()
                arrays["height"] = np.where(np.isfinite(depth), camera_z - depth - calibration.floor_z, np.nan).astype(np.float32)
            if "occupancy" in modalities:
                height = arrays.get("height")
                if height is None:
                    depth = self._resample_bev_observation(
                        camera,
                        "depth_linear",
                        observation["depth_linear"],
                        calibration,
                    ).squeeze()
                    height = np.where(np.isfinite(depth), camera_z - depth - calibration.floor_z, np.nan)
                arrays["occupancy"] = (np.isfinite(height) & (height > 0.10)).astype(np.uint8)
            return BEVRender(
                calibration,
                arrays,
                projection,
                include_robots,
                metadata={
                    "segmentation_info": info,
                    "horizontal_aperture": 10.0
                    * float(calibration.world_bounds[2] - calibration.world_bounds[0]),
                    "vertical_aperture": 10.0
                    * float(calibration.world_bounds[3] - calibration.world_bounds[1]),
                    "capture_horizontal_aperture": 10.0 * capture_width,
                    "capture_vertical_aperture": 10.0 * capture_height,
                    "camera_z": camera_z,
                },
            )
        finally:
            for robot, position, orientation in reversed(displaced_robots):
                robot.set_position_orientation(
                    position=position,
                    orientation=orientation,
                )
            for obj, was_visible in reversed(hidden):
                obj.visible = was_visible
            if camera is not None and simulator_owned_camera:
                self._release_final_robot_bev_modalities(camera)
            if camera is not None and camera.loaded and not simulator_owned_camera:
                camera.remove()
                self._refresh_physics_handles_after_sensor_edit()
            self._og.sim.render()

    def playback_trajectories(
        self,
        trajectories: tuple[Trajectory, ...],
        floor_index: int,
        calibration: BEVCalibration,
    ) -> dict[str, Any]:
        """Render synchronized ego views and mandatory B_world with one reusable BEV sensor."""
        by_id = {trajectory.robot_id: trajectory for trajectory in trajectories}
        robots = {robot.name: robot for robot in self._env.robots}
        if by_id.keys() != robots.keys() or len(by_id) != 3:
            raise SampleRejected(
                "trajectory_robot_identity_mismatch",
                {"trajectory_ids": sorted(by_id), "robot_ids": sorted(robots)},
            )
        frame_counts = {trajectory.frames for trajectory in trajectories}
        if len(frame_counts) != 1:
            raise SampleRejected("trajectory_frame_count_mismatch")
        frames = frame_counts.pop()
        world_modalities = tuple(self.config["bev"]["world_modalities"])
        sensor_modalities = set(world_modalities) & {
            "rgb", "depth_linear", "normal", "semantic", "instance", "instance_id"
        }
        backend_names = {
            "semantic": "seg_semantic",
            "instance": "seg_instance_id" if self._using_final_robot else "seg_instance",
            "instance_id": "seg_instance_id",
        }
        sensor_names = sorted(
            {
                backend_names.get(name, name)
                for name in sensor_modalities
            }
            | ({"depth_linear"} if {"height", "occupancy"} & set(world_modalities) else set())
        )
        scene = self._require_scene()
        ceilings = [] if not self._using_final_robot else [
            obj
            for obj in scene.objects
            if str(getattr(obj, "category", "")) in {"ceilings", "roof"}
        ]
        floors = [obj for obj in scene.objects if str(getattr(obj, "category", "")) == "floors"]
        self._preflight_trajectory_contacts(by_id, robots, floors, frames)
        hidden: list[tuple[Any, bool]] = []
        world_frames: dict[str, list[np.ndarray]] = {name: [] for name in world_modalities}
        robot_frames: dict[str, dict[str, list[np.ndarray]]] = {
            robot_id: {name: [] for name in ("rgb", "depth_linear", "semantic", "instance", "normal")}
            for robot_id in sorted(robots)
        }
        actual_bases: dict[str, list[np.ndarray]] = {robot_id: [] for robot_id in robots}
        actual_cameras: dict[str, list[np.ndarray]] = {robot_id: [] for robot_id in robots}
        maximum_base_pose_error = 0.0
        maximum_camera_pose_error = 0.0
        maximum_robot_mask_error = 0.0
        minimum_robot_pixels = float("inf")
        minimum_depth_valid_ratio = 1.0
        world_bev_flip_axes: tuple[int, ...] | None = None
        world_bev_orientation_scores: dict[str, float] = {}
        world_bev_orientation_max_errors: dict[str, float] = {}
        world_bev_orientation_worst_samples: dict[str, dict[str, Any]] = {}
        robot_mask_diagnostics: list[dict[str, Any]] = []
        robot_instance_ids: dict[str, set[int]] = {
            robot_id: set() for robot_id in robots
        }
        instance_label_samples: set[str] = set()
        collision_frames: list[int] = []
        camera: Any = None
        created_camera = False
        simulator_owned_camera = False
        try:
            camera, created_camera, simulator_owned_camera = self._prepare_bev_sensor(
                sensor_role="rollout",
                floor_index=floor_index,
                sensor_names=sensor_names,
                width=calibration.width,
                height=calibration.height,
            )
            if created_camera:
                camera.load(None)
            self._configure_bev_camera(camera, calibration)
            catalog = self.object_catalog()
            top_z = max((obj.bbox_max_world[2] for obj in catalog), default=calibration.floor_z + 3.0)
            camera_z = top_z + 2.0
            xmin, ymin, xmax, ymax = calibration.world_bounds
            camera.set_position_orientation(
                position=self._th.tensor([(xmin + xmax) / 2, (ymin + ymax) / 2, camera_z]),
                orientation=self._th.tensor([0.0, 0.0, 0.0, 1.0]),
            )
            hidden = self._set_visible(ceilings, False)
            if created_camera:
                camera.initialize()
                self._refresh_physics_handles_after_sensor_edit()
            # Prime Replicator with trajectory frame 0, not the unrelated
            # placement pose that preceded rollout playback.
            for robot_id, robot in robots.items():
                planned = by_id[robot_id].base_to_world[0]
                position, orientation = self._transform_utils.mat2pose(
                    self._th.as_tensor(planned, dtype=self._th.float32)
                )
                robot.set_position_orientation(position=position, orientation=orientation)
                self._restore_final_robot_mast_mount(robot)
                robot.keep_still()
            self._og.sim.step_physics()
            for robot_id, robot in robots.items():
                planned = by_id[robot_id].base_to_world[0]
                position, orientation = self._transform_utils.mat2pose(
                    self._th.as_tensor(planned, dtype=self._th.float32)
                )
                robot.set_position_orientation(position=position, orientation=orientation)
                self._restore_final_robot_mast_mount(robot)
                robot.keep_still()
            for _ in range(2):
                self._og.sim.render()
            for frame_index in range(frames):
                for robot_id, robot in robots.items():
                    planned = by_id[robot_id].base_to_world[frame_index]
                    position, orientation = self._transform_utils.mat2pose(
                        self._th.as_tensor(planned, dtype=self._th.float32)
                    )
                    robot.set_position_orientation(position=position, orientation=orientation)
                    robot.keep_still()
                self._og.sim.step_physics()
                for robot_id, robot in robots.items():
                    planned = by_id[robot_id].base_to_world[frame_index]
                    external_pairs = self._external_robot_contact_pairs(robot, floors)
                    if external_pairs:
                        raise SampleRejected(
                            "trajectory_collision_detected",
                            {
                                "frame_index": frame_index,
                                "robot_id": robot_id,
                                "contact_pairs": external_pairs[:50],
                            },
                        )
                    position, orientation = self._transform_utils.mat2pose(
                        self._th.as_tensor(planned, dtype=self._th.float32)
                    )
                    robot.set_position_orientation(
                        position=position,
                        orientation=orientation,
                    )
                    self._restore_final_robot_mast_mount(robot)
                    robot.keep_still()
                    actual = self._pose_matrix(robot)
                    maximum_base_pose_error = max(
                        maximum_base_pose_error,
                        float(np.linalg.norm(actual[:3, 3] - planned[:3, 3])),
                        float(rotation_angle(actual, planned)),
                    )
                self._og.sim.render()
                if self._using_final_robot:
                    world_observation, world_info = (
                        self._get_final_robot_capture_observation(camera)
                    )
                else:
                    world_observation, world_info = camera.get_obs()
                frame_arrays: dict[str, np.ndarray] = {}
                for public_name in sensor_modalities:
                    backend_name = backend_names.get(public_name, public_name)
                    frame_arrays[public_name] = self._resample_bev_observation(
                        camera,
                        backend_name,
                        world_observation[backend_name],
                        calibration,
                    )
                if "height" in world_modalities:
                    depth = frame_arrays.get("depth_linear")
                    if depth is None:
                        depth = self._resample_bev_observation(
                            camera,
                            "depth_linear",
                            world_observation["depth_linear"],
                            calibration,
                        )
                    depth = np.asarray(depth).squeeze()
                    frame_arrays["height"] = np.where(
                        np.isfinite(depth), camera_z - depth - calibration.floor_z, np.nan
                    ).astype(np.float32)
                if "occupancy" in world_modalities:
                    height = frame_arrays.get("height")
                    if height is None:
                        depth = self._resample_bev_observation(
                            camera,
                            "depth_linear",
                            world_observation["depth_linear"],
                            calibration,
                        ).squeeze()
                        height = np.where(
                            np.isfinite(depth), camera_z - depth - calibration.floor_z, np.nan
                        )
                    frame_arrays["occupancy"] = (
                        np.isfinite(height) & (height > 0.10)
                    ).astype(np.uint8)
                instance_labels = np.asarray(frame_arrays["instance_id"]).squeeze()
                instance_info = world_info.get("seg_instance_id", {})
                frame_robot_instance_ids: dict[str, set[int]] = {
                    robot_id: set() for robot_id in robots
                }
                if isinstance(instance_info, dict):
                    for raw_id, label in instance_info.items():
                        label_text = str(label)
                        instance_label_samples.add(label_text)
                        try:
                            numeric_id = int(raw_id)
                        except (TypeError, ValueError):
                            continue
                        for robot_id, robot in robots.items():
                            if (
                                robot_id in label_text
                                or str(robot.prim_path) in label_text
                            ):
                                frame_robot_instance_ids[robot_id].add(numeric_id)
                                robot_instance_ids[robot_id].add(numeric_id)
                if world_bev_flip_axes is None:
                    flip_candidates: dict[str, tuple[int, ...]] = {
                        "identity": (),
                        "horizontal": (1,),
                        "vertical": (0,),
                        "rotate_180": (0, 1),
                    }
                    image_height, image_width = instance_labels.shape
                    missing_penalty = float(np.hypot(image_width, image_height))
                    for orientation_name, axes in flip_candidates.items():
                        oriented_labels = (
                            np.flip(instance_labels, axis=axes)
                            if axes
                            else instance_labels
                        )
                        score_pixels = 0.0
                        for robot_id in robots:
                            planned_uv = calibration.world_to_pixel(
                                by_id[robot_id].base_to_world[frame_index, :3, 3]
                            )
                            candidate_mask = np.isin(
                                oriented_labels,
                                tuple(frame_robot_instance_ids[robot_id]),
                            )
                            if not np.any(candidate_mask):
                                score_pixels += missing_penalty
                                candidate_error_m = (
                                    missing_penalty * calibration.meters_per_pixel
                                )
                                mask_bounds_uv = None
                            else:
                                candidate_rows, candidate_columns = np.nonzero(
                                    candidate_mask
                                )
                                candidate_distances = np.hypot(
                                    candidate_columns - float(planned_uv[0]),
                                    candidate_rows - float(planned_uv[1]),
                                )
                                candidate_error_m = float(
                                    candidate_distances.min()
                                    * calibration.meters_per_pixel
                                )
                                score_pixels += float(candidate_distances.min())
                                mask_bounds_uv = [
                                    int(candidate_columns.min()),
                                    int(candidate_rows.min()),
                                    int(candidate_columns.max()),
                                    int(candidate_rows.max()),
                                ]
                            if candidate_error_m > world_bev_orientation_max_errors.get(
                                orientation_name, -1.0
                            ):
                                world_bev_orientation_max_errors[orientation_name] = (
                                    candidate_error_m
                                )
                                world_bev_orientation_worst_samples[orientation_name] = {
                                    "frame_index": frame_index,
                                    "robot_id": robot_id,
                                    "planned_uv": np.asarray(planned_uv).tolist(),
                                    "mask_bounds_uv": mask_bounds_uv,
                                    "error_m": candidate_error_m,
                                }
                        world_bev_orientation_scores[orientation_name] = (
                            world_bev_orientation_scores.get(orientation_name, 0.0)
                            + score_pixels * calibration.meters_per_pixel
                        )
                if world_bev_flip_axes:
                    frame_arrays = {
                        name: np.flip(values, axis=world_bev_flip_axes).copy()
                        for name, values in frame_arrays.items()
                    }
                    instance_labels = np.flip(
                        instance_labels, axis=world_bev_flip_axes
                    ).copy()
                for name in world_modalities:
                    world_frames[name].append(frame_arrays[name])
                robot_mask_diagnostics.append(
                    {
                        "instance_labels": instance_labels.copy(),
                        "raw_ids": {
                            robot_id: tuple(sorted(ids))
                            for robot_id, ids in frame_robot_instance_ids.items()
                        },
                        "planned_uv": {
                            robot_id: calibration.world_to_pixel(
                                by_id[robot_id].base_to_world[frame_index, :3, 3]
                            )
                            for robot_id in robots
                        },
                    }
                )
                for robot_id in robots:
                    planned_uv = calibration.world_to_pixel(
                        by_id[robot_id].base_to_world[frame_index, :3, 3]
                    )
                    raw_ids = frame_robot_instance_ids[robot_id]
                    if not raw_ids:
                        column = int(np.rint(planned_uv[0]))
                        row = int(np.rint(planned_uv[1]))
                        if (
                            0 <= row < instance_labels.shape[0]
                            and 0 <= column < instance_labels.shape[1]
                        ):
                            center_id = int(instance_labels[row, column])
                            if center_id:
                                raw_ids.add(center_id)
                    mask = np.isin(instance_labels, tuple(raw_ids))
                    pixels = int(mask.sum())
                    minimum_robot_pixels = min(minimum_robot_pixels, pixels)
                    if pixels:
                        rows, columns = np.nonzero(mask)
                        # Nova Carter is represented by many articulated visual
                        # instances, whose visible union centroid is not its base.
                        pixel_distances = np.hypot(
                            columns - float(planned_uv[0]),
                            rows - float(planned_uv[1]),
                        )
                        nearest_mask_distance = float(
                            pixel_distances.min() * calibration.meters_per_pixel
                        )
                        maximum_robot_mask_error = max(
                            maximum_robot_mask_error,
                            nearest_mask_distance,
                        )
                self._og.sim.render()
                observations = self.robot_observations()
                for robot_id, record in observations.items():
                    planned = by_id[robot_id]
                    actual_bases[robot_id].append(record["base_to_world"])
                    actual_cameras[robot_id].append(record["camera_to_world"])
                    maximum_camera_pose_error = max(
                        maximum_camera_pose_error,
                        float(
                            np.max(
                                np.abs(
                                    record["camera_to_world"]
                                    - planned.camera_to_world[frame_index]
                                )
                            )
                        ),
                    )
                    modalities = record["modalities"]
                    rgb = np.asarray(modalities["rgb"])[..., :3]
                    robot_frames[robot_id]["rgb"].append(rgb)
                    for backend_name, public_name in (
                        ("depth_linear", "depth_linear"),
                        ("seg_semantic", "semantic"),
                        ("seg_instance", "instance"),
                        ("normal", "normal"),
                    ):
                        values = np.asarray(modalities[backend_name])
                        robot_frames[robot_id][public_name].append(values[::2, ::2])
                    depth = np.asarray(modalities["depth_linear"]).squeeze()
                    valid = (
                        np.isfinite(depth)
                        & (depth >= float(self.config["camera"]["near_m"]))
                        & (depth <= float(self.config["camera"]["far_m"]))
                    )
                    minimum_depth_valid_ratio = min(minimum_depth_valid_ratio, float(valid.mean()))
            flip_candidates = {
                "identity": (),
                "horizontal": (1,),
                "vertical": (0,),
                "rotate_180": (0, 1),
            }
            best_orientation = min(
                world_bev_orientation_scores,
                key=lambda name: (
                    world_bev_orientation_max_errors[name],
                    world_bev_orientation_scores[name],
                ),
            )
            world_bev_flip_axes = flip_candidates[best_orientation]
            if world_bev_flip_axes:
                for name in world_modalities:
                    world_frames[name] = [
                        np.flip(values, axis=world_bev_flip_axes).copy()
                        for values in world_frames[name]
                    ]
            minimum_robot_pixels = float("inf")
            maximum_robot_mask_error = 0.0
            for diagnostic in robot_mask_diagnostics:
                instance_labels = diagnostic["instance_labels"]
                if world_bev_flip_axes:
                    instance_labels = np.flip(
                        instance_labels, axis=world_bev_flip_axes
                    )
                for robot_id in robots:
                    planned_uv = diagnostic["planned_uv"][robot_id]
                    raw_ids = set(diagnostic["raw_ids"][robot_id])
                    if not raw_ids:
                        column = int(np.rint(planned_uv[0]))
                        row = int(np.rint(planned_uv[1]))
                        if (
                            0 <= row < instance_labels.shape[0]
                            and 0 <= column < instance_labels.shape[1]
                        ):
                            center_id = int(instance_labels[row, column])
                            if center_id:
                                raw_ids.add(center_id)
                    mask = np.isin(instance_labels, tuple(raw_ids))
                    pixels = int(mask.sum())
                    minimum_robot_pixels = min(minimum_robot_pixels, pixels)
                    if pixels:
                        rows, columns = np.nonzero(mask)
                        nearest_mask_distance = float(
                            np.hypot(
                                columns - float(planned_uv[0]),
                                rows - float(planned_uv[1]),
                            ).min()
                            * calibration.meters_per_pixel
                        )
                        maximum_robot_mask_error = max(
                            maximum_robot_mask_error, nearest_mask_distance
                        )
            self._runtime_findings["world_bev_axis_calibration"] = {
                "orientation": best_orientation,
                "flip_axes": list(world_bev_flip_axes),
                "scores_m": world_bev_orientation_scores,
                "maximum_errors_m": world_bev_orientation_max_errors,
                "worst_samples": world_bev_orientation_worst_samples,
                "frame_count": len(robot_mask_diagnostics),
            }
            position_tolerance = float(self.config["trajectory"]["validation_position_tolerance_m"])
            rotation_tolerance = float(self.config["trajectory"]["validation_rotation_tolerance_rad"])
            mask_tolerance = float(self.config["bev"]["robot_mask_projection_tolerance_m"])
            checks = {
                "exact_pose_playback": maximum_base_pose_error <= max(
                    position_tolerance, rotation_tolerance
                ),
                "camera_pose_matches_trajectory": maximum_camera_pose_error <= position_tolerance,
                "collision_free": not collision_frames,
                "valid_depth": minimum_depth_valid_ratio >= 0.50,
                "world_bev_each_robot_visible": minimum_robot_pixels >= 4,
                "world_bev_robot_mask_projection": maximum_robot_mask_error <= mask_tolerance,
            }
            if not all(checks.values()):
                raise SampleRejected(
                    "rollout_qa_failed",
                    {
                        "checks": checks,
                        "maximum_base_pose_error": maximum_base_pose_error,
                        "maximum_camera_pose_error": maximum_camera_pose_error,
                        "collision_frames": sorted(set(collision_frames)),
                        "minimum_depth_valid_ratio": minimum_depth_valid_ratio,
                        "minimum_robot_pixels": minimum_robot_pixels,
                        "maximum_robot_mask_error_m": maximum_robot_mask_error,
                        "world_bev_axis_calibration": self._runtime_findings[
                            "world_bev_axis_calibration"
                        ],
                        "robot_instance_ids": {
                            robot_id: sorted(raw_ids)
                            for robot_id, raw_ids in robot_instance_ids.items()
                        },
                        "instance_label_samples": sorted(instance_label_samples)[:50],
                    },
                )
            actual_trajectories = tuple(
                Trajectory(
                    robot_id=robot_id,
                    fps=by_id[robot_id].fps,
                    base_to_world=np.stack(actual_bases[robot_id]),
                    camera_to_world=np.stack(actual_cameras[robot_id]),
                )
                for robot_id in sorted(robots)
            )
            return {
                "world_bev": {
                    name: np.stack(values) for name, values in world_frames.items()
                },
                "robot_views": {
                    robot_id: {
                        name: np.stack(values) for name, values in modalities.items()
                    }
                    for robot_id, modalities in robot_frames.items()
                },
                "actual_trajectories": actual_trajectories,
                "checks": checks,
                "metrics": {
                    "maximum_base_pose_error": maximum_base_pose_error,
                    "maximum_camera_pose_error": maximum_camera_pose_error,
                    "minimum_depth_valid_ratio": minimum_depth_valid_ratio,
                    "minimum_robot_pixels": int(minimum_robot_pixels),
                    "maximum_robot_mask_error_m": maximum_robot_mask_error,
                },
            }
        finally:
            for obj, was_visible in reversed(hidden):
                obj.visible = was_visible
            if camera is not None and simulator_owned_camera:
                self._release_final_robot_bev_modalities(camera)
            if camera is not None and camera.loaded and not simulator_owned_camera:
                camera.remove()
                self._refresh_physics_handles_after_sensor_edit()
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

