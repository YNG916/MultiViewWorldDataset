from __future__ import annotations

import uuid
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from multi_view_world_dataset.adapters.base import BEVRender, BaseSimulatorAdapter
from multi_view_world_dataset.assets import (
    ROBOT_APPEARANCE_VARIANTS,
    materialize_mobile_sensor_robot,
)
from multi_view_world_dataset.cameras.transforms import rotation_angle, validate_transform
from multi_view_world_dataset.errors import GeometryError, SampleRejected, SimulatorUnavailableError
from multi_view_world_dataset.rendering.bev import BEVCalibration
from multi_view_world_dataset.rendering.labels import remap_public_labels
from multi_view_world_dataset.rendering.modalities import canonicalize_public_modality
from multi_view_world_dataset.sampling.placement import (
    select_consensus_local_headings,
    select_local_traversable_heading,
    soft_anchor_candidate_order,
)
from multi_view_world_dataset.sampling.interventions import (
    choose_intervention_type,
    eligible_intervention_targets,
    propose_articulation,
    propose_rigid_relocation,
    propose_state_change,
)
from multi_view_world_dataset.sampling.navigation import NavigationContext
from multi_view_world_dataset.sampling.diversity import (
    choose_weighted_label,
    complementary_hybrid_trajectory_sets,
    formation_degenerate,
    joint_trajectory_metrics,
    regime_trajectory_soft_score,
)
from multi_view_world_dataset.sampling.splits import infer_scene_family
from multi_view_world_dataset.sampling.trajectories import (
    lane_preserving_guides,
    sample_geodesic_robot_trajectory_pool,
    sample_geodesic_trajectory_set,
    trajectory_kinematic_metrics,
)
from multi_view_world_dataset.schema.records import (
    BaseSceneRecord, InterventionEvent, InterventionType, ObjectState, Trajectory,
)
from multi_view_world_dataset.utils.runtime import RuntimePaths, installed_versions


def _resize_nearest(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-resize the first two axes without adding an image dependency."""
    array = np.asarray(values)
    if array.shape[:2] == shape:
        return array
    rows = np.rint(np.linspace(0, array.shape[0] - 1, shape[0])).astype(np.int64)
    columns = np.rint(np.linspace(0, array.shape[1] - 1, shape[1])).astype(np.int64)
    return array[rows[:, None], columns[None, :]]


def _points_inside_floor_support(
    points_xy: np.ndarray,
    support_bounds_xy: np.ndarray,
    *,
    tolerance_m: float = 0.0,
) -> np.ndarray:
    """Return whether XY points lie over at least one physical floor AABB."""
    points = np.asarray(points_xy, dtype=np.float64)
    bounds = np.asarray(support_bounds_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape [N,2]")
    if bounds.size == 0:
        return np.zeros(len(points), dtype=bool)
    if bounds.ndim != 2 or bounds.shape[1] != 4:
        raise ValueError("support_bounds_xy must have shape [M,4]")
    tolerance = float(tolerance_m)
    if tolerance < 0.0:
        raise ValueError("tolerance_m must be non-negative")
    return np.any(
        (points[:, None, 0] >= bounds[None, :, 0] - tolerance)
        & (points[:, None, 0] <= bounds[None, :, 2] + tolerance)
        & (points[:, None, 1] >= bounds[None, :, 1] - tolerance)
        & (points[:, None, 1] <= bounds[None, :, 3] + tolerance),
        axis=1,
    )


def _restore_world_to_map_batch_order(
    mapped: np.ndarray,
    first_single: np.ndarray,
    last_single: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Undo the OG 3.9.2 batch-axis flip while tolerating future fixed APIs."""
    values = np.asarray(mapped)
    first = np.asarray(first_single)
    last = np.asarray(last_single)
    if values.ndim != 2 or len(values) < 2:
        return values, False
    forward = np.array_equal(values[0], first) and np.array_equal(values[-1], last)
    reversed_batch = (
        np.array_equal(values[-1], first) and np.array_equal(values[0], last)
    )
    if forward:
        return values, False
    if reversed_batch:
        return values[::-1].copy(), True
    raise ValueError(
        "world_to_map batch output matches neither forward nor reversed input order"
    )


def _continuous_edge_magnitude(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2:
        array = array[..., None]
    elif array.ndim != 3:
        raise ValueError(f"Expected 2D or channel-last image, got {array.shape}")
    squared = np.zeros(array.shape[:2], dtype=np.float64)
    for channel in range(array.shape[-1]):
        plane = array[..., channel]
        finite = np.isfinite(plane)
        fill = float(np.median(plane[finite])) if np.any(finite) else 0.0
        plane = np.where(finite, plane, fill)
        gradient_y, gradient_x = np.gradient(plane)
        squared += gradient_x * gradient_x + gradient_y * gradient_y
    return np.sqrt(squared)


def _label_edge_mask(values: np.ndarray) -> np.ndarray:
    labels = np.asarray(values).squeeze()
    if labels.ndim != 2:
        raise ValueError(f"Expected a 2D label image, got {labels.shape}")
    edges = np.zeros(labels.shape, dtype=bool)
    horizontal = labels[:, 1:] != labels[:, :-1]
    vertical = labels[1:, :] != labels[:-1, :]
    edges[:, 1:] |= horizontal
    edges[:, :-1] |= horizontal
    edges[1:, :] |= vertical
    edges[:-1, :] |= vertical
    return edges.astype(np.float64)


def _edge_correlation(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64).ravel()
    right = np.asarray(second, dtype=np.float64).ravel()
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator > 0.0 else 0.0


def robot_multimodal_alignment_metrics(
    robot_frames: dict[str, dict[str, Any]],
    *,
    keyframe_count: int,
) -> dict[str, Any]:
    """Check that every image modality belongs to the same robot viewpoint."""
    robot_ids = sorted(robot_frames)
    if not robot_ids:
        raise ValueError("robot_frames must not be empty")
    frame_count = len(robot_frames[robot_ids[0]]["depth_linear"])
    if frame_count < 1:
        raise ValueError("robot_frames must contain at least one frame")
    keyframes = np.unique(
        np.rint(np.linspace(0, frame_count - 1, max(1, keyframe_count))).astype(int)
    )
    edge_builders: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "rgb": lambda value: _continuous_edge_magnitude(np.asarray(value)[..., :3]),
        "semantic": _label_edge_mask,
        "instance": _label_edge_mask,
        "normal": lambda value: _continuous_edge_magnitude(np.asarray(value)[..., :3]),
    }
    results: dict[str, Any] = {}
    for modality, edge_builder in edge_builders.items():
        own_best_count = 0
        own_scores: list[float] = []
        margins: list[float] = []
        assignments: dict[str, dict[str, str]] = {}
        for frame_index in keyframes:
            depth_edges = {}
            for robot_id in robot_ids:
                depth = np.asarray(
                    robot_frames[robot_id]["depth_linear"][int(frame_index)]
                ).squeeze()
                depth_edges[robot_id] = _continuous_edge_magnitude(depth)
            target_shape = next(iter(depth_edges.values())).shape
            frame_assignments: dict[str, str] = {}
            for robot_id in robot_ids:
                value = np.asarray(robot_frames[robot_id][modality][int(frame_index)])
                value = _resize_nearest(value, target_shape)
                target_edge = edge_builder(value)
                scores = [
                    _edge_correlation(target_edge, depth_edges[candidate])
                    for candidate in robot_ids
                ]
                own_index = robot_ids.index(robot_id)
                own_score = scores[own_index]
                best_index = int(np.argmax(scores))
                best_score = scores[best_index]
                other_scores = [
                    score for index, score in enumerate(scores) if index != own_index
                ]
                best_other = max(other_scores) if other_scores else own_score
                own_best_count += int(own_score >= best_score - 1.0e-12)
                own_scores.append(own_score)
                margins.append(own_score - best_other)
                frame_assignments[robot_id] = robot_ids[best_index]
            assignments[str(int(frame_index))] = frame_assignments
        comparison_count = len(keyframes) * len(robot_ids)
        results[modality] = {
            "own_view_best_count": own_best_count,
            "comparison_count": comparison_count,
            "own_view_best_fraction": own_best_count / comparison_count,
            "mean_own_edge_correlation": float(np.mean(own_scores)),
            "mean_own_minus_best_other_correlation": float(np.mean(margins)),
            "best_view_assignments": assignments,
        }
    return {
        "keyframe_indices": [int(index) for index in keyframes],
        "modalities": results,
    }


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
        self._final_robot_renderer_label_cache: (
            tuple[dict[str, str], dict[str, dict[str, str]]] | None
        ) = None
        self._public_label_catalog_cache: tuple[ObjectState, ...] | None = None
        self._syntheticdata_helpers: Any = None
        self._started = False
        self._runtime_findings: dict[str, Any] = {}
        self._canonical_floor_bounds: dict[int, tuple[float, float, float, float]] = {}
        self._navigation_contexts: dict[int, NavigationContext] = {}
        self._navigation_configuration_token: str | None = None

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
        try:
            self._og.app.close()
        except SystemExit as error:
            # SimulationApp.close() raises SystemExit in this runtime. Let the
            # CLI / generator return normally (or propagate the active sample
            # failure) instead of silently turning every failure into exit 0.
            self._runtime_findings["simulation_app_close_system_exit"] = str(error)
        finally:
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

    def _apply_final_robot_appearance_variants(self) -> None:
        """Bind one canonical accent material to each robot's visual-only shell."""
        if not self._using_final_robot:
            return
        accent_relative_paths = (
            "chassis_link/mvwd_sensor_rig/tower_accent_band",
            "chassis_link/mvwd_sensor_rig/chassis_identity_deck",
            "chassis_link/mvwd_sensor_rig/chassis_identity_left",
            "chassis_link/mvwd_sensor_rig/chassis_identity_right",
            "mast_carriage/sliding_sleeve_accent",
            "mast_carriage/sensor_head_accent",
            "mast_carriage/sensor_head_top_cap",
            "mast_carriage/front_direction_marker",
        )
        stage = self._og.sim.stage
        applied: dict[str, dict[str, Any]] = {}
        with self._og.sim.editing_usd():
            for robot in sorted(self._env.robots, key=lambda item: item.name):
                if robot.name not in ROBOT_APPEARANCE_VARIANTS:
                    raise SimulatorUnavailableError(
                        f"No canonical appearance variant is defined for {robot.name}"
                    )
                specification = ROBOT_APPEARANCE_VARIANTS[robot.name]
                robot_root = str(robot.prim_path)
                material_path = (
                    f"{robot_root}/Looks/{specification['material_prim']}"
                )
                material_prim = stage.GetPrimAtPath(material_path)
                if not material_prim.IsValid():
                    raise SimulatorUnavailableError(
                        f"Robot appearance material is missing: {material_path}"
                    )
                material = self._lazy.pxr.UsdShade.Material(material_prim)
                bound_paths = []
                for relative_path in accent_relative_paths:
                    prim_path = f"{robot_root}/{relative_path}"
                    prim = stage.GetPrimAtPath(prim_path)
                    if not prim.IsValid():
                        raise SimulatorUnavailableError(
                            f"Robot appearance prim is missing: {prim_path}"
                        )
                    binding = self._lazy.pxr.UsdShade.MaterialBindingAPI.Apply(prim)
                    binding.Bind(
                        material,
                        bindingStrength=(
                            self._lazy.pxr.UsdShade.Tokens.strongerThanDescendants
                        ),
                    )
                    gprim = self._lazy.pxr.UsdGeom.Gprim(prim)
                    gprim.CreateDisplayColorAttr().Set([
                        self._lazy.pxr.Gf.Vec3f(
                            *[float(value) for value in specification["rgb"]]
                        )
                    ])
                    bound_paths.append(prim_path)
                applied[robot.name] = {
                    "variant": str(specification["variant"]),
                    "display_name": str(specification["display_name"]),
                    "material_path": material_path,
                    "bound_visual_prims": bound_paths,
                    "collision_geometry_changed": False,
                }
        self._runtime_findings["final_robot_appearance_variants"] = applied


    def load_scene(self, scene_id: str, *, robot_count: int = 0, development_robot: str = "turtlebot") -> None:
        self._require_started()
        if self._env is not None:
            self._og.clear()
        self._development_bev_sensor = None
        self._final_robot_capture_sensor = None
        self._final_robot_renderer_label_cache = None
        self._public_label_catalog_cache = None
        self._development_camera_mounts.clear()
        self._relation_cache = None
        self._canonical_floor_bounds.clear()
        self._navigation_contexts.clear()
        self._navigation_configuration_token = None
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
        if self._using_final_robot:
            self._apply_final_robot_appearance_variants()
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

    def _world_to_map_preserving_batch(self, map_object: Any, world_xy: Any) -> np.ndarray:
        """Call OG world_to_map without its 3.9.2 reversal of the batch axis."""
        points = self._th.as_tensor(world_xy, dtype=self._th.float32)
        mapped = self._native_value(map_object.world_to_map(points)).astype(int)
        if points.ndim == 2 and len(points) >= 2:
            first = self._native_value(map_object.world_to_map(points[0])).astype(int)
            last = self._native_value(map_object.world_to_map(points[-1])).astype(int)
            mapped, repaired = _restore_world_to_map_batch_order(mapped, first, last)
            if repaired:
                self._runtime_findings["world_to_map_batch_order_workaround"] = {
                    "applied": True,
                    "reason": "OmniGibson 3.9.2 world_to_map reverses [N,2] batch order",
                }
        return mapped

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
                # Open is joint-backed in BEHAVIOR assets and therefore belongs
                # to ARTICULATION, never the generic STATE_CHANGE taxonomy.
                "ToggledOn", "Cooked", "Burnt", "Frozen", "Heated", "OnFire"
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
                        state_specs.append((
                            "OnFloor" if reference.category == "floors" else "OnTop",
                            self._on_top_type,
                        ))
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
        """Return a complete relation snapshot for the current physical state.

        Relations are dynamic object states. Reusing a scene-level cache after
        relocation, articulation, settling, or snapshot restore silently
        attaches stale edges, so every source-of-truth catalog is recomputed.
        ``_relation_cache`` remains only the latest diagnostic/candidate set.
        """
        catalog = self.object_catalog()
        self._relation_cache = self.relation_candidates(catalog)
        return self._attach_relations(catalog, self._relation_cache)

    def _relation_state_type(self, predicate: str) -> Any:
        if predicate in {"OnTop", "OnFloor"}:
            return self._on_top_type
        if predicate == "Inside":
            return self._inside_type
        raise ValueError(f"Unsupported rigid relation predicate: {predicate}")

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
        """Randomize a bounded fraction of distinct movable relation targets."""
        from multi_view_world_dataset.sampling.configurations import exact_state_hash

        original_snapshot = self.dump_snapshot()
        baseline = self.object_catalog_with_relations()
        if self._relation_cache is None:
            self._relation_cache = self.relation_candidates(self.object_catalog())
        eligible_ids = sorted({item["target_instance_id"] for item in self._relation_cache})
        policy = self.config["configuration_sampling"]
        requested = int(np.ceil(float(policy["movable_fraction"]) * len(eligible_ids)))
        requested = max(int(policy["minimum_changed_objects"]), requested)
        requested = min(int(policy["maximum_changed_objects"]), requested, len(eligible_ids))
        if requested < int(policy["minimum_changed_objects"]):
            raise SampleRejected(
                "insufficient_multi_object_configuration_targets",
                {"eligible_target_count": len(eligible_ids), "required": int(policy["minimum_changed_objects"])},
            )
        excluded: set[str] = set()
        changes: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        baseline_by_id = {obj.instance_id: obj for obj in baseline}
        category_counts: dict[str, int] = {}
        room_counts: dict[str, int] = {}
        predicate_counts: dict[str, int] = {}
        for change_index in range(requested):
            available = [relation for relation in self._relation_cache if relation["target_instance_id"] not in excluded]
            novelty = []
            for relation in available:
                obj = baseline_by_id[relation["target_instance_id"]]
                size = float(np.prod(np.asarray(obj.bbox_max_world) - np.asarray(obj.bbox_min_world)))
                novelty.append((
                    category_counts.get(obj.category, 0)
                    + room_counts.get(obj.room_id or "unknown", 0)
                    + predicate_counts.get(relation["predicate"], 0),
                    category_counts.get(f"size:{int(np.floor(np.log10(max(size, 1e-6))))}", 0),
                    relation["target_instance_id"],
                ))
            preferred_target = min(novelty)[2] if novelty else None
            try:
                result = self._randomize_one_relation_preserving_configuration(
                    seed + 104729 * (change_index + 1),
                    excluded_target_ids=tuple(sorted(excluded)),
                    preferred_target_ids=(() if preferred_target is None else (preferred_target,)),
                )
            except SampleRejected:
                break
            target_id = str(result["relation"]["target_instance_id"])
            excluded.add(target_id)
            changed_obj = baseline_by_id[target_id]
            size = float(np.prod(np.asarray(changed_obj.bbox_max_world) - np.asarray(changed_obj.bbox_min_world)))
            size_bin = f"size:{int(np.floor(np.log10(max(size, 1e-6))))}"
            category_counts[changed_obj.category] = category_counts.get(changed_obj.category, 0) + 1
            room_key = changed_obj.room_id or "unknown"
            room_counts[room_key] = room_counts.get(room_key, 0) + 1
            predicate = str(result["relation"]["predicate"])
            predicate_counts[predicate] = predicate_counts.get(predicate, 0) + 1
            category_counts[size_bin] = category_counts.get(size_bin, 0) + 1
            changes.append({
                **result["relation"],
                "translation_m": result["translation_m"],
                "rotation_deg": result["rotation_deg"],
            })
            results.append(result)
        if len(changes) < int(policy["minimum_changed_objects"]):
            self.load_snapshot(original_snapshot)
            raise SampleRejected(
                "multi_object_configuration_sampling_failed",
                {"accepted_changes": len(changes), "requested_changes": requested},
            )
        accepted_snapshot = self.dump_snapshot()
        catalog = self.object_catalog_with_relations()
        self.load_snapshot(accepted_snapshot)
        restored = self.object_catalog_with_relations()
        restored_native = self._native_objects_by_path()
        relations_preserved = all(
            bool(
                restored_native[relation["target_native_path"]].states[
                    self._relation_state_type(str(relation["predicate"]))
                ].get_value(restored_native[relation["reference_native_path"]])
            )
            for relation in changes
        )
        maximum_restore_error, discrete_equal = self._catalog_restore_metrics(catalog, restored)
        if (
            maximum_restore_error > float(self.config["generation"]["snapshot_restore_tolerance"])
            or not discrete_equal or not relations_preserved
        ):
            self.load_snapshot(original_snapshot)
            raise SampleRejected(
                "multi_object_configuration_snapshot_restore_mismatch",
                {"maximum_restore_error": maximum_restore_error, "discrete_equal": discrete_equal, "relations_preserved": relations_preserved},
            )
        return {
            "catalog": restored,
            "snapshot": accepted_snapshot,
            "exact_state_hash": exact_state_hash(restored, decimals=int(self.config["generation"]["exact_hash_decimals"])),
            "baseline_exact_state_hash": exact_state_hash(baseline, decimals=int(self.config["generation"]["exact_hash_decimals"])),
            "changed_instance_ids": sorted(excluded),
            "changes": changes,
            "changed_object_count": len(changes),
            "requested_changed_object_count": requested,
            "stratification": {
                "categories": category_counts,
                "rooms": room_counts,
                "relations": predicate_counts,
            },
            "accepted_attempt": sum(int(item["accepted_attempt"]) for item in results),
            "checks": {"minimum_changed_objects": True, "snapshot_restored": True},
            "maximum_snapshot_restore_error": maximum_restore_error,
            "free_traversable_candidates": results[-1]["free_traversable_candidates"],
            "intervention_target_count": results[-1]["intervention_target_count"],
            "translation_m": float(sum(item["translation_m"] for item in results)),
            "rotation_deg": float(sum(item["rotation_deg"] for item in results)),
        }

    def _randomize_one_relation_preserving_configuration(
        self, seed: int, *, excluded_target_ids: tuple[str, ...] = (),
        preferred_target_ids: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Create one native-validated relation-preserving object change."""
        from multi_view_world_dataset.sampling.configurations import exact_state_hash

        baseline_snapshot = self.dump_snapshot()
        baseline_raw_catalog = self.object_catalog()
        if self._relation_cache is None:
            self._relation_cache = self.relation_candidates(baseline_raw_catalog)
        relations = tuple(
            relation for relation in self._relation_cache
            if relation["target_instance_id"] not in excluded_target_ids
        )
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
                bool(preferred_target_ids) and relations[index]["target_instance_id"] not in preferred_target_ids,
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
            state_type = self._relation_state_type(str(relation["predicate"]))
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
                            relation["predicate"] == "OnFloor"
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
            catalog = self.object_catalog_with_relations()
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
            restored_catalog = self.object_catalog_with_relations()
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
        visible_target_ids: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Apply and verify exactly one v1 intervention in the currently loaded W0."""
        baseline_snapshot = self.dump_snapshot()
        baseline_catalog = self.object_catalog_with_relations()
        baseline_by_id = {obj.instance_id: obj for obj in baseline_catalog}
        native_by_path = self._native_objects_by_path()
        rng = np.random.default_rng(seed)
        weights = self.config["intervention"]["type_weights"]
        preferred = forced_type or choose_intervention_type(weights, rng)
        # The accepted type is selected once per episode. Resampling may change
        # the target/parameters, never silently fall back to another taxonomy.
        type_order = [preferred]
        maximum_attempts = int(self.config["intervention"]["maximum_attempts"])
        failures: list[dict[str, Any]] = []
        attempt = 0
        for intervention_type in type_order:
            candidates = [
                obj
                for obj in eligible_intervention_targets(baseline_catalog, intervention_type)
                if obj.instance_id not in excluded_target_ids
                and (not visible_target_ids or obj.instance_id in visible_target_ids)
            ]
            if intervention_type is InterventionType.RIGID_RELOCATION:
                candidates = [
                    obj
                    for obj in candidates
                    if any(
                        relation.get("predicate") in {"OnFloor", "OnTop", "Inside"}
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
                            if relation.get("predicate") in {
                                "OnFloor", "OnTop", "Inside"
                            }
                        )
                        reference = baseline_by_id[str(relation["reference_instance_id"])]
                        native_reference = native_by_path[reference.native_path]
                        relation_state_type = self._relation_state_type(
                            str(relation["predicate"])
                        )
                        if relation["predicate"] == "OnFloor":
                            delta_xy = np.asarray(
                                event.parameters["translation_xy_m"], dtype=np.float64
                            )
                            yaw_delta = float(event.parameters["yaw_delta_rad"])
                            candidate_transform = target.object_to_world.copy()
                            candidate_transform[:2, 3] += delta_xy
                            cosine, sine = np.cos(yaw_delta), np.sin(yaw_delta)
                            yaw_rotation = np.asarray([
                                [cosine, -sine, 0.0],
                                [sine, cosine, 0.0],
                                [0.0, 0.0, 1.0],
                            ])
                            candidate_transform[:3, :3] = (
                                yaw_rotation @ candidate_transform[:3, :3]
                            )
                            position, orientation = self._transform_utils.mat2pose(
                                self._th.as_tensor(
                                    candidate_transform, dtype=self._th.float32
                                )
                            )
                            native_target.set_position_orientation(
                                position=position, orientation=orientation
                            )
                        else:
                            sampled = bool(
                                native_target.states[relation_state_type].set_value(
                                    native_reference,
                                    True,
                                    reset_before_sampling=True,
                                    use_trav_map=False,
                                )
                            )
                            if not sampled:
                                failures.append({
                                    "reason": "rigid_relation_resampling_failed",
                                    "predicate": relation["predicate"],
                                    "target": target.instance_id,
                                })
                                continue
                        for native_object in list(self._require_scene().objects):
                            if hasattr(native_object, "keep_still"):
                                native_object.keep_still()
                        self._og.sim.step_physics()
                        relation_valid = bool(
                            native_target.states[relation_state_type].get_value(
                                native_reference
                            )
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
                        validated_transform = self._pose_matrix(native_target)
                        realized_xy = (
                            validated_transform[:2, 3]
                            - target.object_to_world[:2, 3]
                        )
                        realized_yaw = float(np.arctan2(
                            validated_transform[1, 0], validated_transform[0, 0]
                        ) - np.arctan2(
                            target.object_to_world[1, 0], target.object_to_world[0, 0]
                        ))
                        realized_yaw = float(
                            (realized_yaw + np.pi) % (2.0 * np.pi) - np.pi
                        )
                        event = replace(event, parameters={
                            **event.parameters,
                            "translation_xy_m": realized_xy.tolist(),
                            "yaw_delta_rad": realized_yaw,
                            "preserved_relation": dict(relation),
                            "placement_backend": (
                                "direct_floor_pose"
                                if relation["predicate"] == "OnFloor"
                                else "omnigibson_native_relation_sampler"
                            ),
                        })
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
        self._relation_cache = None

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

    def _floor_supported_traversability_source(self, floor_index: int) -> Any:
        """Mask OG traversability to locations backed by physical floor geometry."""
        trav_map = self._require_scene().trav_map
        source = self._th.clone(trav_map.floor_map[floor_index])
        pixels = self._th.stack(self._th.where(source == 255), dim=1)
        floor_id = f"floor_{floor_index:02d}"
        supports = [
            obj
            for obj in self.object_catalog()
            if obj.floor_id == floor_id
            and str(obj.category).lower() in {"floor", "floors"}
        ]
        if not supports or not len(pixels):
            self._runtime_findings["floor_support_filter"] = {
                "available": False,
                "floor_id": floor_id,
                "support_object_count": len(supports),
                "candidate_count": int(len(pixels)),
            }
            if not supports:
                raise SampleRejected(
                    "floor_support_geometry_unavailable",
                    {"floor_id": floor_id},
                )
            return source
        world_xy = self._native_value(trav_map.map_to_world(pixels)).astype(
            np.float64
        )
        support_bounds = np.asarray(
            [
                [
                    obj.bbox_min_world[0],
                    obj.bbox_min_world[1],
                    obj.bbox_max_world[0],
                    obj.bbox_max_world[1],
                ]
                for obj in supports
            ],
            dtype=np.float64,
        )
        tolerance_m = float(
            self.config["placement"]["floor_support_aabb_tolerance_m"]
        )
        supported = _points_inside_floor_support(
            world_xy,
            support_bounds,
            tolerance_m=tolerance_m,
        )
        unsupported_pixels = pixels[
            self._th.as_tensor(~supported, device=pixels.device)
        ]
        if len(unsupported_pixels):
            source[unsupported_pixels[:, 0], unsupported_pixels[:, 1]] = 0
        self._runtime_findings["floor_support_filter"] = {
            "available": True,
            "floor_id": floor_id,
            "support_object_count": len(supports),
            "tolerance_m": tolerance_m,
            "source_candidate_count": int(len(pixels)),
            "supported_candidate_count": int(np.count_nonzero(supported)),
            "removed_candidate_count": int(np.count_nonzero(~supported)),
        }
        return source

    def _robot_eroded_traversability(self, floor_index: int, robot: Any) -> Any:
        """Use the installed OG 3.9.2 robot-aware traversability erosion."""
        trav_map = self._require_scene().trav_map
        source = self._floor_supported_traversability_source(floor_index)
        chassis_extent = self._native_value(
            robot.reset_joint_pos_aabb_extent[:2]
        ).astype(np.float64)
        footprint_radius_m = float(np.linalg.norm(chassis_extent) / 2.0)
        installed_radius_m = footprint_radius_m + 0.2
        radius_pixels = int(np.ceil(
            installed_radius_m / float(trav_map.map_resolution)
        ))
        eroded = trav_map._erode_trav_map(
            self._th.clone(source), robot=robot
        )
        self._runtime_findings["traversability_clearance"] = {
            "api": "trav_map._erode_trav_map(robot=robot)",
            "chassis_extent_xy_m": chassis_extent.tolist(),
            "footprint_radius_m": footprint_radius_m,
            "installed_radius_m": installed_radius_m,
            "installed_cv2_kernel_width_pixels": radius_pixels,
            "map_resolution_m": float(trav_map.map_resolution),
        }
        return eroded

    def _external_robot_contact_pairs(
        self,
        robot: Any,
        support_surfaces: list[Any],
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
            *(str(surface.prim_path) for surface in support_surfaces),
        )
        return [
            [query_path, other_path]
            for query_path, other_path in contact_pairs
            if not any(
                other_path == prefix or other_path.startswith(prefix + "/")
                for prefix in ignored_prefixes
            )
        ]

    def _robot_support_surfaces(self) -> list[Any]:
        """Return scene objects whose contact physically supports robot motion."""
        return [
            obj
            for obj in self._require_scene().objects
            if str(getattr(obj, "category", "")).lower()
            in {"floor", "floors", "lawn"}
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

    def calibrated_floor_bounds(
        self, floor_index: int, meters_per_pixel: float, margin_m: float
    ) -> BEVCalibration:
        """Return one static scene/floor extent shared by every BEV product."""
        if floor_index not in self._canonical_floor_bounds:
            floor_id = f"floor_{floor_index:02d}"
            catalog = self.object_catalog()
            structural = [
                obj for obj in catalog if obj.floor_id == floor_id and obj.structural
            ]
            world_xy, _, _, _ = self._trajectory_traversability(
                floor_index, self._env.robots[0]
            )
            xmin, ymin = np.min(world_xy, axis=0)
            xmax, ymax = np.max(world_xy, axis=0)
            if structural:
                xmin = min(float(xmin), min(obj.bbox_min_world[0] for obj in structural))
                ymin = min(float(ymin), min(obj.bbox_min_world[1] for obj in structural))
                xmax = max(float(xmax), max(obj.bbox_max_world[0] for obj in structural))
                ymax = max(float(ymax), max(obj.bbox_max_world[1] for obj in structural))
            canonical_resolution = max(
                float(self.config["bev"]["environment_meters_per_pixel"]),
                float(self.config["bev"]["world_meters_per_pixel"]),
            )
            bounds = (
                np.floor((xmin - margin_m) / canonical_resolution) * canonical_resolution,
                np.floor((ymin - margin_m) / canonical_resolution) * canonical_resolution,
                np.ceil((xmax + margin_m) / canonical_resolution) * canonical_resolution,
                np.ceil((ymax + margin_m) / canonical_resolution) * canonical_resolution,
            )
            self._canonical_floor_bounds[floor_index] = tuple(map(float, bounds))
        return BEVCalibration(
            f"floor_{floor_index:02d}", self._floor_heights()[floor_index],
            meters_per_pixel, self._canonical_floor_bounds[floor_index],
        )

    def traversability_bev(
        self, floor_index: int, calibration: BEVCalibration
    ) -> np.ndarray:
        """Backward-compatible alias for any-yaw footprint navigability."""
        return self.traversability_bev_layers(floor_index, calibration)[
            "any_yaw_navigable"
        ]

    def traversability_bev_layers(
        self, floor_index: int, calibration: BEVCalibration
    ) -> dict[str, np.ndarray]:
        """Rasterize point support and orientation-aware footprint semantics."""
        from multi_view_world_dataset.adapters.navigation import (
            _footprint_offsets,
            _point_free_mask,
            _robot_footprint,
        )
        from multi_view_world_dataset.sampling.navigation import oriented_safe_masks

        point_free = _point_free_mask(self, floor_index)
        footprint = _robot_footprint(self)
        safe_masks = oriented_safe_masks(
            point_free, _footprint_offsets(self, footprint, floor_index)
        )
        native_layers = {
            "point_traversability": point_free.astype(np.float32),
            "any_yaw_navigable": np.any(safe_masks, axis=0).astype(np.float32),
            "yaw_freedom": np.mean(safe_masks, axis=0, dtype=np.float32),
        }
        rows, columns = np.indices((calibration.height, calibration.width))
        pixels = np.column_stack((columns.ravel(), rows.ravel()))
        world = calibration.pixel_to_world(pixels)[:, :2]
        trav_map = self._require_scene().trav_map
        mapped = self._world_to_map_preserving_batch(trav_map, world)
        shape = point_free.shape
        valid = (
            (mapped[:, 0] >= 0) & (mapped[:, 0] < shape[0])
            & (mapped[:, 1] >= 0) & (mapped[:, 1] < shape[1])
        )
        result: dict[str, np.ndarray] = {}
        for name, native in native_layers.items():
            output = np.zeros(len(mapped), dtype=np.float32)
            output[valid] = native[mapped[valid, 0], mapped[valid, 1]]
            result[name] = output.reshape(
                calibration.height, calibration.width
            )
        return result

    def _observation_region_labels(
        self, floor_index: int, world_xy: np.ndarray
    ) -> np.ndarray:
        """Label traversable samples with OG room instances or spatial fallback regions."""
        scene = self._require_scene()
        labels = np.full(len(world_xy), "", dtype=object)
        if floor_index == 0:
            try:
                seg_map = scene.seg_map
                pixels = self._world_to_map_preserving_batch(seg_map, world_xy)
                room_map = self._native_value(seg_map.room_ins_map)
                mapping = dict(seg_map.room_ins_id_to_ins_name)
                inside = (
                    (pixels[:, 0] >= 0) & (pixels[:, 0] < room_map.shape[0])
                    & (pixels[:, 1] >= 0) & (pixels[:, 1] < room_map.shape[1])
                )
                raw = np.zeros(len(world_xy), dtype=int)
                raw[inside] = room_map[pixels[inside, 0], pixels[inside, 1]].astype(int)
                labels = np.asarray(
                    [str(mapping.get(int(value), "")) for value in raw], dtype=object
                )
                self._runtime_findings["observation_region_api"] = (
                    "scene.seg_map.room_ins_map+world_to_map"
                )
            except Exception as error:
                self._runtime_findings["observation_region_api_warning"] = str(error)
        # OG 3.9.2 only ships floor-0 room segmentation. Keep other floors and
        # unlabeled doorway pixels scientifically explicit as 2 m conceptual regions.
        missing = labels == ""
        if np.any(missing):
            cells = np.floor(world_xy[missing] / 2.0).astype(int)
            labels[missing] = [f"conceptual_{x:+04d}_{y:+04d}" for x, y in cells]
        return labels.astype(str)

    def prepare_navigation_context(
        self,
        configuration_token: str,
        seed: int,
        *,
        force: bool = False,
    ) -> dict[int, NavigationContext]:
        """Build one reusable footprint-aware context per feasible floor."""
        from multi_view_world_dataset.adapters.navigation import (
            build_navigation_contexts,
        )

        return build_navigation_contexts(
            self, configuration_token, seed, force=force
        )

    def navigation_context_metadata(self) -> dict[str, Any]:
        """Return JSON-safe configuration-level navigation diagnostics."""
        return {
            str(floor_index): context.metadata()
            for floor_index, context in sorted(self._navigation_contexts.items())
        }

    def sample_route_first_trajectory_sets(
        self,
        seed: int,
        *,
        discouraged_region_ids: tuple[str, ...] = (),
    ) -> tuple[
        dict[str, float],
        tuple[tuple[tuple[Trajectory, ...], dict[str, Any]], ...],
    ]:
        """Select and sparsely validate route triplets from the cached bank."""
        from multi_view_world_dataset.adapters.navigation import (
            sample_route_first_trajectory_sets,
        )

        return sample_route_first_trajectory_sets(
            self,
            seed,
            discouraged_region_ids=discouraged_region_ids,
        )

    def place_development_robots(
        self, seed: int, *, discouraged_region_ids: tuple[str, ...] = ()
    ) -> dict[str, float]:
        """Sample separated starts from an OG room / conceptual observation region."""
        warnings.warn(
            "place_development_robots is legacy compatibility only; use the "
            "route-first production pipeline",
            DeprecationWarning, stacklevel=2,
        )
        scene = self._require_scene()
        robots = list(self._env.robots)
        if len(robots) != 3:
            raise SimulatorUnavailableError("Development placement requires exactly three loaded robots")
        rng = np.random.default_rng(seed)
        placement = self.config["placement"]
        floor_index = int(rng.integers(int(scene.n_floors)))
        world_xy, is_path_traversable, _, reachable_candidates = self._trajectory_traversability(
            floor_index, robots[0]
        )
        # The installed static traversability raster does not encode the current
        # randomized movable furniture. Remove its expanded world AABBs before
        # choosing starts; contact QA below remains authoritative.
        floor_id = f"floor_{floor_index:02d}"
        dynamic_objects = [
            obj for obj in self.object_catalog()
            if not obj.structural and obj.floor_id == floor_id
            and obj.bbox_max_world[2] > float(scene.get_floor_height(floor_index)) + 0.10
        ]
        configured_start_clearance_m = float(
            placement.get("dynamic_object_start_clearance_m", 0.35)
        )
        start_clearance_m = configured_start_clearance_m
        free = np.ones(len(world_xy), dtype=bool)
        for obj in dynamic_objects:
            free &= ~(
                (world_xy[:, 0] >= obj.bbox_min_world[0] - start_clearance_m)
                & (world_xy[:, 0] <= obj.bbox_max_world[0] + start_clearance_m)
                & (world_xy[:, 1] >= obj.bbox_min_world[1] - start_clearance_m)
                & (world_xy[:, 1] <= obj.bbox_max_world[1] + start_clearance_m)
            )
        world_xy = world_xy[free]
        if len(world_xy) < 3:
            raise SampleRejected("insufficient_traversable_starts", {"floor_index": floor_index})
        labels = self._observation_region_labels(floor_index, world_xy)
        regions = {
            label: np.flatnonzero(labels == label)
            for label in sorted(set(labels))
            if np.count_nonzero(labels == label) >= 3
        }
        if not regions:
            raise SampleRejected("no_observation_regions", {"floor_index": floor_index})
        regime = choose_weighted_label(placement["observation_regime_weights"], rng)
        requested_regime = regime
        region_ids = tuple(regions)
        reuse_penalty = float(placement.get("sibling_region_reuse_penalty", 1.0))
        anchor_weights = np.asarray([
            1.0 / (1.0 + reuse_penalty * int(region_id in discouraged_region_ids))
            for region_id in region_ids
        ])
        anchor = str(rng.choice(region_ids, p=anchor_weights / anchor_weights.sum()))
        adjacency_distance = float(placement["region_adjacency_distance_m"])
        anchor_points = world_xy[regions[anchor]]
        anchor_sample = anchor_points[
            np.linspace(0, len(anchor_points) - 1, min(256, len(anchor_points))).astype(int)
        ]
        adjacent = []
        for region_id in region_ids:
            if region_id == anchor:
                continue
            points = world_xy[regions[region_id]]
            sample = points[np.linspace(0, len(points) - 1, min(256, len(points))).astype(int)]
            if float(np.min(np.linalg.norm(anchor_sample[:, None] - sample[None, :], axis=2))) <= adjacency_distance:
                adjacent.append(region_id)
        if not adjacent:
            regime = "dense_shared"
        if regime == "dense_shared":
            target_regions = (anchor,)
        elif regime == "partial_chain":
            target_regions = (anchor, str(rng.choice(adjacent)))
        else:
            candidates = [anchor, *adjacent]
            target_regions = tuple(candidates[index] for index in rng.permutation(len(candidates))[:3])
        spatial_scales = placement["regime_spatial_soft_scale_m"]
        region_sampling_anchors: dict[str, np.ndarray] = {}
        if regime == "dense_shared":
            region_sampling_anchors[anchor] = anchor_points[int(rng.integers(len(anchor_points)))]
        elif regime == "partial_chain" and len(target_regions) == 2:
            other = target_regions[1]
            other_points = world_xy[regions[other]]
            other_sample = other_points[
                np.linspace(0, len(other_points) - 1, min(256, len(other_points))).astype(int)
            ]
            distances = np.linalg.norm(
                anchor_sample[:, None] - other_sample[None, :], axis=2
            )
            anchor_index, other_index = np.unravel_index(
                int(np.argmin(distances)), distances.shape
            )
            region_sampling_anchors[anchor] = anchor_sample[anchor_index]
            region_sampling_anchors[other] = other_sample[other_index]
        headroom_by_regime = placement[
            "regime_trajectory_separation_headroom_m"
        ]
        minimum_distance = (
            float(placement["minimum_pairwise_distance_m"])
            + float(headroom_by_regime[regime])
        )
        selected_indices: list[int] = []
        attempts = 0
        maximum_attempts = int(placement["maximum_attempts"])
        target_pool = np.concatenate([regions[name] for name in target_regions])
        nearby_region_ids = tuple(dict.fromkeys((anchor, *adjacent)))
        nearby_pool = np.concatenate([
            regions[name] for name in nearby_region_ids
        ])
        all_pool = np.arange(len(world_xy))
        # Geodesic distance is never shorter than Euclidean displacement.
        # Requiring the full arc-length lower bound here prevents starts in
        # tiny connected components from reaching the expensive path sampler.
        minimum_reachable_displacement = float(
            self.config["trajectory"]["path_length_min_m"]
        )
        maximum_reachable_displacement = float(
            self.config["trajectory"]["path_length_max_m"]
        )
        viability_cache: dict[tuple[float, float], bool] = {}
        heading_exit_cache: dict[tuple[float, float], bool] = {}
        placement_probe_min_m = float(placement["initial_heading_probe_min_m"])
        placement_probe_max_m = float(placement["initial_heading_probe_max_m"])
        placement_probe_spacing_m = float(
            self.config["trajectory"]["line_validation_spacing_m"]
        )

        def start_has_path_neighborhood(point: np.ndarray) -> bool:
            key = tuple(map(float, point))
            if key not in viability_cache:
                reachable = reachable_candidates(point)
                distances = np.linalg.norm(reachable - point, axis=1)
                viability_cache[key] = bool(
                    np.any(
                        (distances >= minimum_reachable_displacement)
                        & (distances <= maximum_reachable_displacement)
                    )
                )
            return viability_cache[key]

        def start_has_long_traversable_exit(point: np.ndarray) -> bool:
            key = tuple(map(float, point))
            if key not in heading_exit_cache:
                try:
                    select_local_traversable_heading(
                        point,
                        reachable_candidates(point),
                        0.0,
                        is_path_traversable,
                        minimum_probe_m=placement_probe_min_m,
                        maximum_probe_m=placement_probe_max_m,
                        validation_spacing_m=placement_probe_spacing_m,
                    )
                    heading_exit_cache[key] = True
                except SampleRejected:
                    heading_exit_cache[key] = False
            return heading_exit_cache[key]

        consensus_preflight_checks = 0
        maximum_consensus_deviation_rad = np.deg2rad(
            float(self.config["trajectory"]["initial_heading_tolerance_deg"])
        )
        consensus_search_step_rad = np.deg2rad(
            float(placement["heading_consensus_search_step_deg"])
        )
        # Region membership is a sampling prior, not a compactness-like hard
        # constraint. Expand gracefully when a small room cannot fit 3 robots.
        for group_attempt in range(maximum_attempts):
            selected_indices = []
            for robot_index in range(len(robots)):
                desired = target_regions[robot_index % len(target_regions)]
                pools = (regions[desired], target_pool, nearby_pool, all_pool)
                chosen = None
                for pool in pools:
                    for index in (
                        soft_anchor_candidate_order(
                            pool, world_xy, region_sampling_anchors[desired],
                            float(spatial_scales[regime]), rng,
                            probability_floor=float(placement["spatial_soft_probability_floor"]),
                        )
                        if desired in region_sampling_anchors
                        else rng.permutation(pool)
                    )[:maximum_attempts]:
                        attempts += 1
                        point = world_xy[int(index)]
                        if not start_has_path_neighborhood(point):
                            continue
                        if not start_has_long_traversable_exit(point):
                            continue
                        if all(np.linalg.norm(point - world_xy[prior]) >= minimum_distance for prior in selected_indices):
                            chosen = int(index)
                            break
                    if chosen is not None:
                        break
                if chosen is None:
                    break
                selected_indices.append(chosen)
            if len(selected_indices) == len(robots):
                if regime in {"dense_shared", "partial_chain"}:
                    consensus_preflight_checks += 1
                    selected_points = world_xy[selected_indices]
                    try:
                        select_consensus_local_headings(
                            selected_points,
                            [reachable_candidates(point) for point in selected_points],
                            0.0,
                            is_path_traversable,
                            minimum_probe_m=placement_probe_min_m,
                            maximum_probe_m=placement_probe_max_m,
                            validation_spacing_m=placement_probe_spacing_m,
                            maximum_deviation_rad=maximum_consensus_deviation_rad,
                            angular_step_rad=consensus_search_step_rad,
                        )
                    except SampleRejected:
                        selected_indices = []
                        continue
                break
        if len(selected_indices) != 3:
            raise SampleRejected(
                "region_balanced_placement_failed",
                {"regime": regime, "target_regions": target_regions, "attempts": attempts},
            )
        selected_xy = world_xy[selected_indices]
        z = float(scene.get_floor_height(floor_index))
        selected = np.column_stack((selected_xy, np.full(3, z)))
        sampled_heights: dict[str, float] = {}
        pitch = np.deg2rad(float(self.config["camera"]["pitch_deg"]))
        cosine, sine = np.cos(pitch), np.sin(pitch)
        camera_rotation_base = np.array(
            [[0.0, sine, cosine], [-1.0, 0.0, 0.0], [0.0, -cosine, sine]]
        )
        cv_to_usd = np.diag([1.0, -1.0, -1.0, 1.0])
        shared_heading = float(rng.uniform(-np.pi, np.pi))
        trajectory_guides: dict[str, np.ndarray] = {}
        if regime == "dense_shared":
            guide_points = lane_preserving_guides(
                selected_xy,
                shared_heading,
                float(placement["lane_guide_distance_m"]),
            )
            provisional_yaws = np.full(len(robots), shared_heading)
            trajectory_guides = {
                robot.name: guide_points[index].copy()
                for index, robot in enumerate(robots)
            }
            heading_policy = "soft_lane_preserving_shared_scene_direction"
        elif regime == "partial_chain":
            anchor_guide = region_sampling_anchors[anchor]
            other_guide = region_sampling_anchors[target_regions[1]]
            chain_heading = float(np.arctan2(
                other_guide[1] - anchor_guide[1],
                other_guide[0] - anchor_guide[0],
            ))
            guide_points = lane_preserving_guides(
                selected_xy,
                chain_heading,
                float(placement["lane_guide_distance_m"]),
            )
            provisional_yaws = np.full(len(robots), chain_heading)
            trajectory_guides = {
                robot.name: guide_points[index].copy()
                for index, robot in enumerate(robots)
            }
            heading_policy = "soft_lane_preserving_adjacent_region_direction"
        else:
            provisional_yaws = rng.uniform(-np.pi, np.pi, len(robots))
            heading_policy = "independent_scene_directions"
        provisional_yaws = (provisional_yaws + np.pi) % (2.0 * np.pi) - np.pi
        # Preserve each regime's scene-view direction as a soft prior, but
        # project it onto an actually reachable outgoing direction. This avoids
        # sampling a robot that can only slide sideways relative to its initial
        # body and camera heading.
        minimum_heading_probe_m = float(placement["initial_heading_probe_min_m"])
        maximum_heading_probe_m = float(placement["initial_heading_probe_max_m"])
        heading_validation_spacing_m = float(
            self.config["trajectory"]["line_validation_spacing_m"]
        )
        desired_yaws = provisional_yaws.copy()
        heading_adjustments = []
        for index, (point, desired_yaw) in enumerate(
            zip(selected_xy, desired_yaws, strict=True)
        ):
            reachable = reachable_candidates(point)
            try:
                chosen, heading_error = select_local_traversable_heading(
                    point,
                    reachable,
                    float(desired_yaw),
                    is_path_traversable,
                    minimum_probe_m=minimum_heading_probe_m,
                    maximum_probe_m=maximum_heading_probe_m,
                    validation_spacing_m=heading_validation_spacing_m,
                )
            except SampleRejected as error:
                raise SampleRejected(
                    error.reason,
                    {"robot_index": index, "start_xy": point.tolist()},
                ) from error
            provisional_yaws[index] = chosen
            heading_adjustments.append(heading_error)
        if regime in {"dense_shared", "partial_chain"}:
            maximum_regime_heading_error = np.deg2rad(
                float(self.config["trajectory"]["initial_heading_tolerance_deg"])
            )
            if max(heading_adjustments) > maximum_regime_heading_error:
                reachable_pools = [
                    reachable_candidates(point)
                    for point in selected_xy
                ]
                (
                    provisional_yaws,
                    heading_adjustments,
                    consensus_yaw,
                ) = select_consensus_local_headings(
                    selected_xy,
                    reachable_pools,
                    float(desired_yaws[0]),
                    is_path_traversable,
                    minimum_probe_m=minimum_heading_probe_m,
                    maximum_probe_m=maximum_heading_probe_m,
                    validation_spacing_m=heading_validation_spacing_m,
                    maximum_deviation_rad=maximum_regime_heading_error,
                    angular_step_rad=np.deg2rad(
                        float(placement["heading_consensus_search_step_deg"])
                    ),
                )
                desired_yaws.fill(consensus_yaw)
            heading_policy = f"{heading_policy}_with_local_exit_consensus"
        # Every projected heading was densely validated from its start through
        # at least minimum_heading_probe_m. Persist that guaranteed endpoint as
        # the per-robot guide. The trajectory sampler uses it once as a hard
        # direct candidate, then keeps the remaining family draws stochastic.
        guide_distance = minimum_heading_probe_m
        guide_directions = np.column_stack(
            (np.cos(provisional_yaws), np.sin(provisional_yaws))
        )
        guide_points = selected_xy + guide_distance * guide_directions
        trajectory_guides = {
            robot.name: guide_points[index].copy()
            for index, robot in enumerate(robots)
        }
        heights = self.config["camera"]["heights_m"]
        for robot, point, yaw in zip(robots, selected, provisional_yaws, strict=True):
            orientation = self._transform_utils.euler2quat(
                self._th.tensor([0.0, 0.0, float(yaw)])
            )
            robot.set_position_orientation(
                position=self._th.as_tensor(point, dtype=self._th.float32),
                orientation=orientation,
            )
            height = float(rng.choice(heights))
            sampled_heights[robot.name] = height
            if self._using_final_robot:
                mast_joint = next(joint for name, joint in robot.joints.items() if name.endswith("mvwd_mast_joint"))
                mast_joint.set_pos(height - float(min(heights)), drive=False)
            base_to_world = self._pose_matrix(robot)
            camera_to_base = np.eye(4)
            camera_to_base[:3, :3] = camera_rotation_base
            camera_to_base[:3, 3] = [0.08 if self._using_final_robot else 0.0, 0.0, height]
            usd_camera_to_world = base_to_world @ camera_to_base @ cv_to_usd
            camera_position, camera_orientation = self._transform_utils.mat2pose(
                self._th.as_tensor(usd_camera_to_world, dtype=self._th.float32)
            )
            sensors = [sensor for sensor in robot.sensors.values() if isinstance(sensor, self._vision_sensor_type)]
            if len(sensors) != 1:
                raise SimulatorUnavailableError(f"Expected one VisionSensor on {robot.name}")
            if not self._using_final_robot:
                sensors[0].set_position_orientation(position=camera_position, orientation=camera_orientation)
            self._development_camera_mounts[robot.name] = camera_to_base.copy()
        settle_steps = min(30, int(self.config["generation"]["settle_steps"]))
        for _ in range(settle_steps):
            for robot in robots:
                robot.keep_still()
            self._og.sim.step_physics()
        for robot, point, yaw in zip(robots, selected, provisional_yaws, strict=True):
            orientation = self._transform_utils.euler2quat(self._th.tensor([0.0, 0.0, float(yaw)]))
            robot.set_position_orientation(position=self._th.as_tensor(point, dtype=self._th.float32), orientation=orientation)
            self._restore_final_robot_mast_mount(robot)
            robot.keep_still()
        floors = self._robot_support_surfaces()
        for robot in robots:
            pairs = self._external_robot_contact_pairs(robot, floors)
            if pairs:
                raise SampleRejected("initial_robot_collision", {"robot_id": robot.name, "contact_pairs": pairs[:50]})
        self._runtime_findings["sampled_floor_index"] = floor_index
        self._runtime_findings["placement_observation_regime"] = regime
        self._runtime_findings["placement_requested_observation_regime"] = requested_regime
        self._runtime_findings["placement_start_region_ids"] = [str(labels[index]) for index in selected_indices]
        self._runtime_findings["placement_target_region_ids"] = list(target_regions)
        self._runtime_findings["placement_region_sampling_anchors_xy"] = {
            name: point.tolist() for name, point in region_sampling_anchors.items()
        }
        self._runtime_findings["placement_trajectory_guides_xy"] = {
            robot_id: point.tolist()
            for robot_id, point in trajectory_guides.items()
        }
        self._runtime_findings["placement_region_adjacency"] = {anchor: sorted(adjacent)}
        self._runtime_findings["placement_heading_mode"] = (
            f"{heading_policy}_then_accepted_trajectory_tangent"
        )
        self._runtime_findings["placement_desired_headings_rad"] = desired_yaws.tolist()
        self._runtime_findings["placement_reachable_headings_rad"] = provisional_yaws.tolist()
        self._runtime_findings["placement_heading_adjustments_rad"] = heading_adjustments
        self._runtime_findings["placement_heading_probe"] = {
            "minimum_m": minimum_heading_probe_m,
            "maximum_m": maximum_heading_probe_m,
            "validation_spacing_m": heading_validation_spacing_m,
            "source": "robot_eroded_direct_local_exit",
        }
        self._runtime_findings["placement_attempts"] = attempts
        self._runtime_findings["placement_minimum_start_distance_m"] = minimum_distance
        self._runtime_findings["placement_dynamic_object_filter"] = {
            "configured_clearance_m": configured_start_clearance_m,
            "clearance_m": start_clearance_m,
            "dynamic_object_count": len(dynamic_objects),
            "remaining_candidate_count": len(world_xy),
            "path_viable_candidate_checks": len(viability_cache),
            "long_exit_candidate_checks": len(heading_exit_cache),
            "heading_consensus_preflight_checks": consensus_preflight_checks,
        }
        self._runtime_findings["development_camera_heights_m"] = sampled_heights
        return sampled_heights

    def _trajectory_traversability(
        self,
        floor_index: int,
        robot: Any,
    ) -> tuple[
        np.ndarray, Callable[[np.ndarray], bool],
        Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, float] | None],
        Callable[[np.ndarray], np.ndarray],
    ]:
        """Return strict robot-eroded candidates, validator, and OG planner."""
        scene = self._require_scene()
        trav_map = scene.trav_map
        eroded = self._robot_eroded_traversability(floor_index, robot)
        pixels = self._th.stack(self._th.where(eroded == 255), dim=1)
        world_xy = self._native_value(trav_map.map_to_world(pixels)).astype(np.float64)
        pixels_native = self._native_value(pixels).astype(np.int64)

        eroded_native = self._native_value(eroded) == 255
        floor_id = f"floor_{floor_index:02d}"
        floor_height = float(scene.get_floor_height(floor_index))
        obstacle_height_m = float(
            self.config["placement"].get(
                "dynamic_object_path_obstacle_height_m", 0.65
            )
        )
        dynamic_objects = [
            obj
            for obj in self.object_catalog()
            if not obj.structural
            and obj.floor_id == floor_id
            and obj.bbox_max_world[2] > floor_height + 0.10
            and obj.bbox_min_world[2] < floor_height + obstacle_height_m
        ]
        configured_dynamic_clearance_m = float(
            self.config["placement"].get("dynamic_object_path_clearance_m", 0.35)
        )
        dynamic_clearance_m = configured_dynamic_clearance_m
        dynamically_free = np.ones(len(world_xy), dtype=bool)
        for obj in dynamic_objects:
            dynamically_free &= ~(
                (world_xy[:, 0] >= obj.bbox_min_world[0] - dynamic_clearance_m)
                & (world_xy[:, 0] <= obj.bbox_max_world[0] + dynamic_clearance_m)
                & (world_xy[:, 1] >= obj.bbox_min_world[1] - dynamic_clearance_m)
                & (world_xy[:, 1] <= obj.bbox_max_world[1] + dynamic_clearance_m)
            )
        blocked_pixels = pixels_native[~dynamically_free]
        if len(blocked_pixels):
            eroded_native[blocked_pixels[:, 0], blocked_pixels[:, 1]] = False
        pixels_native = pixels_native[dynamically_free]
        world_xy = world_xy[dynamically_free]
        self._runtime_findings["trajectory_dynamic_object_filter"] = {
            "configured_clearance_m": configured_dynamic_clearance_m,
            "clearance_m": dynamic_clearance_m,
            "obstacle_height_m": obstacle_height_m,
            "dynamic_object_count": len(dynamic_objects),
            "blocked_candidate_count": int(np.count_nonzero(~dynamically_free)),
        }

        # Restrict every robot to its actual 8-connected component of the
        # robot-eroded map. Sampling Euclidean-near goals from another component
        # made the native planner repeat hundreds of guaranteed failures.
        unassigned = {tuple(map(int, pixel)) for pixel in pixels_native}
        component_by_cell: dict[tuple[int, int], int] = {}
        component_cells: dict[int, np.ndarray] = {}
        component_index = 0
        while unassigned:
            seed_cell = unassigned.pop()
            stack = [seed_cell]
            members = [seed_cell]
            component_by_cell[seed_cell] = component_index
            while stack:
                row, column = stack.pop()
                for row_delta in (-1, 0, 1):
                    for column_delta in (-1, 0, 1):
                        if row_delta == 0 and column_delta == 0:
                            continue
                        neighbor = (row + row_delta, column + column_delta)
                        if neighbor in unassigned:
                            unassigned.remove(neighbor)
                            component_by_cell[neighbor] = component_index
                            members.append(neighbor)
                            stack.append(neighbor)
            component_cells[component_index] = np.asarray(members, dtype=np.int64)
            component_index += 1

        height, width = eroded_native.shape
        def is_path_traversable(path_xy: np.ndarray) -> bool:
            points = np.asarray(path_xy, dtype=np.float64)
            map_points = self._world_to_map_preserving_batch(trav_map, points)
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

        def reachable_candidates(source_xy: np.ndarray) -> np.ndarray:
            source_pixel = self._world_to_map_preserving_batch(trav_map, source_xy)
            component = component_by_cell.get(tuple(map(int, source_pixel)))
            if component is None:
                return np.empty((0, 2), dtype=np.float64)
            member_pixels = component_cells[component]
            return self._native_value(
                trav_map.map_to_world(
                    self._th.as_tensor(member_pixels, dtype=self._th.int64)
                )
            ).astype(np.float64)

        planning_source = self._th.where(
            self._th.as_tensor(eroded_native, dtype=self._th.bool),
            self._th.full_like(eroded, 255),
            self._th.zeros_like(eroded),
        )

        def plan_segment(
            source_xy: np.ndarray,
            target_xy: np.ndarray,
        ) -> tuple[np.ndarray, float] | None:
            original_waypoint_interval = int(trav_map.waypoint_interval)
            original_floor_map = trav_map.floor_map[floor_index]
            original_default_erosion_radius = float(
                trav_map.default_erosion_radius
            )
            trav_map.waypoint_interval = 1
            trav_map.floor_map[floor_index] = planning_source
            # This source is already conservatively robot-eroded. A sub-cell
            # radius makes OG's cv2 erosion a 1x1 identity kernel while still
            # using the installed shortest-path API and exact signature.
            trav_map.default_erosion_radius = 0.5 * float(
                trav_map.map_resolution
            )
            try:
                path, distance = scene.get_shortest_path(
                    floor_index,
                    self._th.as_tensor(source_xy, dtype=self._th.float32),
                    self._th.as_tensor(target_xy, dtype=self._th.float32),
                    entire_path=True,
                    robot=None,
                )
            finally:
                trav_map.floor_map[floor_index] = original_floor_map
                trav_map.waypoint_interval = original_waypoint_interval
                trav_map.default_erosion_radius = original_default_erosion_radius
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
            "erosion_source": "installed_omnigibson_3.9.2_robot_eroded_then_shortest_path",
            "planner_internal_erosion": "one_pixel_identity_kernel",
            "connected_component_count": len(component_cells),
            "connected_component_sizes": sorted(
                (len(cells) for cells in component_cells.values()), reverse=True
            ),
        }
        return world_xy, is_path_traversable, plan_segment, reachable_candidates

    def sample_robot_trajectories(
        self, seed: int, *, candidate_pool_size_override: int | None = None,
        maximum_attempts_override: int | None = None,
        observations_override: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[tuple[Trajectory, ...], dict[str, Any]]:
        observations = observations_override or self.robot_observations()
        warnings.warn(
            "sample_robot_trajectories is legacy compatibility only; use "
            "sample_route_first_trajectory_sets",
            DeprecationWarning, stacklevel=2,
        )
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
        world_xy, is_path_traversable, plan_segment, reachable_candidates = self._trajectory_traversability(
            floor_index, self._env.robots[0]
        )
        trajectory_config = self.config["trajectory"]
        reachable_by_robot = {
            robot_id: reachable_candidates(transform[:2, 3])
            for robot_id, transform in starts.items()
        }
        if any(not len(points) for points in reachable_by_robot.values()):
            raise SampleRejected(
                "trajectory_start_outside_eroded_component",
                {
                    "reachable_candidate_counts": {
                        robot_id: len(points)
                        for robot_id, points in reachable_by_robot.items()
                    }
                },
            )
        observation_regime = str(
            self._runtime_findings.get("placement_observation_regime", "partial_chain")
        )
        trajectory_guides = {
            robot_id: np.asarray(point, dtype=np.float64)
            for robot_id, point in self._runtime_findings.get(
                "placement_trajectory_guides_xy", {}
            ).items()
        }
        trajectories = sample_geodesic_trajectory_set(
            starts,
            mounts,
            reachable_by_robot,
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
            # Placement contributes only a regime-level scene-view prior. The
            # persisted yaw is still the exact accepted path tangent.
            initial_heading_tolerance_rad=np.deg2rad(
                float(trajectory_config["initial_heading_tolerance_deg"])
            ),
            derive_initial_heading_from_tangent=(
                trajectory_config["initial_heading_policy"] == "trajectory_tangent"
            ),
            initial_heading_probability_floor=float(
                trajectory_config["regime_initial_heading_probability_floor"][
                    observation_regime
                ]
            ),
            line_validation_spacing_m=float(trajectory_config["line_validation_spacing_m"]),
            smoothing_validation_spacing_m=float(trajectory_config["smoothing_validation_spacing_m"]),
            smoothing_strengths=trajectory_config["smoothing_strengths"],
            candidate_pool_size=int(
                candidate_pool_size_override or trajectory_config["candidate_pool_size"]
            ),
            maximum_attempts=int(
                maximum_attempts_override or trajectory_config["sampling_maximum_attempts"]
            ),
            joint_pool_rounds=int(trajectory_config["joint_pool_rounds"]),
            maximum_joint_valid_candidates=int(
                trajectory_config["maximum_joint_valid_candidates"]
            ),
            maximum_control_turn_rad=np.deg2rad(
                float(trajectory_config["maximum_control_turn_deg"])
            ),
            observation_regime=observation_regime,
            trajectory_guides_xy=trajectory_guides or None,
            guide_soft_scale_m=float(
                trajectory_config["regime_trajectory_guide_soft_scale_m"][observation_regime]
            ),
            guide_probability_floor=float(
                trajectory_config["regime_trajectory_guide_probability_floor"][observation_regime]
            ),
            formation_degeneracy_limits=self.config["placement"]["formation_degeneracy"],
            regime_coverage_saturation_m2=trajectory_config["regime_coverage_saturation_m2"],
            regime_initial_heading_prior_weights=trajectory_config["regime_initial_heading_prior_weights"],
            regime_view_connectivity_weights=trajectory_config["regime_view_connectivity_weights"],
            camera_hfov_deg=float(self.config["camera"]["hfov_deg"]),
        )
        joint_metrics = joint_trajectory_metrics(
            trajectories, camera_hfov_deg=float(self.config["camera"]["hfov_deg"])
        )
        traversed_regions = {
            trajectory.robot_id: sorted(set(self._observation_region_labels(
                floor_index, trajectory.base_to_world[:, :2, 3]
            ).tolist()))
            for trajectory in trajectories
        }
        metrics = {
            "floor_index": floor_index,
            "traversable_candidate_count": int(len(world_xy)),
            "reachable_candidate_counts": {
                robot_id: len(points)
                for robot_id, points in reachable_by_robot.items()
            },
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
            "minimum_pairwise_distance_m": float(
                joint_metrics["minimum_inter_robot_distance_m"]
            ),
            "joint_diversity": joint_metrics,
            "observation_regime": str(
                self._runtime_findings.get("placement_observation_regime", "partial_chain")
            ),
            "trajectory_guides_xy": {
                robot_id: point.tolist()
                for robot_id, point in trajectory_guides.items()
            },
            "start_region_ids": list(
                self._runtime_findings.get("placement_start_region_ids", [])
            ),
            "target_region_ids": list(
                self._runtime_findings.get("placement_target_region_ids", [])
            ),
            "traversed_region_ids": traversed_regions,
            "unique_traversed_region_count": len({
                region for values in traversed_regions.values() for region in values
            }),
        }
        return trajectories, metrics

    def sample_robot_trajectory_sets(
        self, seed: int
    ) -> tuple[tuple[tuple[Trajectory, ...], dict[str, Any]], ...]:
        """Generate a nested pool of joint sets for one fixed placement."""
        count = int(self.config["trajectory"]["trajectory_sets_per_placement"])
        warnings.warn(
            "sample_robot_trajectory_sets is legacy compatibility only; use "
            "sample_route_first_trajectory_sets",
            DeprecationWarning, stacklevel=2,
        )
        candidates: list[tuple[tuple[Trajectory, ...], dict[str, Any]]] = []
        observations = self.robot_observations()
        failures: list[dict[str, Any]] = []
        for index in range(count):
            try:
                candidates.append(
                    self.sample_robot_trajectories(
                        seed + 130363 * (index + 1),
                        candidate_pool_size_override=max(
                            3,
                            int(np.ceil(
                                self.config["trajectory"]["candidate_pool_size"] / count
                            )),
                        ),
                        maximum_attempts_override=max(
                            50,
                            int(np.ceil(
                                self.config["trajectory"]["sampling_maximum_attempts"] / count
                            )),
                        ),
                        observations_override=observations,
                    )
                )
            except SampleRejected as error:
                failures.append({"reason": error.reason, "details": error.details})
        if not candidates:
            raise SampleRejected(
                "trajectory_set_pool_exhausted",
                {"requested_set_count": count, "failures": failures[:20]},
            )
        rng = np.random.default_rng(seed)
        utilities = []
        for _, metrics in candidates:
            joint = metrics["joint_diversity"]
            regime = str(metrics["observation_regime"])
            utilities.append(
                regime_trajectory_soft_score(
                    joint, regime, self.config["trajectory"]["regime_coverage_saturation_m2"],
                    jitter=float(rng.uniform(0.0, 0.25)),
                    heading_prior_weights=self.config["trajectory"]["regime_initial_heading_prior_weights"],
                    view_connectivity_weights=self.config["trajectory"]["regime_view_connectivity_weights"],
                )
            )
        ranked: list[tuple[tuple[Trajectory, ...], dict[str, Any]]] = []
        ranking = np.argsort(-np.asarray(utilities))
        candidate_diversity = [item[1]["joint_diversity"] for item in candidates]
        for rank, candidate_index in enumerate(ranking.tolist()):
            trajectories, original_metrics = candidates[int(candidate_index)]
            metrics = dict(original_metrics)
            metrics["nested_trajectory_sets"] = {
                "requested": count,
                "accepted": len(candidates),
                "candidate_index": int(candidate_index),
                "soft_score_rank": rank,
                "soft_score": float(utilities[int(candidate_index)]),
                "rejection_count": len(failures),
                "generation_failures": failures[:20],
                "candidate_joint_diversity": candidate_diversity,
            }
            ranked.append((trajectories, metrics))
        return tuple(ranked)

    def complementary_trajectory_hybrids(
        self,
        candidate_sets: tuple[tuple[tuple[Trajectory, ...], dict[str, Any]], ...],
        candidate_failures: list[dict[str, Any]],
    ) -> tuple[tuple[tuple[Trajectory, ...], dict[str, Any]], ...]:
        """Build a bounded hybrid pool only from complementary measured overlap graphs."""
        trajectory_config = self.config["trajectory"]
        specs = complementary_hybrid_trajectory_sets(
            candidate_sets,
            candidate_failures,
            minimum_pairwise_distance_m=float(
                self.config["placement"]["minimum_pairwise_distance_m"]
            ),
            minimum_waypoint_trajectories=int(
                trajectory_config["minimum_waypoint_trajectories"]
            ),
            formation_degeneracy_limits=self.config["placement"]["formation_degeneracy"],
            coverage_saturation_m2=trajectory_config["regime_coverage_saturation_m2"],
            heading_prior_weights=trajectory_config[
                "regime_initial_heading_prior_weights"
            ],
            view_connectivity_weights=trajectory_config[
                "regime_view_connectivity_weights"
            ],
            camera_hfov_deg=float(self.config["camera"]["hfov_deg"]),
            maximum_candidates=int(
                trajectory_config["maximum_complementary_hybrid_candidates"]
            ),
        )
        hybrids: list[tuple[tuple[Trajectory, ...], dict[str, Any]]] = []
        for hybrid_rank, (trajectories, source_by_robot, joint_metrics) in enumerate(specs):
            cloned = tuple(
                replace(
                    trajectory,
                    metadata={
                        **trajectory.metadata,
                        "joint_pool_hybrid": True,
                        "hybrid_source_candidate_rank": int(
                            source_by_robot[trajectory.robot_id]
                        ),
                        "joint_diversity_metrics": dict(joint_metrics),
                    },
                )
                for trajectory in trajectories
            )
            template_source = min(source_by_robot.values())
            template_metrics = candidate_sets[template_source][1]
            robot_metrics = {
                robot_id: candidate_sets[source_index][1]["robots"][robot_id]
                for robot_id, source_index in source_by_robot.items()
            }
            traversed_regions = {
                robot_id: candidate_sets[source_index][1]["traversed_region_ids"][
                    robot_id
                ]
                for robot_id, source_index in source_by_robot.items()
            }
            nested = dict(template_metrics.get("nested_trajectory_sets", {}))
            nested.update({
                "candidate_kind": "complementary_hybrid",
                "hybrid_rank": hybrid_rank,
                "source_candidate_by_robot": source_by_robot,
                "source_overlap_evidence": joint_metrics["complementary_hybrid"],
            })
            metrics = {
                **template_metrics,
                "robots": robot_metrics,
                "minimum_pairwise_distance_m": float(
                    joint_metrics["minimum_inter_robot_distance_m"]
                ),
                "joint_diversity": joint_metrics,
                "traversed_region_ids": traversed_regions,
                "unique_traversed_region_count": len({
                    region
                    for values in traversed_regions.values()
                    for region in values
                }),
                "nested_trajectory_sets": nested,
            }
            hybrids.append((cloned, metrics))
        return tuple(hybrids)

    def measured_overlap_bridge_trajectories(
        self,
        candidate_sets: tuple[tuple[tuple[Trajectory, ...], dict[str, Any]], ...],
        candidate_failures: list[dict[str, Any]],
        seed: int,
    ) -> tuple[tuple[tuple[Trajectory, ...], dict[str, Any]], ...]:
        """Mutate native RouteBank triplets using measured GT edge evidence."""
        del seed  # RouteBank ordering and measured evidence are deterministic.
        from multi_view_world_dataset.adapters.navigation import (
            measured_overlap_route_mutations,
        )

        return measured_overlap_route_mutations(
            self,
            candidate_sets,
            candidate_failures,
            maximum_candidates=int(
                self.config["trajectory"][
                    "maximum_measured_overlap_bridge_candidates"
                ]
            ),
        )

    def _legacy_geodesic_overlap_bridge_trajectories(
        self,
        candidate_sets: tuple[tuple[tuple[Trajectory, ...], dict[str, Any]], ...],
        candidate_failures: list[dict[str, Any]],
        seed: int,
    ) -> tuple[tuple[tuple[Trajectory, ...], dict[str, Any]], ...]:
        """Resample only an isolated robot after GT depth measured one edge."""
        trajectory_config = self.config["trajectory"]
        maximum_candidates = int(
            trajectory_config["maximum_measured_overlap_bridge_candidates"]
        )
        if maximum_candidates <= 0:
            return ()
        failure_by_rank = {
            int(item["candidate_rank"]): item
            for item in candidate_failures
            if isinstance(item.get("candidate_rank"), int)
            and item.get("candidate_kind", "base") == "base"
            and item.get("reason") == "trajectory_temporal_overlap_failed"
        }
        robot_ids = tuple(
            sorted(trajectory.robot_id for trajectory in candidate_sets[0][0])
        )
        floor_index = int(candidate_sets[0][1]["floor_index"])
        _, is_path_traversable, plan_segment, reachable_candidates = (
            self._trajectory_traversability(floor_index, self._env.robots[0])
        )
        floor_z = float(self._require_scene().get_floor_height(floor_index))
        minimum_distance = float(
            self.config["placement"]["minimum_pairwise_distance_m"]
        )
        target_depth = float(
            trajectory_config["measured_overlap_bridge_target_depth_m"]
        )
        guide_distance = float(
            trajectory_config["measured_overlap_bridge_guide_distance_m"]
        )
        ranked: list[
            tuple[float, tuple[Trajectory, ...], dict[str, Any]]
        ] = []
        sampling_index = 0
        for source_rank, (source_trajectories, source_metrics) in enumerate(
            candidate_sets
        ):
            failure = failure_by_rank.get(source_rank)
            if failure is None:
                continue
            edges = {
                tuple(sorted(map(str, edge)))
                for edge in failure.get("details", {}).get("union_edges", ())
                if len(edge) == 2
            }
            if len(edges) != 1:
                continue
            preserved_edge = next(iter(edges))
            isolated_ids = sorted(set(robot_ids) - set(preserved_edge))
            if len(isolated_ids) != 1:
                continue
            isolated_id = isolated_ids[0]
            source_by_id = {
                trajectory.robot_id: trajectory
                for trajectory in source_trajectories
            }
            isolated_source = source_by_id[isolated_id]
            start_xy = isolated_source.base_to_world[0, :2, 3]
            camera_mount = (
                np.linalg.inv(isolated_source.base_to_world[0])
                @ isolated_source.camera_to_world[0]
            )
            reachable = reachable_candidates(start_xy)
            if not len(reachable):
                continue
            keyframes = failure.get("details", {}).get("keyframes", ())
            edge_key = "|".join(preserved_edge)
            edge_frames = [
                int(keyframe["frame_index"])
                for keyframe in keyframes
                if preserved_edge
                in {
                    tuple(sorted(map(str, edge)))
                    for edge in keyframe.get("edges", ())
                }
            ]
            phases = edge_frames or [
                0,
                isolated_source.frames // 2,
                isolated_source.frames - 1,
            ]
            for anchor_id in preserved_edge:
                anchor = source_by_id[anchor_id]
                for phase in phases[:2]:
                    phase = min(max(int(phase), 0), anchor.frames - 1)
                    camera = anchor.camera_to_world[phase]
                    forward = np.asarray(camera[:2, 2], dtype=np.float64)
                    norm = float(np.linalg.norm(forward))
                    if norm <= 1.0e-8:
                        continue
                    forward /= norm
                    shared_target = camera[:2, 3] + target_depth * forward
                    target_source = "camera_ray_fallback"
                    centroid = next(
                        (
                            keyframe.get(
                                "shared_surface_centroids_world", {}
                            ).get(edge_key)
                            for keyframe in keyframes
                            if int(keyframe["frame_index"]) == phase
                        ),
                        None,
                    )
                    if centroid is not None:
                        centroid_xy = np.asarray(centroid, dtype=np.float64)[:2]
                        if np.isfinite(centroid_xy).all():
                            shared_target = centroid_xy
                            target_source = "gt_depth_shared_surface_centroid"
                    direction_specs = (
                        ("shared_target", shared_target - start_xy),
                        ("parallel_ray", forward),
                    )
                    for strategy, desired_direction in direction_specs:
                        direction_norm = float(np.linalg.norm(desired_direction))
                        if direction_norm <= 1.0e-8:
                            continue
                        direction = desired_direction / direction_norm
                        desired_yaw = float(np.arctan2(direction[1], direction[0]))
                        sampling_start = isolated_source.base_to_world[0].copy()
                        cosine, sine = np.cos(desired_yaw), np.sin(desired_yaw)
                        sampling_start[:2, :2] = [
                            [cosine, -sine],
                            [sine, cosine],
                        ]
                        guide = start_xy + guide_distance * direction
                        sampling_index += 1
                        try:
                            rescue_pool = sample_geodesic_robot_trajectory_pool(
                                isolated_id,
                                sampling_start,
                                camera_mount,
                                reachable,
                                floor_z,
                                np.random.default_rng(
                                    seed + 104729 * sampling_index
                                ),
                                frames=isolated_source.frames,
                                fps=isolated_source.fps,
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
                                is_path_traversable=is_path_traversable,
                                path_family_weights=trajectory_config[
                                    "path_family_weights"
                                ],
                                initial_heading_tolerance_rad=np.deg2rad(
                                    float(
                                        trajectory_config[
                                            "initial_heading_tolerance_deg"
                                        ]
                                    )
                                ),
                                derive_initial_heading_from_tangent=True,
                                initial_heading_probability_floor=float(
                                    trajectory_config[
                                        "measured_overlap_bridge_heading_floor"
                                    ]
                                ),
                                maximum_control_turn_rad=np.deg2rad(
                                    float(
                                        trajectory_config[
                                            "maximum_control_turn_deg"
                                        ]
                                    )
                                ),
                                line_validation_spacing_m=float(
                                    trajectory_config[
                                        "line_validation_spacing_m"
                                    ]
                                ),
                                smoothing_validation_spacing_m=float(
                                    trajectory_config[
                                        "smoothing_validation_spacing_m"
                                    ]
                                ),
                                smoothing_strengths=trajectory_config[
                                    "smoothing_strengths"
                                ],
                                candidate_pool_size=int(
                                    trajectory_config[
                                        "measured_overlap_bridge_pool_size"
                                    ]
                                ),
                                maximum_attempts=int(
                                    trajectory_config[
                                        "measured_overlap_bridge_sampling_attempts"
                                    ]
                                ),
                                guide_xy=guide,
                                guide_soft_scale_m=float(
                                    trajectory_config[
                                        "measured_overlap_bridge_guide_scale_m"
                                    ]
                                ),
                                guide_probability_floor=float(
                                    trajectory_config[
                                        "measured_overlap_bridge_guide_floor"
                                    ]
                                ),
                            )
                        except SampleRejected:
                            continue
                        for rescue in rescue_pool:
                            trajectories = tuple(
                                rescue if robot_id == isolated_id
                                else source_by_id[robot_id]
                                for robot_id in robot_ids
                            )
                            positions = np.stack(
                                [
                                    trajectory.base_to_world[:, :2, 3]
                                    for trajectory in trajectories
                                ]
                            )
                            pairwise = np.stack(
                                [
                                    np.linalg.norm(
                                        positions[left] - positions[right],
                                        axis=1,
                                    )
                                    for left in range(len(trajectories))
                                    for right in range(left + 1, len(trajectories))
                                ]
                            )
                            if float(pairwise.min()) < minimum_distance:
                                continue
                            if (
                                sum(
                                    trajectory.path_family != "direct"
                                    for trajectory in trajectories
                                )
                                < int(
                                    trajectory_config[
                                        "minimum_waypoint_trajectories"
                                    ]
                                )
                            ):
                                continue
                            joint_metrics = joint_trajectory_metrics(
                                trajectories,
                                camera_hfov_deg=float(
                                    self.config["camera"]["hfov_deg"]
                                ),
                            )
                            if formation_degenerate(
                                joint_metrics,
                                self.config["placement"][
                                    "formation_degeneracy"
                                ],
                            ):
                                continue
                            joint_metrics["formation_degenerate"] = False
                            evidence = {
                                "strategy": "measured_overlap_directed_resample",
                                "source_candidate_rank": source_rank,
                                "preserved_measured_edge": list(preserved_edge),
                                "isolated_robot_id": isolated_id,
                                "anchor_robot_id": anchor_id,
                                "anchor_frame_index": phase,
                                "view_direction_strategy": strategy,
                                "shared_target_xy": shared_target.tolist(),
                                "shared_target_source": target_source,
                            }
                            joint_metrics["measured_overlap_bridge"] = evidence
                            cloned = tuple(
                                replace(
                                    trajectory,
                                    metadata={
                                        **trajectory.metadata,
                                        "joint_pool_hybrid": True,
                                        "joint_diversity_metrics": dict(
                                            joint_metrics
                                        ),
                                        "measured_overlap_bridge": evidence,
                                    },
                                )
                                for trajectory in trajectories
                            )
                            robot_metrics = dict(source_metrics["robots"])
                            robot_metrics[isolated_id] = {
                                **trajectory_kinematic_metrics(rescue),
                                "path_family": rescue.path_family,
                                "control_waypoints_xy": (
                                    rescue.control_waypoints_xy.tolist()
                                ),
                                "planner_geodesic_length_m": float(
                                    rescue.metadata[
                                        "planner_geodesic_length_m"
                                    ]
                                ),
                            }
                            traversed_regions = dict(
                                source_metrics["traversed_region_ids"]
                            )
                            traversed_regions[isolated_id] = sorted(
                                set(
                                    self._observation_region_labels(
                                        floor_index,
                                        rescue.base_to_world[:, :2, 3],
                                    ).tolist()
                                )
                            )
                            nested = dict(
                                source_metrics.get(
                                    "nested_trajectory_sets", {}
                                )
                            )
                            nested.update(
                                {
                                    "candidate_kind": (
                                        "measured_overlap_bridge"
                                    ),
                                    "source_overlap_evidence": evidence,
                                }
                            )
                            metrics = {
                                **source_metrics,
                                "robots": robot_metrics,
                                "minimum_pairwise_distance_m": float(
                                    joint_metrics[
                                        "minimum_inter_robot_distance_m"
                                    ]
                                ),
                                "joint_diversity": joint_metrics,
                                "traversed_region_ids": traversed_regions,
                                "unique_traversed_region_count": len(
                                    {
                                        region
                                        for values in traversed_regions.values()
                                        for region in values
                                    }
                                ),
                                "nested_trajectory_sets": nested,
                            }
                            regime = str(
                                source_metrics["observation_regime"]
                            )
                            score = regime_trajectory_soft_score(
                                joint_metrics,
                                regime,
                                trajectory_config[
                                    "regime_coverage_saturation_m2"
                                ],
                                heading_prior_weights=trajectory_config[
                                    "regime_initial_heading_prior_weights"
                                ],
                                view_connectivity_weights=trajectory_config[
                                    "regime_view_connectivity_weights"
                                ],
                            )
                            ranked.append((float(score), cloned, metrics))
        ranked.sort(key=lambda item: -item[0])
        return tuple(
            (trajectories, metrics)
            for _, trajectories, metrics in ranked[:maximum_candidates]
        )

    def trajectory_traversability_inspection(self, floor_index: int) -> dict[str, Any]:
        """Expose footprint-safe navigation, regions, footprint, and RouteBank."""
        scene = self._require_scene()
        trav_map = scene.trav_map
        context = self._navigation_contexts.get(floor_index)
        if context is None:
            world_xy, _, _, _ = self._trajectory_traversability(
                floor_index, self._env.robots[0]
            )
            traversable = np.zeros(
                tuple(trav_map.floor_map[floor_index].shape), dtype=np.uint8
            )
            pixels = self._world_to_map_preserving_batch(trav_map, world_xy)
            traversable[pixels[:, 0], pixels[:, 1]] = 1
            return {
                "traversable": traversable,
                "map_resolution_m": float(trav_map.map_resolution),
                "map_size": int(trav_map.map_size),
            }
        return {
            "traversable": context.planner_mask.astype(np.uint8),
            "point_traversability": context.point_free_mask.astype(np.uint8),
            "any_yaw_navigable": np.any(
                context.footprint_safe_masks, axis=0
            ).astype(np.uint8),
            "yaw_freedom": np.mean(
                context.footprint_safe_masks, axis=0
            ).astype(np.float32),
            "orientation_safe_masks": context.footprint_safe_masks.astype(np.uint8),
            "map_resolution_m": float(trav_map.map_resolution),
            "map_size": int(trav_map.map_size),
            "region_label_grid": context.region_label_grid,
            "region_graph": context.region_graph.metadata(),
            "footprint_polygon_xy": context.footprint.polygon_xy,
            "route_bank_paths_xy": [
                route.trajectory.base_to_world[:, :2, 3]
                for route in context.route_bank
            ],
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

    def robot_depth_observations(self) -> dict[str, dict[str, Any]]:
        """Capture mounted-sensor depth and its exact pose without shared AOV work."""
        self._require_scene()
        result: dict[str, dict[str, Any]] = {}
        for robot in self._env.robots:
            sensors = [
                sensor for sensor in robot.sensors.values()
                if isinstance(sensor, self._vision_sensor_type)
            ]
            if len(sensors) != 1:
                raise SimulatorUnavailableError(
                    f"Expected one VisionSensor on {robot.name}"
                )
            sensor = sensors[0]
            observation, _ = sensor.get_obs()
            if "depth_linear" not in observation:
                raise SimulatorUnavailableError(
                    f"Mounted camera is missing depth_linear on {robot.name}"
                )
            depth = self._reshape_vision_observation(
                sensor, "depth_linear", observation["depth_linear"]
            )
            usd_camera_to_world = self._pose_matrix(sensor)
            camera_to_world = validate_transform(
                usd_camera_to_world @ np.diag([1.0, -1.0, -1.0, 1.0])
            )
            result[robot.name] = {
                "depth_linear": depth,
                "camera_to_world": camera_to_world,
            }
        return result

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
                # RGB / depth from the robot-owned mounted sensor are current,
                # but Replicator AOVs on the movable shared sensor lag by one
                # observation after a pose change. Render ticks alone do not flush it.
                self._get_final_robot_capture_observation(observation_sensor)
                aov_flush_render_ticks = 4
                for _ in range(aov_flush_render_ticks):
                    self._og.sim.render()
                self._runtime_findings["final_robot_per_pose_aov_flush"] = (
                    "discard_full_observation_then_render"
                )
                self._runtime_findings[
                    "final_robot_per_pose_aov_flush_render_ticks"
                ] = aov_flush_render_ticks
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
                observation, info = self._publicize_observation_labels(observation, info)
            mounted_camera_to_world = validate_transform(
                requested_usd_camera_to_world @ np.diag([1.0, -1.0, -1.0, 1.0])
            )
            capture_camera_to_world = validate_transform(
                usd_camera_to_world @ np.diag([1.0, -1.0, -1.0, 1.0])
            )
            # RGB/depth are produced by the mounted sensor. Public trajectory
            # camera poses therefore follow that sensor, while AOV producer
            # poses are retained separately below.
            camera_to_world = mounted_camera_to_world
            base_to_world = self._pose_matrix(robot)
            camera_to_base = np.linalg.inv(base_to_world) @ mounted_camera_to_world
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
                "mounted_camera_to_world": mounted_camera_to_world,
                "capture_camera_to_world": capture_camera_to_world,
                "modality_camera_to_world": {
                    "rgb": mounted_camera_to_world,
                    "depth_linear": mounted_camera_to_world,
                    "seg_semantic": capture_camera_to_world,
                    "seg_instance": capture_camera_to_world,
                    "normal": capture_camera_to_world,
                },
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
        """Map leaf renderer IDs to semantic parent paths and classes once per graph."""
        if self._final_robot_renderer_label_cache is not None:
            cache_hits = int(
                self._runtime_findings.get(
                    "final_robot_renderer_mapping_cache_hits", 0
                )
            )
            self._runtime_findings["final_robot_renderer_mapping_cache_hits"] = (
                cache_hits + 1
            )
            return self._final_robot_renderer_label_cache
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
        self._final_robot_renderer_label_cache = (id_to_path, id_to_semantic)
        return self._final_robot_renderer_label_cache

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

    def _publicize_observation_labels(
        self, observation: dict[str, Any], info: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Replace transient renderer IDs with Dataset-v1.1 public IDs."""
        instance = observation.get("seg_instance_id", observation.get("seg_instance"))
        renderer_info = info.get("seg_instance_id", info.get("seg_instance", {}))
        if instance is None or not isinstance(renderer_info, dict):
            return observation, info
        catalog = self._public_label_catalog_cache
        if catalog is None:
            catalog = self.object_catalog()
            self._public_label_catalog_cache = catalog
            self._runtime_findings["public_label_catalog_cache_size"] = len(catalog)
        robot_paths = {robot.name: str(robot.prim_path) for robot in self._env.robots}
        public_instance, public_semantic, mapping = remap_public_labels(
            self._native_value(instance), renderer_info, catalog, robot_paths
        )
        like = instance
        # NumPy 2.x arrays expose ``device``; explicitly recognize tensors.
        torch_module = getattr(self, "_th", None)
        tensor_type = getattr(torch_module, "Tensor", ()) if torch_module is not None else ()
        if tensor_type and isinstance(like, tensor_type):
            public_instance_value = self._th.as_tensor(public_instance, device=like.device)
            public_semantic_value = self._th.as_tensor(public_semantic, device=like.device)
        else:
            public_instance_value = public_instance
            public_semantic_value = public_semantic
        observation["seg_instance_id"] = public_instance_value
        observation["seg_instance"] = public_instance_value
        observation["seg_semantic"] = public_semantic_value
        state_by_id = {obj.instance_id: obj for obj in catalog}
        public_info = {"0": "BACKGROUND"}
        semantic_info = {"0": {"class": "background"}, "1": {"class": "unknown"}, "2": {"class": "robot"}}
        for public_id, state_id in mapping["public_instance_to_state"].items():
            public_info[public_id] = state_by_id[state_id].native_path
            semantic_info[str(3 + (__import__("zlib").crc32(state_by_id[state_id].category.encode("utf-8")) & 0x3FFFFFFF))] = {"class": state_by_id[state_id].category}
        for public_id, robot_id in mapping["reserved_robot_instances"].items():
            public_info[public_id] = robot_paths[robot_id]
        info["seg_instance_id"] = public_info
        info["seg_instance"] = public_info
        info["seg_semantic"] = semantic_info
        info["mvwd_public_id_mapping"] = mapping
        self._runtime_findings["public_instance_mapping_source"] = (
            "renderer_id->native_path->ObjectState.instance_id->public_integer"
        )
        return observation, info

    def _get_final_robot_capture_observation(
        self, camera: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        observation, info = camera.get_obs()
        segmentation, segmentation_info = (
            self._final_robot_segmentation_observation(camera)
        )
        observation.update(segmentation)
        info.update(segmentation_info)
        return self._publicize_observation_labels(observation, info)

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
        self._final_robot_renderer_label_cache = None
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
                # OmniGibson 3.9.2 does not flush a projection / pose change on
                # this persistent Replicator render product through bare render
                # ticks. The first observation can therefore still be the prior
                # robot-perspective view even though USD reports orthographic.
                # Consume that stale observation, then render the authoritative
                # top-down frame before saving any modality.
                self._get_final_robot_capture_observation(camera)
                projection_flush_render_ticks = 4
                for _ in range(projection_flush_render_ticks):
                    self._og.sim.render()
                flushes = int(
                    self._runtime_findings.get(
                        "environment_bev_projection_flush_count", 0
                    )
                )
                self._runtime_findings["environment_bev_projection_flush_count"] = (
                    flushes + 1
                )
                self._runtime_findings[
                    "environment_bev_projection_flush_render_ticks"
                ] = projection_flush_render_ticks
                observation, info = self._get_final_robot_capture_observation(
                    camera
                )
            else:
                observation, info = camera.get_obs()
                observation, info = self._publicize_observation_labels(observation, info)
            arrays: dict[str, np.ndarray] = {}
            for public_name in sensor_modalities:
                backend_name = backend_names.get(public_name, public_name)
                arrays[public_name] = canonicalize_public_modality(
                    public_name,
                    self._resample_bev_observation(
                        camera, backend_name, observation[backend_name], calibration
                    ),
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
        floors = self._robot_support_surfaces()
        self._preflight_trajectory_contacts(by_id, robots, floors, frames)
        hidden: list[tuple[Any, bool]] = []
        world_frames: dict[str, list[np.ndarray]] = {name: [] for name in world_modalities}
        robot_frames: dict[str, dict[str, list[np.ndarray]]] = {
            robot_id: {name: [] for name in ("rgb", "depth_linear", "semantic", "instance", "normal")}
            for robot_id in sorted(robots)
        }
        actual_bases: dict[str, list[np.ndarray]] = {robot_id: [] for robot_id in robots}
        actual_cameras: dict[str, list[np.ndarray]] = {robot_id: [] for robot_id in robots}
        capture_cameras: dict[str, list[np.ndarray]] = {robot_id: [] for robot_id in robots}
        capture_translation_errors: dict[str, list[float]] = {robot_id: [] for robot_id in robots}
        capture_rotation_errors: dict[str, list[float]] = {robot_id: [] for robot_id in robots}
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
        maximum_world_bev_camera_pose_error = 0.0
        world_bev_projection_orthographic = True
        maximum_world_bev_occupancy_fraction = 0.0
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
            world_bev_capture_position = self._th.tensor(
                [(xmin + xmax) / 2, (ymin + ymax) / 2, camera_z]
            )
            world_bev_capture_orientation = self._th.tensor(
                [0.0, 0.0, 0.0, 1.0]
            )
            camera.set_position_orientation(
                position=world_bev_capture_position,
                orientation=world_bev_capture_orientation,
            )
            hidden = self._set_visible(ceilings, False)
            if created_camera:
                camera.initialize()
                self._refresh_physics_handles_after_sensor_edit()
            intended_world_bev_camera_to_world = self._pose_matrix(camera)
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
                if (
                    frame_index == 0
                    or (frame_index + 1) % 10 == 0
                    or frame_index == frames - 1
                ):
                    print(
                        f"[mvwd] world BEV frame {frame_index + 1}/{frames}",
                        flush=True,
                    )
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
                # Keep the shared render product in orthographic mode for the
                # complete world pass; projection changes are not flushed by
                # render ticks alone in OmniGibson 3.9.2.
                self._og.sim.render()
                actual_world_bev_camera_to_world = self._pose_matrix(camera)
                maximum_world_bev_camera_pose_error = max(
                    maximum_world_bev_camera_pose_error,
                    float(
                        np.linalg.norm(
                            actual_world_bev_camera_to_world[:3, 3]
                            - intended_world_bev_camera_to_world[:3, 3]
                        )
                    ),
                    float(
                        rotation_angle(
                            actual_world_bev_camera_to_world,
                            intended_world_bev_camera_to_world,
                        )
                    ),
                )
                projection = str(
                    self._lazy.pxr.UsdGeom.Camera(camera.prim)
                    .GetProjectionAttr()
                    .Get()
                )
                world_bev_projection_orthographic &= projection == "orthographic"
                if self._using_final_robot:
                    world_observation, world_info = (
                        self._get_final_robot_capture_observation(camera)
                    )
                else:
                    world_observation, world_info = camera.get_obs()
                    world_observation, world_info = self._publicize_observation_labels(
                        world_observation, world_info
                    )
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
                    maximum_world_bev_occupancy_fraction = max(
                        maximum_world_bev_occupancy_fraction,
                        float(np.mean(frame_arrays["occupancy"])),
                    )
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
            # Switch projection only once, between the complete world and ego
            # passes. A full observation read is required to flush Replicator's
            # cached orthographic render product; bare render ticks are insufficient.
            if self._using_final_robot:
                self.robot_observations()
            self._runtime_findings["rollout_capture_schedule"] = (
                "all_world_bev_frames_then_all_robot_view_frames"
            )
            for frame_index in range(frames):
                if (
                    frame_index == 0
                    or (frame_index + 1) % 10 == 0
                    or frame_index == frames - 1
                ):
                    print(
                        f"[mvwd] ego frame {frame_index + 1}/{frames}",
                        flush=True,
                    )
                for robot_id, robot in robots.items():
                    planned_base = by_id[robot_id].base_to_world[frame_index]
                    position, orientation = self._transform_utils.mat2pose(
                        self._th.as_tensor(planned_base, dtype=self._th.float32)
                    )
                    robot.set_position_orientation(
                        position=position, orientation=orientation
                    )
                    robot.keep_still()
                self._og.sim.step_physics()
                for robot_id, robot in robots.items():
                    planned_base = by_id[robot_id].base_to_world[frame_index]
                    position, orientation = self._transform_utils.mat2pose(
                        self._th.as_tensor(planned_base, dtype=self._th.float32)
                    )
                    robot.set_position_orientation(
                        position=position, orientation=orientation
                    )
                    self._restore_final_robot_mast_mount(robot)
                    robot.keep_still()
                self._og.sim.render()
                observations = self.robot_observations()
                for robot_id, record in observations.items():
                    planned = by_id[robot_id]
                    actual_bases[robot_id].append(record["base_to_world"])
                    actual_cameras[robot_id].append(record["camera_to_world"])
                    capture_cameras[robot_id].append(record["capture_camera_to_world"])
                    capture_translation_errors[robot_id].append(
                        float(record["capture_pose_translation_error_m"])
                    )
                    capture_rotation_errors[robot_id].append(
                        float(record["capture_pose_rotation_error_rad"])
                    )
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
                    rgb = canonicalize_public_modality(
                        "rgb", np.asarray(modalities["rgb"])
                    )
                    robot_frames[robot_id]["rgb"].append(rgb)
                    for backend_name, public_name in (
                        ("depth_linear", "depth_linear"),
                        ("seg_semantic", "semantic"),
                        ("seg_instance", "instance"),
                        ("normal", "normal"),
                    ):
                        values = canonicalize_public_modality(
                            public_name, np.asarray(modalities[backend_name])
                        )
                        robot_frames[robot_id][public_name].append(values[::2, ::2])
                    depth = np.asarray(modalities["depth_linear"]).squeeze()
                    valid = (
                        np.isfinite(depth)
                        & (depth >= float(self.config["camera"]["near_m"]))
                        & (depth <= float(self.config["camera"]["far_m"]))
                    )
                    minimum_depth_valid_ratio = min(
                        minimum_depth_valid_ratio, float(valid.mean())
                    )
            multimodal_alignment = robot_multimodal_alignment_metrics(
                robot_frames,
                keyframe_count=int(
                    self.config["camera"].get(
                        "multimodal_alignment_keyframe_count", 7
                    )
                ),
            )
            minimum_multimodal_alignment_fraction = float(
                self.config["camera"].get(
                    "minimum_multimodal_alignment_fraction", 0.75
                )
            )
            multimodal_alignment_passed = all(
                metrics["own_view_best_fraction"]
                >= minimum_multimodal_alignment_fraction
                for metrics in multimodal_alignment["modalities"].values()
            )
            maximum_capture_translation_error = max(
                (max(values, default=0.0) for values in capture_translation_errors.values()),
                default=0.0,
            )
            maximum_capture_rotation_error = max(
                (max(values, default=0.0) for values in capture_rotation_errors.values()),
                default=0.0,
            )
            self._runtime_findings["robot_multimodal_view_alignment"] = (
                multimodal_alignment
            )
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
                "robot_multimodal_view_alignment": multimodal_alignment_passed,
                "capture_sensor_translation_alignment": (
                    maximum_capture_translation_error
                    <= float(self.config["camera"]["maximum_capture_translation_error_m"])
                ),
                "capture_sensor_rotation_alignment": (
                    maximum_capture_rotation_error
                    <= float(self.config["camera"]["maximum_capture_rotation_error_rad"])
                ),
                "world_bev_each_robot_visible": minimum_robot_pixels >= 4,
                "world_bev_robot_mask_projection": maximum_robot_mask_error <= mask_tolerance,
                "world_bev_camera_pose_stable": (
                    maximum_world_bev_camera_pose_error
                    <= max(position_tolerance, rotation_tolerance)
                ),
                "world_bev_projection_orthographic": world_bev_projection_orthographic,
                "world_bev_occupancy_not_saturated": (
                    maximum_world_bev_occupancy_fraction
                    < float(self.config["bev"]["maximum_occupancy_fraction"])
                ),
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
                        "robot_multimodal_alignment": multimodal_alignment,
                        "minimum_multimodal_alignment_fraction": (
                            minimum_multimodal_alignment_fraction
                        ),
                        "maximum_capture_translation_error_m": (
                            maximum_capture_translation_error
                        ),
                        "maximum_capture_rotation_error_rad": (
                            maximum_capture_rotation_error
                        ),
                        "minimum_robot_pixels": minimum_robot_pixels,
                        "maximum_robot_mask_error_m": maximum_robot_mask_error,
                        "maximum_world_bev_camera_pose_error": (
                            maximum_world_bev_camera_pose_error
                        ),
                        "world_bev_projection_orthographic": (
                            world_bev_projection_orthographic
                        ),
                        "maximum_world_bev_occupancy_fraction": (
                            maximum_world_bev_occupancy_fraction
                        ),
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
                    **{
                        name: np.stack(values)
                        for name, values in world_frames.items()
                    },
                    "calibration_world_bounds": np.asarray(
                        calibration.world_bounds
                    ),
                    "calibration_pixel_to_world": (
                        calibration.pixel_to_world_transform
                    ),
                    "calibration_world_to_pixel": (
                        calibration.world_to_pixel_transform
                    ),
                    "calibration_meters_per_pixel": np.asarray(
                        calibration.meters_per_pixel
                    ),
                    "calibration_floor_z": np.asarray(calibration.floor_z),
                },
                "robot_views": {
                    robot_id: {
                        name: np.stack(values) for name, values in modalities.items()
                    }
                    for robot_id, modalities in robot_frames.items()
                },
                "actual_trajectories": actual_trajectories,
                "camera_capture_metadata": {
                    robot_id: {
                        "mounted_camera_to_world": np.stack(actual_cameras[robot_id]),
                        "capture_camera_to_world": np.stack(capture_cameras[robot_id]),
                        "capture_pose_translation_error_m": np.asarray(capture_translation_errors[robot_id]),
                        "capture_pose_rotation_error_rad": np.asarray(capture_rotation_errors[robot_id]),
                    }
                    for robot_id in sorted(robots)
                },
                "checks": checks,
                "metrics": {
                    "maximum_base_pose_error": maximum_base_pose_error,
                    "maximum_camera_pose_error": maximum_camera_pose_error,
                    "minimum_depth_valid_ratio": minimum_depth_valid_ratio,
                    "robot_multimodal_alignment": multimodal_alignment,
                    "minimum_multimodal_alignment_fraction": (
                        minimum_multimodal_alignment_fraction
                    ),
                    "maximum_capture_translation_error_m": (
                        maximum_capture_translation_error
                    ),
                    "maximum_capture_rotation_error_rad": (
                        maximum_capture_rotation_error
                    ),
                    "minimum_robot_pixels": int(minimum_robot_pixels),
                    "maximum_robot_mask_error_m": maximum_robot_mask_error,
                    "maximum_world_bev_camera_pose_error": (
                        maximum_world_bev_camera_pose_error
                    ),
                    "world_bev_projection_orthographic": (
                        world_bev_projection_orthographic
                    ),
                    "maximum_world_bev_occupancy_fraction": (
                        maximum_world_bev_occupancy_fraction
                    ),
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
