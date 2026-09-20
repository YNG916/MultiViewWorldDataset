#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import traceback
from typing import Any

import numpy as np

from multi_view_world_dataset.adapters.omnigibson import OmniGibsonAdapter
from multi_view_world_dataset.assets import ROBOT_APPEARANCE_VARIANTS
from multi_view_world_dataset.rendering.inspection import save_rgb
from multi_view_world_dataset.utils.config import load_yaml_config
from multi_view_world_dataset.utils.runtime import resolve_runtime_paths
from multi_view_world_dataset.utils.serialization import dump_json


def _place_validation_robots(
    adapter: OmniGibsonAdapter,
    seed: int,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Place three robots far apart on the current footprint-eroded map."""
    robots = sorted(adapter._env.robots, key=lambda item: item.name)
    floor_index = 0
    world_xy, _, _, _ = adapter._trajectory_traversability(
        floor_index, robots[0]
    )
    if len(world_xy) < len(robots):
        raise RuntimeError("insufficient eroded traversability for asset validation")
    rng = np.random.default_rng(seed)
    chosen = [np.asarray(world_xy[int(rng.integers(len(world_xy)))])]
    while len(chosen) < len(robots):
        distances = np.stack(
            [np.linalg.norm(world_xy - point, axis=1) for point in chosen]
        )
        candidate_index = int(np.argmax(np.min(distances, axis=0)))
        chosen.append(np.asarray(world_xy[candidate_index]))
    minimum_separation = min(
        float(np.linalg.norm(left - right))
        for index, left in enumerate(chosen)
        for right in chosen[index + 1:]
    )
    if minimum_separation < float(
        adapter.config["placement"]["minimum_pairwise_distance_m"]
    ):
        raise RuntimeError(
            f"validation placement separation is too small: {minimum_separation}"
        )

    floor_z = float(adapter._require_scene().get_floor_height(floor_index))
    orientation = adapter._transform_utils.euler2quat(
        adapter._th.tensor([0.0, 0.0, 0.0])
    )
    initial_heights: dict[str, float] = {}
    positions: dict[str, list[float]] = {}
    for robot, xy in zip(robots, chosen, strict=True):
        position = np.asarray([xy[0], xy[1], floor_z], dtype=np.float32)
        robot.set_position_orientation(
            position=adapter._th.as_tensor(position),
            orientation=orientation,
        )
        joint = next(
            joint
            for name, joint in robot.joints.items()
            if name.endswith("mvwd_mast_joint")
        )
        joint.set_pos(0.0, drive=False)
        mount = np.eye(4)
        mount[2, 3] = 0.8
        adapter._development_camera_mounts[robot.name] = mount
        robot.keep_still()
        initial_heights[robot.name] = 0.8
        positions[robot.name] = position.tolist()
    for _ in range(4):
        for robot in robots:
            robot.keep_still()
        adapter._og.sim.step_physics()
    for robot, xy in zip(robots, chosen, strict=True):
        robot.set_position_orientation(
            position=adapter._th.as_tensor(
                [float(xy[0]), float(xy[1]), floor_z]
            ),
            orientation=orientation,
        )
        adapter._restore_final_robot_mast_mount(robot)
        robot.keep_still()
    adapter._og.sim.step_physics()
    floor_supports = adapter._robot_support_surfaces()
    for robot in robots:
        contact_pairs = adapter._external_robot_contact_pairs(
            robot, floor_supports
        )
        if contact_pairs:
            raise RuntimeError(
                f"{robot.name} validation placement collides: {contact_pairs[:10]}"
            )
    return initial_heights, positions


def _authored_visual_bounds(prim: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return local axis-aligned bounds for an unrotated project visual."""
    translate_attr = prim.GetAttribute("xformOp:translate")
    scale_attr = prim.GetAttribute("xformOp:scale")
    translate_value = translate_attr.Get() if translate_attr.IsValid() else None
    scale_value = scale_attr.Get() if scale_attr.IsValid() else None
    center = np.asarray(
        translate_value if translate_value is not None else (0.0, 0.0, 0.0),
        dtype=np.float64,
    )
    scale = np.abs(
        np.asarray(
            scale_value if scale_value is not None else (1.0, 1.0, 1.0),
            dtype=np.float64,
        )
    )
    if prim.GetTypeName() == "Cube":
        size = float(prim.GetAttribute("size").Get())
        half_extent = 0.5 * size * scale
    elif prim.GetTypeName() == "Cylinder":
        if str(prim.GetAttribute("axis").Get()) != "Z":
            raise RuntimeError(f"unexpected chassis cylinder axis: {prim.GetPath()}")
        radius = float(prim.GetAttribute("radius").Get())
        height = float(prim.GetAttribute("height").Get())
        half_extent = np.asarray(
            [radius * scale[0], radius * scale[1], 0.5 * height * scale[2]]
        )
    else:
        raise RuntimeError(
            f"unsupported authored chassis visual type: {prim.GetTypeName()}"
        )
    return center - half_extent, center + half_extent


def _validate_chassis_visuals(
    adapter: OmniGibsonAdapter,
    stage: Any,
    root: str,
) -> dict[str, Any]:
    relative_paths = (
        "integration_base_plate",
        "lower_chassis_center",
        "lower_chassis_front_cap",
        "lower_chassis_rear_cap",
        "chassis_identity_deck",
        "chassis_identity_left",
        "chassis_identity_right",
    )
    rig_root = f"{root}/chassis_link/mvwd_sensor_rig"
    bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in relative_paths:
        prim = stage.GetPrimAtPath(f"{rig_root}/{name}")
        if not prim.IsValid():
            raise RuntimeError(f"lower chassis visual is missing: {prim.GetPath()}")
        if any(
            "PhysicsCollisionAPI" in schema
            for schema in prim.GetAppliedSchemas()
        ):
            raise RuntimeError(f"lower chassis visual gained collision API: {name}")
        bounds[name] = _authored_visual_bounds(prim)

    material_opacity: dict[str, float] = {}
    for material_name in (
        "body_shell",
        "trim_dark",
        "lens",
        "accent_orange",
        "accent_blue",
        "accent_green",
    ):
        shader = stage.GetPrimAtPath(f"{root}/Looks/{material_name}/Shader")
        if not shader.IsValid():
            raise RuntimeError(f"visual material shader is missing: {material_name}")
        opacity = shader.GetAttribute("inputs:opacity").Get()
        if opacity is None or abs(float(opacity) - 1.0) > 1.0e-7:
            raise RuntimeError(
                f"visual material {material_name} is not explicitly opaque: {opacity}"
            )
        material_opacity[material_name] = float(opacity)

    shell_min = np.min(np.stack([value[0] for value in bounds.values()]), axis=0)
    shell_max = np.max(np.stack([value[1] for value in bounds.values()]), axis=0)
    robot = adapter._env.robots[0]
    world_to_base = np.linalg.inv(adapter._pose_matrix(robot))
    disabled = set(getattr(robot, "disabled_collision_link_names", ()))
    collision_points = []
    collision_links = []
    for link_name, link in sorted(robot.links.items()):
        if (
            link_name in disabled
            or not bool(getattr(link, "has_collision_meshes", False))
        ):
            continue
        points = getattr(link, "collision_boundary_points_world", None)
        if points is None:
            continue
        points = adapter._native_value(points).astype(np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            continue
        local = np.column_stack((points, np.ones(len(points)))) @ world_to_base.T
        collision_points.append(local[:, :3])
        collision_links.append(str(link_name))
    if not collision_points:
        raise RuntimeError("robot exposes no collision boundary points")
    collision = np.concatenate(collision_points)
    collision_min = collision.min(axis=0)
    collision_max = collision.max(axis=0)
    tolerance_m = 0.015
    if np.any(shell_min[:2] < collision_min[:2] - tolerance_m) or np.any(
        shell_max[:2] > collision_max[:2] + tolerance_m
    ):
        raise RuntimeError(
            "visual lower chassis exceeds the existing collision XY envelope: "
            f"visual={shell_min[:2], shell_max[:2]}, "
            f"collision={collision_min[:2], collision_max[:2]}"
        )

    body_names = (
        "lower_chassis_center",
        "lower_chassis_front_cap",
        "lower_chassis_rear_cap",
    )
    body_min = np.min(np.stack([bounds[name][0] for name in body_names]), axis=0)
    body_max = np.max(np.stack([bounds[name][1] for name in body_names]), axis=0)
    plate_min, plate_max = bounds["integration_base_plate"]
    deck_min, _ = bounds["chassis_identity_deck"]
    body_plate_overlap = float(body_max[2] - plate_min[2])
    plate_deck_gap = float(deck_min[2] - plate_max[2])
    if body_plate_overlap <= 0.0:
        raise RuntimeError("lower chassis body does not overlap its mounting plate")
    if abs(plate_deck_gap) > 1.0e-6:
        raise RuntimeError(
            f"identity deck is not seated on the mounting plate: {plate_deck_gap}"
        )
    return {
        "visual_prim_paths": [f"{rig_root}/{name}" for name in relative_paths],
        "all_visual_prims_collision_free": True,
        "material_opacity": material_opacity,
        "visual_envelope_local_m": {
            "minimum": shell_min.tolist(),
            "maximum": shell_max.tolist(),
        },
        "body_envelope_local_m": {
            "minimum": body_min.tolist(),
            "maximum": body_max.tolist(),
        },
        "native_collision_envelope_local_m": {
            "minimum": collision_min.tolist(),
            "maximum": collision_max.tolist(),
            "source_links": collision_links,
        },
        "body_plate_overlap_m": body_plate_overlap,
        "plate_deck_gap_m": plate_deck_gap,
    }


def _usd_look_at(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Build a USD camera transform whose local -Z axis looks at target."""
    forward = np.asarray(target, dtype=np.float64) - np.asarray(
        position, dtype=np.float64
    )
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((right, up, -forward))
    transform[:3, 3] = position
    return transform


def _render_chassis_validation_views(
    adapter: OmniGibsonAdapter,
    output_root: Path,
) -> dict[str, list[str]]:
    """Render unobstructed close views without changing robot physics."""
    camera = adapter._final_robot_capture_sensor
    if camera is None:
        raise RuntimeError("final robot capture sensor is unavailable")
    adapter._configure_final_robot_ego_capture(camera)
    stage = adapter._og.sim.stage
    pxr = adapter._lazy.pxr
    robot_paths = {str(robot.prim_path) for robot in adapter._env.robots}
    with adapter._og.sim.editing_usd():
        for obj in adapter._require_scene().objects:
            if str(getattr(obj, "category", "")) == "floors":
                continue
            path = str(obj.prim_path)
            if path in robot_paths:
                continue
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                pxr.UsdGeom.Imageable(prim).MakeInvisible()
        light = pxr.UsdLux.DistantLight.Define(
            stage, "/World/mvwd_robot_asset_validation_light"
        )
        light.CreateIntensityAttr(2500.0)

    robots = sorted(adapter._env.robots, key=lambda item: item.name)
    for robot in robots:
        joint = next(
            joint
            for name, joint in robot.joints.items()
            if name.endswith("mvwd_mast_joint")
        )
        adapter._development_camera_mounts[robot.name][2, 3] = 0.8
        joint.set_pos(0.0, drive=False)
        adapter._restore_final_robot_mast_mount(robot)
        robot.keep_still()
    adapter._og.sim.step_physics()

    views = {
        "front_right": np.asarray([1.15, -1.15, 0.80]),
        "right_side": np.asarray([0.0, -1.35, 0.62]),
        "rear_left": np.asarray([-1.45, 1.00, 0.75]),
    }
    target_local = np.asarray([-0.23, 0.0, 0.30, 1.0])
    output: dict[str, list[str]] = {}
    view_root = output_root / "inspection" / "robot_asset_views"
    for robot in robots:
        with adapter._og.sim.editing_usd():
            for candidate in robots:
                imageable = pxr.UsdGeom.Imageable(
                    stage.GetPrimAtPath(str(candidate.prim_path))
                )
                if candidate is robot:
                    imageable.MakeVisible()
                else:
                    imageable.MakeInvisible()
        base_to_world = adapter._pose_matrix(robot)
        target_world = (base_to_world @ target_local)[:3]
        output[robot.name] = []
        for view_name, offset in views.items():
            camera_position = (
                base_to_world
                @ np.asarray([offset[0], offset[1], offset[2], 1.0])
            )[:3]
            camera_to_world = _usd_look_at(camera_position, target_world)
            position, orientation = adapter._transform_utils.mat2pose(
                adapter._th.as_tensor(
                    camera_to_world, dtype=adapter._th.float32
                )
            )
            camera.set_position_orientation(
                position=position, orientation=orientation
            )
            for _ in range(4):
                adapter._og.sim.render()
            observation, _ = camera.get_obs()
            rgb = adapter._reshape_vision_observation(
                camera, "rgb", observation["rgb"]
            )
            rgb = adapter._native_value(rgb)
            if float(np.std(rgb[..., :3])) < 2.0:
                raise RuntimeError(
                    f"asset validation view is blank: {robot.name}/{view_name}"
                )
            path = view_root / f"{robot.name}_{view_name}.png"
            save_rgb(path, rgb)
            output[robot.name].append(str(path.relative_to(output_root)))
    return output


def validate_robot_asset(
    config_path: Path,
    scene_id: str,
    output_root: Path,
    cache_root: Path,
) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    runtime = resolve_runtime_paths(
        config,
        output_root=output_root,
        cache_root=cache_root,
    )
    adapter = OmniGibsonAdapter(runtime, config)
    report: dict[str, Any] = {
        "status": "error",
        "scene_id": scene_id,
        "model": config["robot"]["final_model"],
    }
    try:
        adapter.start()
        adapter.load_scene(
            scene_id,
            robot_count=3,
            development_robot=str(config["robot"]["final_model"]),
        )
        initial_heights, validation_positions = _place_validation_robots(
            adapter, seed=20260919
        )
        stage = adapter._og.sim.stage
        appearances = adapter.runtime_report()["final_robot_appearance_variants"]
        expected_ids = tuple(ROBOT_APPEARANCE_VARIANTS)
        if tuple(sorted(appearances)) != expected_ids:
            raise RuntimeError(
                f"appearance identities differ: {sorted(appearances)} != {expected_ids}"
            )

        root = str(adapter._env.robots[0].prim_path)
        chassis_visual_qa = _validate_chassis_visuals(adapter, stage, root)
        outer = stage.GetPrimAtPath(
            f"{root}/chassis_link/mvwd_sensor_rig/tower_outer_shell"
        )
        sleeve = stage.GetPrimAtPath(f"{root}/mast_carriage/sliding_sleeve")
        top_cap = stage.GetPrimAtPath(
            f"{root}/mast_carriage/sensor_head_top_cap"
        )
        if not outer.IsValid() or not sleeve.IsValid() or not top_cap.IsValid():
            raise RuntimeError("shrouded mast visual prims are missing")
        top_cap_scale = [
            float(value)
            for value in top_cap.GetAttribute("xformOp:scale").Get()
        ]
        if top_cap_scale[0] < 0.18 or top_cap_scale[1] < 0.13:
            raise RuntimeError(
                f"BEV identity top cap is too small: {top_cap_scale}"
            )
        if any(
            "PhysicsCollisionAPI" in schema
            for schema in top_cap.GetAppliedSchemas()
        ):
            raise RuntimeError("BEV identity top cap must remain visual-only")
        outer_height = float(outer.GetAttribute("height").Get())
        outer_z = float(outer.GetAttribute("xformOp:translate").Get()[2])
        sleeve_height = float(sleeve.GetAttribute("height").Get())
        sleeve_z = float(sleeve.GetAttribute("xformOp:translate").Get()[2])
        fixed_upper = outer_z + 0.5 * outer_height
        front_visual_paths = {
            "housing": f"{root}/mast_carriage/sensor_head_housing",
            "lens": f"{root}/mast_carriage/front_lens",
            "direction_marker": f"{root}/mast_carriage/front_direction_marker",
        }
        front_visuals = {
            name: stage.GetPrimAtPath(path)
            for name, path in front_visual_paths.items()
        }
        invalid_front_paths = [
            front_visual_paths[name]
            for name, prim in front_visuals.items()
            if not prim.IsValid()
        ]
        if invalid_front_paths:
            raise RuntimeError(
                f"front-facing visual prims are missing: {invalid_front_paths}"
            )
        front_x = {
            name: float(prim.GetAttribute("xformOp:translate").Get()[0])
            for name, prim in front_visuals.items()
        }
        if front_x["lens"] <= front_x["housing"]:
            raise RuntimeError(
                f"camera lens is not on the +X front face: {front_x}"
            )

        cv_to_usd = np.diag([1.0, -1.0, -1.0, 1.0])
        baseline_rotations: dict[str, np.ndarray] = {}
        height_records: list[dict[str, Any]] = []
        minimum_overlap = float("inf")
        for height in (0.8, 1.0, 1.2, 1.4):
            extension = height - 0.8
            for robot in adapter._env.robots:
                joint = next(
                    joint
                    for name, joint in robot.joints.items()
                    if name.endswith("mvwd_mast_joint")
                )
                adapter._development_camera_mounts[robot.name][2, 3] = height
                joint.set_pos(extension, drive=False)
                robot.keep_still()
            adapter._og.sim.step_physics()
            for robot in adapter._env.robots:
                adapter._restore_final_robot_mast_mount(robot)
                robot.keep_still()
            adapter._og.sim.render()
            per_robot: dict[str, Any] = {}
            for robot in sorted(adapter._env.robots, key=lambda item: item.name):
                sensor = next(
                    sensor
                    for sensor in robot.sensors.values()
                    if str(sensor.prim_path).endswith("/dataset_camera")
                )
                base_to_world = adapter._pose_matrix(robot)
                camera_to_world = adapter._pose_matrix(sensor) @ cv_to_usd
                camera_to_base = np.linalg.inv(base_to_world) @ camera_to_world
                joint = next(
                    joint
                    for name, joint in robot.joints.items()
                    if name.endswith("mvwd_mast_joint")
                )
                joint_position = float(
                    adapter._native_value(joint.get_state()[0]).reshape(-1)[0]
                )
                if robot.name not in baseline_rotations:
                    baseline_rotations[robot.name] = camera_to_base[:3, :3].copy()
                rotation_error = float(
                    np.linalg.norm(
                        camera_to_base[:3, :3] - baseline_rotations[robot.name]
                    )
                )
                per_robot[robot.name] = {
                    "camera_height_m": float(camera_to_base[2, 3]),
                    "mast_joint_value_m": joint_position,
                    "camera_xy_m": camera_to_base[:2, 3].tolist(),
                    "rotation_drift_frobenius": rotation_error,
                }
                if abs(camera_to_base[2, 3] - height) > 1.0e-5:
                    raise RuntimeError(
                        f"{robot.name} camera height {camera_to_base[2, 3]} != {height}"
                    )
                if abs(joint_position - extension) > 1.0e-5:
                    raise RuntimeError(
                        f"{robot.name} mast joint {joint_position} != {extension}"
                    )
                if rotation_error > 1.0e-6:
                    raise RuntimeError(
                        f"{robot.name} camera orientation drifted by {rotation_error}"
                    )
            sleeve_lower = height + sleeve_z - 0.5 * sleeve_height
            overlap = fixed_upper - sleeve_lower
            minimum_overlap = min(minimum_overlap, overlap)
            if overlap <= 0.0:
                raise RuntimeError(
                    f"visible mast gap at height={height}: overlap={overlap}"
                )
            height_records.append({
                "requested_height_m": height,
                "mast_extension_m": extension,
                "shroud_overlap_m": overlap,
                "robots": per_robot,
            })
        rendered_views = _render_chassis_validation_views(adapter, output_root)

        report.update({
            "status": "pass",
            "appearances": appearances,
            "collision_free_initial_camera_heights_m": initial_heights,
            "validation_robot_positions_world_m": validation_positions,
            "height_records": height_records,
            "minimum_shroud_overlap_m": minimum_overlap,
            "camera_intrinsics_semantics": {
                "hfov_deg": float(config["camera"]["hfov_deg"]),
                "pitch_deg": float(config["camera"]["pitch_deg"]),
                "roll_deg": float(config["camera"]["roll_deg"]),
                "near_m": float(config["camera"]["near_m"]),
                "far_m": float(config["camera"]["far_m"]),
            },
            "visual_shell_collision_geometry_changed": False,
            "bev_identity_top_cap_scale_m": top_cap_scale,
            "front_visual_local_x_m": front_x,
            "lower_chassis_visual_qa": chassis_visual_qa,
            "rendered_chassis_views": rendered_views,
        })
        return report
    except BaseException as error:
        report.update({
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        })
        raise
    finally:
        dump_json(output_root / "robot_asset_validation.json", report)
        adapter.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load and validate the official final robot and all appearances."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/final_robot_preview.yaml"))
    parser.add_argument("--scene", default="Beechwood_0_int")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    result = validate_robot_asset(
        args.config, args.scene, args.output_root, args.cache_root
    )
    print(result)
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

