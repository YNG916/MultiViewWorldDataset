from __future__ import annotations

import zlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def stable_semantic_id(category: str) -> int:
    return 3 + (zlib.crc32(category.encode("utf-8")) & 0x3FFFFFFF)


def public_instance_catalog(objects: Sequence[Any]) -> dict[str, int]:
    return {
        obj.instance_id: index
        for index, obj in enumerate(sorted(objects, key=lambda item: item.instance_id), start=4)
    }


def remap_public_labels(
    instance_image: np.ndarray,
    renderer_info: Mapping[Any, Any],
    objects: Sequence[Any],
    robot_native_paths: Mapping[str, str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Map transient renderer labels through native paths to stable public IDs."""
    labels = np.asarray(instance_image).squeeze()
    instance_by_object = public_instance_catalog(objects)
    objects_by_path = sorted(objects, key=lambda item: len(item.native_path), reverse=True)
    robot_ids = {robot_id: index + 1 for index, robot_id in enumerate(sorted(robot_native_paths))}
    raw_to_instance: dict[int, int] = {0: 0}
    raw_to_semantic: dict[int, int] = {0: 0}
    raw_resolution: dict[str, Any] = {}
    for raw_key, raw_label in renderer_info.items():
        try:
            raw_id = int(raw_key)
        except (TypeError, ValueError):
            continue
        if isinstance(raw_label, Mapping):
            path = str(raw_label.get("path", raw_label.get("name", raw_label)))
        else:
            path = str(raw_label)
        public_instance = 0
        public_semantic = 1
        resolved_object_id = None
        for robot_id, robot_path in robot_native_paths.items():
            if path == robot_path or path.startswith(robot_path + "/") or robot_id in path:
                public_instance = robot_ids[robot_id]
                public_semantic = 2
                resolved_object_id = robot_id
                break
        if public_instance == 0:
            for obj in objects_by_path:
                if path == obj.native_path or path.startswith(obj.native_path + "/"):
                    public_instance = instance_by_object[obj.instance_id]
                    public_semantic = stable_semantic_id(obj.category)
                    resolved_object_id = obj.instance_id
                    break
        raw_to_instance[raw_id] = public_instance
        raw_to_semantic[raw_id] = public_semantic
        raw_resolution[str(raw_id)] = {
            "renderer_label": path,
            "resolved_state_id": resolved_object_id,
            "public_instance_id": public_instance,
            "public_semantic_id": public_semantic,
        }
    public_instance = np.zeros(labels.shape, dtype=np.uint32)
    public_semantic = np.zeros(labels.shape, dtype=np.uint32)
    for raw_id in np.unique(labels):
        raw = int(raw_id)
        mask = labels == raw_id
        public_instance[mask] = raw_to_instance.get(raw, 0)
        public_semantic[mask] = raw_to_semantic.get(raw, 1 if raw else 0)
    info = {
        "raw_to_public": raw_resolution,
        "public_instance_to_state": {
            str(value): key for key, value in instance_by_object.items()
        },
        "reserved_robot_instances": {
            str(robot_ids[key]): key for key in sorted(robot_ids)
        },
    }
    return public_instance, public_semantic, info
