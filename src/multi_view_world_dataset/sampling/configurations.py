from __future__ import annotations

import hashlib
import json
from typing import Iterable

import numpy as np

from multi_view_world_dataset.cameras.transforms import rotation_angle
from multi_view_world_dataset.schema.records import ObjectState
from multi_view_world_dataset.utils.serialization import to_jsonable


def exact_state_hash(objects: Iterable[ObjectState], decimals: int = 8) -> str:
    """Stable hash independent of object iteration order and native Python object identity."""
    records = []
    for obj in sorted(objects, key=lambda item: item.instance_id):
        record = to_jsonable(obj)
        record["object_to_world"] = np.round(obj.object_to_world, decimals).tolist()
        record["joint_values"] = np.round(np.asarray(obj.joint_values), decimals).tolist()
        records.append(record)
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def near_duplicate_configuration(
    candidate: Iterable[ObjectState],
    accepted: Iterable[ObjectState],
    *,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
) -> bool:
    left = {obj.instance_id: obj for obj in candidate if obj.movable}
    right = {obj.instance_id: obj for obj in accepted if obj.movable}
    if left.keys() != right.keys():
        return False
    rotation_threshold = np.deg2rad(rotation_threshold_deg)
    for instance_id, obj in left.items():
        other = right[instance_id]
        translation = np.linalg.norm(obj.object_to_world[:3, 3] - other.object_to_world[:3, 3])
        if translation > translation_threshold_m or rotation_angle(obj.object_to_world, other.object_to_world) > rotation_threshold:
            return False
        if obj.joint_values != other.joint_values or obj.semantic_states != other.semantic_states:
            return False
    return True

