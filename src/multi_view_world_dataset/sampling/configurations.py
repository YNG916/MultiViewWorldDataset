from __future__ import annotations

import hashlib
import json
from typing import Iterable

import numpy as np

from multi_view_world_dataset.cameras.transforms import rotation_angle
from multi_view_world_dataset.errors import SampleRejected
from multi_view_world_dataset.schema.records import InterventionType, ObjectState
from multi_view_world_dataset.utils.serialization import to_jsonable


def exact_state_hash(objects: Iterable[ObjectState], decimals: int = 8) -> str:
    """Stable hash independent of object iteration order and native Python object identity."""
    records = []
    for obj in sorted(objects, key=lambda item: item.instance_id):
        record = to_jsonable(obj)
        record["object_to_world"] = np.round(obj.object_to_world, decimals).tolist()
        joint_values = np.asarray(obj.joint_values, dtype=np.float64)
        if not np.isfinite(joint_values).all():
            raise SampleRejected(
                "nonfinite_configuration_joint_values",
                {"instance_id": obj.instance_id},
            )
        record["joint_values"] = np.round(joint_values, decimals).tolist()
        records.append(record)
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def nonrigid_configuration_candidates(
    objects: Iterable[ObjectState],
) -> dict[str, tuple[InterventionType, ...]]:
    """Real fixed-base changes usable when no movable relation targets exist."""
    candidates: dict[str, tuple[InterventionType, ...]] = {}
    for obj in objects:
        if obj.structural:
            continue
        kinds: list[InterventionType] = []
        if (
            obj.articulated
            and len(obj.joint_limits) == len(obj.joint_values)
            and any(
                np.isfinite((lower, upper, current)).all() and upper > lower
                for (lower, upper), current in zip(
                    obj.joint_limits, obj.joint_values, strict=True
                )
            )
        ):
            kinds.append(InterventionType.ARTICULATION)
        if any(isinstance(value, (bool, np.bool_)) for value in obj.semantic_states.values()):
            kinds.append(InterventionType.STATE_CHANGE)
        if kinds:
            candidates[obj.instance_id] = tuple(kinds)
    return candidates


def near_duplicate_configuration(
    candidate: Iterable[ObjectState],
    accepted: Iterable[ObjectState],
    *,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
    include_nonrigid: bool = False,
) -> bool:
    candidate_objects = tuple(candidate)
    accepted_objects = tuple(accepted)
    left = {obj.instance_id: obj for obj in candidate_objects if obj.movable}
    right = {obj.instance_id: obj for obj in accepted_objects if obj.movable}
    if left.keys() != right.keys():
        return False
    rotation_threshold = np.deg2rad(rotation_threshold_deg)
    for instance_id, obj in left.items():
        other = right[instance_id]
        translation = np.linalg.norm(obj.object_to_world[:3, 3] - other.object_to_world[:3, 3])
        if translation > translation_threshold_m or rotation_angle(obj.object_to_world, other.object_to_world) > rotation_threshold:
            return False
        if not include_nonrigid and (
            obj.joint_values != other.joint_values
            or obj.semantic_states != other.semantic_states
        ):
            return False
    if include_nonrigid:
        candidate_by_id = {obj.instance_id: obj for obj in candidate_objects if not obj.structural}
        accepted_by_id = {obj.instance_id: obj for obj in accepted_objects if not obj.structural}
        if candidate_by_id.keys() != accepted_by_id.keys():
            return False
        for instance_id, obj in candidate_by_id.items():
            other = accepted_by_id[instance_id]
            translation = np.linalg.norm(obj.object_to_world[:3, 3] - other.object_to_world[:3, 3])
            if translation > translation_threshold_m or rotation_angle(obj.object_to_world, other.object_to_world) > rotation_threshold:
                return False
            if obj.semantic_states != other.semantic_states:
                return False
            if not np.allclose(obj.joint_values, other.joint_values, atol=1.0e-4, rtol=0.0):
                return False
    return True

