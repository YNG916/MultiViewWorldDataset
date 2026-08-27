from __future__ import annotations

from dataclasses import replace
import uuid

import numpy as np

from multi_view_world_dataset.errors import SampleRejected
from multi_view_world_dataset.schema.records import ApplicationMode, InterventionEvent, InterventionType, ObjectState


def choose_intervention_type(weights: dict[str, float], rng: np.random.Generator) -> InterventionType:
    names = list(weights)
    probabilities = np.asarray([weights[name] for name in names], dtype=np.float64)
    probabilities /= probabilities.sum()
    return InterventionType(str(rng.choice(names, p=probabilities)))


def propose_rigid_relocation(
    target: ObjectState,
    rng: np.random.Generator,
    *,
    translation_range_m: tuple[float, float] = (0.3, 1.5),
    rotation_range_deg: tuple[float, float] = (30.0, 120.0),
) -> InterventionEvent:
    distance = float(rng.uniform(*translation_range_m))
    direction = float(rng.uniform(-np.pi, np.pi))
    yaw = float(np.deg2rad(rng.uniform(*rotation_range_deg)) * rng.choice([-1.0, 1.0]))
    return InterventionEvent(
        event_id=f"event_{uuid.UUID(int=int(rng.integers(0, 2**63)))}",
        intervention_type=InterventionType.RIGID_RELOCATION,
        target_instance_id=target.instance_id,
        application_mode=ApplicationMode.PRE_ROLLOUT,
        time_index=None,
        parameters={
            "translation_xy_m": [distance * np.cos(direction), distance * np.sin(direction)],
            "yaw_delta_rad": yaw,
            "preserve_floor_id": target.floor_id,
            "prefer_room_id": target.room_id,
            "preserve_relations": list(target.relations),
        },
        before_object_state=target,
    )



def eligible_intervention_targets(
    catalog: tuple[ObjectState, ...] | list[ObjectState],
    intervention_type: InterventionType,
) -> tuple[ObjectState, ...]:
    """Return targets that can support the requested atomic event at schema level."""
    if intervention_type is InterventionType.RIGID_RELOCATION:
        candidates = (obj for obj in catalog if obj.movable and not obj.structural)
    elif intervention_type is InterventionType.ARTICULATION:
        candidates = (
            obj
            for obj in catalog
            if obj.articulated
            and len(obj.joint_names) == len(obj.joint_values)
            and len(obj.joint_limits) == len(obj.joint_values)
        )
    elif intervention_type is InterventionType.STATE_CHANGE:
        candidates = (obj for obj in catalog if obj.semantic_states)
    else:
        candidates = iter(())
    return tuple(sorted(candidates, key=lambda obj: obj.instance_id))


def propose_articulation(
    target: ObjectState,
    rng: np.random.Generator,
    *,
    endpoint_margin_fraction: float = 0.10,
) -> InterventionEvent:
    """Propose a meaningful finite joint change; simulator validation remains mandatory."""
    if not 0 <= endpoint_margin_fraction < 0.5:
        raise ValueError("endpoint_margin_fraction must be in [0, 0.5)")
    usable: list[tuple[int, float, float, float]] = []
    for index, (limits, current) in enumerate(zip(target.joint_limits, target.joint_values, strict=True)):
        lower, upper = (float(value) for value in limits)
        if np.isfinite([lower, upper, current]).all() and upper > lower:
            usable.append((index, lower, upper, float(current)))
    if not target.articulated or not usable:
        raise SampleRejected("no_finite_articulation_target", {"instance_id": target.instance_id})
    index, lower, upper, current = usable[int(rng.integers(len(usable)))]
    span = upper - lower
    low_target = lower + endpoint_margin_fraction * span
    high_target = upper - endpoint_margin_fraction * span
    value = high_target if abs(high_target - current) >= abs(low_target - current) else low_target
    joint_values = list(target.joint_values)
    joint_values[index] = value
    after = replace(target, joint_values=tuple(joint_values))
    return InterventionEvent(
        event_id=f"event_{uuid.UUID(int=int(rng.integers(0, 2**63)))}",
        intervention_type=InterventionType.ARTICULATION,
        target_instance_id=target.instance_id,
        application_mode=ApplicationMode.PRE_ROLLOUT,
        time_index=None,
        parameters={
            "joint_name": target.joint_names[index],
            "joint_index": index,
            "value_before": current,
            "value_after": value,
            "joint_limits": [lower, upper],
        },
        before_object_state=target,
        after_object_state=after,
    )


def propose_state_change(
    target: ObjectState,
    rng: np.random.Generator,
) -> InterventionEvent:
    """Toggle one advertised boolean object state and preserve the exact before/after record."""
    if not target.semantic_states:
        raise SampleRejected("no_meaningful_state_target", {"instance_id": target.instance_id})
    state_name = sorted(target.semantic_states)[int(rng.integers(len(target.semantic_states)))]
    value_before = bool(target.semantic_states.get(state_name, False))
    value_after = not value_before
    semantic_states = dict(target.semantic_states)
    semantic_states[state_name] = value_after
    after = replace(target, semantic_states=semantic_states)
    return InterventionEvent(
        event_id=f"event_{uuid.UUID(int=int(rng.integers(0, 2**63)))}",
        intervention_type=InterventionType.STATE_CHANGE,
        target_instance_id=target.instance_id,
        application_mode=ApplicationMode.PRE_ROLLOUT,
        time_index=None,
        parameters={
            "state_name": state_name,
            "value_before": value_before,
            "value_after": value_after,
        },
        before_object_state=target,
        after_object_state=after,
    )
