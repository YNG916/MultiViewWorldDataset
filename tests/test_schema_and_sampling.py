from dataclasses import replace

import numpy as np
import pytest

from multi_view_world_dataset.sampling.configurations import exact_state_hash, near_duplicate_configuration
from multi_view_world_dataset.sampling.interventions import (
    eligible_intervention_targets,
    propose_articulation,
    propose_state_change,
)
from multi_view_world_dataset.sampling.splits import assign_scene_family_splits, infer_scene_family
from multi_view_world_dataset.sampling.trajectories import generate_smooth_trajectory, trajectories_equal
from multi_view_world_dataset.schema.records import (
    ApplicationMode,
    InterventionEvent,
    InterventionType,
    ObjectState,
)


def make_object(instance_id="obj_a", x=0.0):
    transform = np.eye(4)
    transform[0, 3] = x
    return ObjectState(
        instance_id=instance_id,
        asset_uid="asset",
        category="chair",
        native_path="/World/chair",
        structural=False,
        movable=True,
        articulated=False,
        available_states=(),
        object_to_world=transform,
        bbox_min_world=(-0.5, -0.5, 0),
        bbox_max_world=(0.5, 0.5, 1),
        scale=(1, 1, 1),
    )


def test_state_hash_is_order_independent_and_near_duplicate():
    a, b = make_object("a"), make_object("b", 1)
    assert exact_state_hash((a, b)) == exact_state_hash((b, a))
    shifted = replace(a, object_to_world=np.array([[1, 0, 0, 0.01], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    assert near_duplicate_configuration(
        (shifted, b), (a, b), translation_threshold_m=0.03, rotation_threshold_deg=3
    )


def test_scene_family_splits_are_disjoint():
    scenes = ["House_0_int", "House_1_int", "Other_0_garden", "school_lab"]
    assert infer_scene_family("House_1_int") == "House"
    splits = assign_scene_family_splits(scenes, {"train": 0.5, "val": 0.25, "test": 0.25}, seed=3)
    assert splits["House_0_int"] == splits["House_1_int"]


def test_smooth_trajectory_and_exact_pairing():
    camera_to_base = np.eye(4)
    camera_to_base[2, 3] = 1.0
    trajectory = generate_smooth_trajectory(
        "robot_00", (0, 0, 0), (2, 1, 0.5), 0, camera_to_base, frames=60, fps=10
    )
    assert trajectory.frames == 60
    assert trajectories_equal(trajectory, trajectory)
    changed = replace(trajectory, base_to_world=trajectory.base_to_world.copy())
    changed.base_to_world[10, 0, 3] += 0.01
    assert not trajectories_equal(trajectory, changed)


def test_future_timed_event_schema_is_compatible():
    target = make_object()
    event = InterventionEvent(
        "event", InterventionType.RIGID_RELOCATION, target.instance_id,
        ApplicationMode.TIMED, 12, {}, target, None,
    )
    assert event.time_index == 12
    with pytest.raises(ValueError):
        InterventionEvent(
            "bad", InterventionType.RIGID_RELOCATION, target.instance_id,
            ApplicationMode.PRE_ROLLOUT, 12, {},
        )


def test_articulation_and_state_interventions_preserve_before_after_records():
    target = replace(
        make_object(),
        articulated=True,
        joint_names=("door_hinge",),
        joint_limits=((0.0, 1.5),),
        joint_values=(0.1,),
        available_states=("ToggledOn",),
        semantic_states={"ToggledOn": False},
    )
    rng = np.random.default_rng(17)
    assert eligible_intervention_targets((target,), InterventionType.ARTICULATION) == (target,)
    articulation = propose_articulation(target, rng)
    assert articulation.before_object_state == target
    assert articulation.after_object_state.joint_values != target.joint_values
    state = propose_state_change(target, rng)
    assert state.parameters["value_before"] is False
    assert state.parameters["value_after"] is True
    assert state.after_object_state.semantic_states["ToggledOn"] is True


def test_state_hash_accepts_unbounded_joint_limits():
    target = replace(
        make_object(),
        articulated=True,
        joint_names=("continuous",),
        joint_limits=((-float("inf"), float("inf")),),
        joint_values=(0.0,),
    )
    assert len(exact_state_hash((target,))) == 64
