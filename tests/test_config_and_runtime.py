from pathlib import Path

import pytest

from multi_view_world_dataset.errors import ConfigurationError, SimulatorUnavailableError
from multi_view_world_dataset.utils.config import load_yaml_config, validate_config
from multi_view_world_dataset.utils.runtime import resolve_runtime_paths


REPOSITORY = Path(__file__).resolve().parents[1]


def test_all_profiles_are_valid():
    for name in ("default", "smoke", "integration", "pilot"):
        config = load_yaml_config(REPOSITORY / "configs" / f"{name}.yaml")
        assert config["dataset"]["robots"] == 3
        assert config["camera"]["hfov_deg"] == 70.0


def test_nonrigid_fallback_requires_separate_schema_version():
    config = load_yaml_config(
        REPOSITORY / "configs" / "production_v12_nonrigid_fallback.yaml"
    )
    assert config["dataset"]["schema_version"] == "1.2.0"
    assert config["configuration_sampling"]["nonrigid_fallback"] is True
    assert config["configuration_sampling"]["minimum_changed_objects"] == 2
    assert config["configuration_sampling"]["nonrigid_max_changed_objects"] == 4
    config["configuration_sampling"]["nonrigid_max_changed_objects"] = 1
    with pytest.raises(ConfigurationError, match="nonrigid_max_changed_objects"):
        validate_config(config)
    config["configuration_sampling"]["nonrigid_max_changed_objects"] = 4
    config["dataset"]["schema_version"] = "1.1.0"
    with pytest.raises(ConfigurationError, match="nonrigid configuration fallback"):
        validate_config(config)



def test_frozen_camera_setting_cannot_drift():
    config = load_yaml_config(REPOSITORY / "configs" / "default.yaml")
    config["camera"]["pitch_deg"] = 0.0
    with pytest.raises(ConfigurationError, match="pitch_deg"):
        validate_config(config)


def test_native_relation_attempt_budget_must_be_positive():
    config = load_yaml_config(REPOSITORY / "configs" / "smoke.yaml")
    config["generation"]["native_relation_low_level_attempts"] = 0
    with pytest.raises(ConfigurationError, match="native_relation_low_level_attempts"):
        validate_config(config)


def test_episode_sampling_rounds_must_be_positive():
    config = load_yaml_config(REPOSITORY / "configs" / "smoke.yaml")
    config["generation"]["maximum_episode_sampling_rounds"] = 0
    with pytest.raises(ConfigurationError, match="maximum_episode_sampling_rounds"):
        validate_config(config)


def test_scene_sampling_restarts_must_be_non_negative():
    config = load_yaml_config(REPOSITORY / "configs" / "smoke.yaml")
    config["generation"]["maximum_scene_sampling_restarts"] = -1
    with pytest.raises(ConfigurationError, match="maximum_scene_sampling_restarts"):
        validate_config(config)


def test_worker_progress_poll_must_be_shorter_than_stall_timeout():
    config = load_yaml_config(REPOSITORY / "configs" / "smoke.yaml")
    config["generation"]["worker_progress_stall_timeout_s"] = 10
    config["generation"]["worker_progress_poll_interval_s"] = 10
    with pytest.raises(ConfigurationError, match="poll_interval_s"):
        validate_config(config)


def test_post_render_effect_requires_per_intervention_thresholds():
    config = load_yaml_config(REPOSITORY / "configs" / "smoke.yaml")
    config["intervention"]["post_render_effect"]["minimum_mean_rgb_delta"] = 3.0
    with pytest.raises(ConfigurationError, match="minimum_mean_rgb_delta"):
        validate_config(config)


def test_minimum_gt_valid_candidates_before_rescue_is_bounded():
    config = load_yaml_config(REPOSITORY / "configs" / "smoke.yaml")
    config["trajectory"]["minimum_gt_valid_candidates_before_rescue"] = 0
    with pytest.raises(
        ConfigurationError,
        match="minimum_gt_valid_candidates_before_rescue",
    ):
        validate_config(config)


def test_navigation_route_bank_minimum_is_deprecated_diagnostic_only():
    config = load_yaml_config(REPOSITORY / "configs" / "smoke.yaml")
    config["navigation"]["route_bank_minimum_size"] = (
        config["navigation"]["route_bank_target_size"] + 1
    )
    validate_config(config)


def test_navigation_view_cohort_controls_are_validated():
    config = load_yaml_config(REPOSITORY / "configs" / "smoke.yaml")
    config["navigation"][
        "route_bank_view_cohort_completion_probability"
    ] = 1.1
    with pytest.raises(ConfigurationError, match="cohort_completion_probability"):
        validate_config(config)

def test_realized_regime_soft_weight_must_be_non_negative():
    config = load_yaml_config(REPOSITORY / "configs" / "smoke.yaml")
    config["trajectory"]["overlap_preflight"][
        "realized_regime_soft_weight"
    ] = -0.1
    with pytest.raises(ConfigurationError, match="realized_regime_soft_weight"):
        validate_config(config)



def test_runtime_path_precedence_and_failure(tmp_path):
    behavior = tmp_path / "behavior"
    (behavior / "OmniGibson" / "omnigibson").mkdir(parents=True)
    output_cli = tmp_path / "cli-output"
    config = load_yaml_config(REPOSITORY / "configs" / "smoke.yaml")
    runtime = resolve_runtime_paths(
        config,
        behavior_root=behavior,
        output_root=output_cli,
        environ={"BEHAVIOR_ROOT": str(tmp_path / "wrong"), "DATASET_DEV_OUTPUT": str(tmp_path / "env-output")},
    )
    assert runtime.behavior_root == behavior
    assert runtime.output_root == output_cli
    with pytest.raises(SimulatorUnavailableError):
        resolve_runtime_paths(config, environ={})
