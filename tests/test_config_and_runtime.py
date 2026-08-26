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


def test_frozen_camera_setting_cannot_drift():
    config = load_yaml_config(REPOSITORY / "configs" / "default.yaml")
    config["camera"]["pitch_deg"] = 0.0
    with pytest.raises(ConfigurationError, match="pitch_deg"):
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
