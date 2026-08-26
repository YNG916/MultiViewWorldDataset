from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from multi_view_world_dataset.errors import SimulatorUnavailableError


@dataclass(frozen=True)
class MaterializedRobotAsset:
    usd_path: Path
    definition_path: Path
    nova_carter_uri: str


def materialize_mobile_sensor_robot(
    template_root: Path,
    generated_root: Path,
    nova_carter_uri: str,
) -> MaterializedRobotAsset:
    """Materialize portable templates after the adapter has resolved the installed asset URI."""
    usd_template = template_root / "mobile_sensor_robot_v1.usda.in"
    yaml_template = template_root / "mobile_sensor_robot_v1.yaml.in"
    if not usd_template.is_file() or not yaml_template.is_file():
        raise SimulatorUnavailableError(f"Final robot templates are missing under {template_root}")
    model_root = generated_root / "mobile_sensor_robot_v1"
    usd_root = model_root / "usd"
    usd_root.mkdir(parents=True, exist_ok=True)
    usd_path = usd_root / "mobile_sensor_robot_v1.usda"
    definition_path = model_root / "mobile_sensor_robot_v1.yaml"
    usd_path.write_text(
        usd_template.read_text(encoding="utf-8").replace("__NOVA_CARTER_USD__", nova_carter_uri), encoding="utf-8"
    )
    definition_path.write_text(
        yaml_template.read_text(encoding="utf-8").replace("__GENERATED_USD_PATH__", str(usd_path)), encoding="utf-8"
    )
    return MaterializedRobotAsset(usd_path, definition_path, nova_carter_uri)

