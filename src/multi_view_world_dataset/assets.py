from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from multi_view_world_dataset.errors import SimulatorUnavailableError

# Canonical appearance identity is part of Dataset-v1.1 metadata. The colors
# are deliberately broad, saturated surfaces rather than small decals so that
# robot identity remains legible in world BEV and ego views.
ROBOT_APPEARANCE_VARIANTS: dict[str, dict[str, object]] = {
    "robot_00": {
        "variant": "orange",
        "display_name": "warm orange",
        "material_prim": "accent_orange",
        "rgb": (0.95, 0.31, 0.055),
        "rgb8": (242, 79, 14),
    },
    "robot_01": {
        "variant": "blue",
        "display_name": "cool blue",
        "material_prim": "accent_blue",
        "rgb": (0.04, 0.34, 0.92),
        "rgb8": (10, 87, 235),
    },
    "robot_02": {
        "variant": "green",
        "display_name": "signal green",
        "material_prim": "accent_green",
        "rgb": (0.05, 0.62, 0.25),
        "rgb8": (13, 158, 64),
    },
}


def robot_appearance_metadata() -> dict[str, dict[str, object]]:
    """Return a JSON-safe copy of the canonical robot appearance mapping."""
    return {
        robot_id: {
            "variant": str(values["variant"]),
            "display_name": str(values["display_name"]),
            "material_prim": str(values["material_prim"]),
            "rgb": [float(value) for value in values["rgb"]],
        }
        for robot_id, values in ROBOT_APPEARANCE_VARIANTS.items()
    }


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
    model_root = generated_root / "models" / "mobile_sensor_robot_v1"
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

