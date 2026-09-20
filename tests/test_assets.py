from pathlib import Path

from multi_view_world_dataset.assets import (
    ROBOT_APPEARANCE_VARIANTS,
    materialize_mobile_sensor_robot,
    robot_appearance_metadata,
)


def test_materializes_final_robot_as_omnigibson_overlay(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    template_root = (
        repository_root / "assets" / "robots" / "mobile_sensor_robot_v1"
    )
    nova_uri = "https://assets.example/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"

    asset = materialize_mobile_sensor_robot(template_root, tmp_path, nova_uri)

    model_root = tmp_path / "models" / "mobile_sensor_robot_v1"
    assert asset.definition_path == model_root / "mobile_sensor_robot_v1.yaml"
    assert asset.usd_path == model_root / "usd" / "mobile_sensor_robot_v1.usda"
    usd = asset.usd_path.read_text(encoding="utf-8")
    definition = asset.definition_path.read_text(encoding="utf-8")
    assert nova_uri in usd
    assert "PhysicsPrismaticJoint" in usd
    assert "mvwd_mast_joint" in usd
    assert "PhysicsCollisionAPI" in usd
    assert "__NOVA_CARTER_USD__" not in usd
    assert str(asset.usd_path) in definition
    assert "__GENERATED_USD_PATH__" not in definition
    assert "tower_outer_shell" in usd
    assert "sliding_sleeve" in usd
    assert "sensor_head_housing" in usd
    assert "sensor_head_top_cap" in usd
    assert "front_lens" in usd
    assert "visibility = \"invisible\"" in usd
    assert "lower edge is 0.775 m" in usd
    for material in ("accent_orange", "accent_blue", "accent_green"):
        assert f'def Material "{material}"' in usd
    for visual_prim in (
        "tower_outer_shell",
        "tower_accent_band",
        "sliding_sleeve",
        "sensor_head_housing",
        "sensor_head_top_cap",
    ):
        block = usd.split(f'"{visual_prim}"', 1)[1].split("}", 1)[0]
        assert "PhysicsCollisionAPI" not in block
    assert tuple(ROBOT_APPEARANCE_VARIANTS) == ("robot_00", "robot_01", "robot_02")
    assert {entry["variant"] for entry in robot_appearance_metadata().values()} == {"orange", "blue", "green"}
