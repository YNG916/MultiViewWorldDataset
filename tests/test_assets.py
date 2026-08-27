from pathlib import Path

from multi_view_world_dataset.assets import materialize_mobile_sensor_robot


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
