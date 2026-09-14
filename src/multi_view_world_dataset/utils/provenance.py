from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from multi_view_world_dataset.utils.serialization import to_jsonable

SEMANTICS_VERSION = "Dataset-v1.1"


def configuration_fingerprint(config: dict[str, Any]) -> str:
    payload = {
        "semantics_version": SEMANTICS_VERSION,
        "resolved_config": to_jsonable(config),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def default_taxonomy() -> dict[str, Any]:
    return {
        "version": SEMANTICS_VERSION,
        "semantic_labels": {
            "0": {"name": "background", "reserved": True},
            "1": {"name": "unknown", "reserved": True},
            "2": {"name": "robot", "reserved": True},
        },
        "instance_id_convention": {
            "0": "background",
            "1": "robot_00",
            "2": "robot_01",
            "3": "robot_02",
            "scene_objects_start_at": 4,
        },
        "instance_catalogs": {},
    }


def generator_source_fingerprint(repository_root: Path) -> str:
    """Hash generator sources so incompatible code cannot resume one root."""
    digest = hashlib.sha256()
    paths = sorted((repository_root / "src").rglob("*.py"))
    paths.extend(sorted((repository_root / "configs").glob("*.yaml")))
    paths.append(repository_root / "pyproject.toml")
    for path in paths:
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(repository_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
