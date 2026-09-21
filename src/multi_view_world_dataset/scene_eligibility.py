from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from multi_view_world_dataset.errors import ConfigurationError
from multi_view_world_dataset.sampling.splits import infer_scene_family


@dataclass(frozen=True)
class SceneEligibilityRecord:
    scene_id: str
    eligible: bool
    feasibility_class: str
    reason: str
    robot_asset: str
    robot_asset_version: str
    footprint_profile: str


@dataclass(frozen=True)
class SceneEligibilityManifest:
    version: str
    feasibility_source: str
    footprint_metadata: Mapping[str, Any]
    records: tuple[SceneEligibilityRecord, ...]

    @property
    def by_scene(self) -> dict[str, SceneEligibilityRecord]:
        return {record.scene_id: record for record in self.records}

    @property
    def eligible_scene_ids(self) -> tuple[str, ...]:
        return tuple(sorted(record.scene_id for record in self.records if record.eligible))

    @property
    def excluded_scene_ids(self) -> tuple[str, ...]:
        return tuple(sorted(record.scene_id for record in self.records if not record.eligible))


def load_scene_eligibility(path: str | Path) -> SceneEligibilityManifest:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigurationError(f"Scene eligibility manifest does not exist: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    scene_mapping = raw.get("scenes")
    if not isinstance(scene_mapping, dict) or not scene_mapping:
        raise ConfigurationError("Scene eligibility manifest must contain a non-empty scenes mapping")
    records: list[SceneEligibilityRecord] = []
    for scene_id, values in sorted(scene_mapping.items()):
        if not isinstance(values, dict):
            raise ConfigurationError(f"Invalid eligibility record for {scene_id}")
        try:
            records.append(SceneEligibilityRecord(
                scene_id=str(scene_id),
                eligible=bool(values["eligible"]),
                feasibility_class=str(values["feasibility_class"]),
                reason=str(values["reason"]),
                robot_asset=str(values["robot_asset"]),
                robot_asset_version=str(values["robot_asset_version"]),
                footprint_profile=str(values["footprint_profile"]),
            ))
        except KeyError as error:
            raise ConfigurationError(
                f"Eligibility record {scene_id} is missing {error.args[0]}"
            ) from error
    return SceneEligibilityManifest(
        version=str(raw.get("version", "unknown")),
        feasibility_source=str(raw.get("feasibility_source", "unknown")),
        footprint_metadata=dict(raw.get("footprint_metadata", {})),
        records=tuple(records),
    )


def reconcile_scene_eligibility(
    discovered_scene_ids: Sequence[str],
    manifest: SceneEligibilityManifest,
) -> dict[str, Any]:
    discovered = set(map(str, discovered_scene_ids))
    recorded = set(manifest.by_scene)
    missing_from_manifest = sorted(discovered - recorded)
    unavailable_installed = sorted(recorded - discovered)
    if missing_from_manifest or unavailable_installed:
        raise ConfigurationError(
            "Installed scenes do not match the scene eligibility manifest; "
            f"missing_from_manifest={missing_from_manifest}, "
            f"not_installed={unavailable_installed}. Regenerate or explicitly update the manifest."
        )
    eligible = list(manifest.eligible_scene_ids)
    excluded = [
        {
            "scene_id": record.scene_id,
            "reason": record.reason,
            "feasibility_class": record.feasibility_class,
        }
        for record in manifest.records
        if not record.eligible
    ]
    return {
        "manifest_version": manifest.version,
        "feasibility_source": manifest.feasibility_source,
        "all_discovered_scenes": sorted(discovered),
        "eligible_scenes": eligible,
        "excluded_scenes": excluded,
        "footprint_metadata": dict(manifest.footprint_metadata),
    }


def validate_scene_family_split_disjointness(split_mapping: Mapping[str, str]) -> None:
    family_splits: dict[str, set[str]] = {}
    for scene_id, split in split_mapping.items():
        family_splits.setdefault(infer_scene_family(scene_id), set()).add(str(split))
    leaked = {
        family: sorted(splits)
        for family, splits in family_splits.items()
        if len(splits) > 1
    }
    if leaked:
        raise ConfigurationError(f"Scene families leak across primary splits: {leaked}")


def resolve_scene_eligibility_path(
    repository_root: Path, config: Mapping[str, Any]
) -> Path:
    configured = Path(str(config["dataset"]["scene_eligibility_manifest"]))
    return configured if configured.is_absolute() else repository_root / configured
