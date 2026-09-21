from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import yaml

from multi_view_world_dataset.dataset_diagnostics import summarize_generated_dataset
from multi_view_world_dataset.errors import ConfigurationError
from multi_view_world_dataset.utils.serialization import dump_json


def _fractions(counts: Mapping[str, int]) -> dict[str, float]:
    total = sum(int(value) for value in counts.values())
    return {
        str(key): int(value) / total for key, value in sorted(counts.items())
    } if total else {}


def _bar_svg(path: Path, title: str, counts: Mapping[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = sorted((str(key), int(value)) for key, value in counts.items())
    width = 720
    height = 80 + 42 * max(1, len(items))
    maximum = max((value for _, value in items), default=1)
    rows = []
    for index, (label, value) in enumerate(items):
        y = 55 + 42 * index
        bar_width = int(480 * value / max(1, maximum))
        rows.append(
            f'<text x="10" y="{y + 17}" font-size="14">{label}</text>'
            f'<rect x="180" y="{y}" width="{bar_width}" height="24" fill="#3979c3"/>'
            f'<text x="{190 + bar_width}" y="{y + 17}" font-size="14">{value}</text>'
        )
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="10" y="28" font-size="20" font-weight="bold">{title}</text>'
        + "".join(rows) + "</svg>\n",
        encoding="utf-8",
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ConfigurationError(f"Invalid resolved config: {path}")
    return value


def generate_pilot_report(dataset_root: str | Path) -> tuple[Path, dict[str, Any]]:
    root = Path(dataset_root).expanduser().resolve()
    manifest_path = root / "global" / "production_manifest.json"
    if not manifest_path.is_file():
        raise ConfigurationError(f"Missing finalized production manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = _load_yaml(root / "global" / "resolved_config.yaml")
    report_root = root / "global"
    _, combined = summarize_generated_dataset(
        root, output_path=report_root / "dataset_diagnostics.json"
    )
    per_scene_root = report_root / "per_scene_summary"
    per_scene_root.mkdir(parents=True, exist_ok=True)
    per_scene: dict[str, Any] = {}
    for scene_id in manifest["selected_scenes"]:
        shard = root / "shards" / scene_id
        _, scene_report = summarize_generated_dataset(
            shard, output_path=per_scene_root / f"{scene_id}.json"
        )
        status = json.loads((shard / "shard_status.json").read_text(encoding="utf-8"))
        eligibility = manifest["scene_eligibility_records"][scene_id]
        per_scene[scene_id] = {
            "feasibility_class": eligibility["feasibility_class"],
            "selection_reason": config.get("production", {}).get(
                "pilot_scene_selection", {}
            ).get(scene_id),
            "status": status,
            "diagnostics": scene_report,
        }

    episode_count = int(combined["finalized_episode_count"])
    expected_episodes = (
        len(manifest["selected_scenes"])
        * int(config["dataset"]["accepted_configurations_per_scene"])
        * int(config["dataset"]["accepted_episodes_per_configuration"])
    )
    regime_counts = combined["trajectory"]["realized_regime_counts"]
    intervention_counts = combined["intervention"]["type_counts"]
    changed_distribution = combined["configuration"][
        "changed_object_count_distribution"
    ]
    reject_counts = combined["rejections"]["reason_counts"]
    exhaustion_count = int(reject_counts.get(
        "trajectory_set_gt_candidates_exhausted", 0
    ))
    readiness = config.get("production_readiness", {})
    maximum_exhaustion_fraction = float(
        readiness.get("maximum_gt_candidate_exhaustion_fraction", 0.10)
    )
    maximum_dominant_fraction = float(
        readiness.get("maximum_dominant_distribution_fraction", 0.90)
    )
    minimum_changed_bins = int(
        readiness.get("minimum_changed_object_count_bins", 3)
    )
    regime_fractions = _fractions(regime_counts)
    intervention_fractions = _fractions(intervention_counts)
    changed_values = [int(key) for key, value in changed_distribution.items() if value]
    production_status = json.loads(
        (root / "production_status.json").read_text(encoding="utf-8")
    )
    criteria = {
        "all_selected_scene_shards_complete": all(
            production_status.get("scenes", {}).get(scene, {}).get("status") == "complete"
            for scene in manifest["selected_scenes"]
        ),
        "expected_episode_count_complete": episode_count == expected_episodes,
        "healthy_and_constrained_scene_present": {
            record["feasibility_class"] for record in per_scene.values()
        } >= {"healthy", "constrained"},
        "all_union_graphs_connected": (
            combined["overlap"]["union_connected_episode_count"] == episode_count
        ),
        "gt_candidate_exhaustion_not_common": (
            exhaustion_count / max(1, episode_count) <= maximum_exhaustion_fraction
        ),
        "overlap_regime_not_collapsed": (
            len(regime_fractions) >= 2
            and max(regime_fractions.values(), default=1.0) <= maximum_dominant_fraction
        ),
        "intervention_type_not_collapsed": (
            len(intervention_fractions) >= 2
            and max(intervention_fractions.values(), default=1.0) <= maximum_dominant_fraction
        ),
        "dynamic_configuration_uses_2_to_6_objects": (
            bool(changed_values)
            and min(changed_values) >= 2
            and max(changed_values) <= 6
            and len(changed_values) >= minimum_changed_bins
        ),
        "observation_metadata_complete": (
            combined["identity_and_calibration"][
                "complete_observation_metadata_episode_count"
            ] == episode_count
        ),
        "world_bev_calibration_complete": (
            combined["identity_and_calibration"][
                "complete_world_bev_calibration_episode_count"
            ] == episode_count
        ),
    }
    eligible_scene_count = len(manifest["eligible_scenes"])
    full_configurations_per_scene = int(
        config.get("production", {}).get("full_configurations_per_scene", 150)
    )
    episodes_per_configuration = int(
        config.get("production", {}).get("episodes_per_configuration", 3)
    )
    full_configurations = eligible_scene_count * full_configurations_per_scene
    full_episodes = full_configurations * episodes_per_configuration
    robot_frames = full_episodes * 3 * 60 * 2
    world_bev_frames = full_episodes * 60 * 2
    episode_size = combined["storage"]["bytes_per_episode"]
    configuration_size = combined["storage"]["bytes_per_configuration"]
    projected_episode_bytes = float(episode_size.get("mean", 0.0)) * full_episodes
    projected_configuration_bytes = (
        float(configuration_size.get("mean", 0.0)) * full_configurations
    )
    projected_total_bytes = projected_episode_bytes + projected_configuration_bytes
    headroom_fraction = float(readiness.get("storage_headroom_fraction", 0.25))
    required_with_headroom = projected_total_bytes * (1.0 + headroom_fraction)
    free_bytes = shutil.disk_usage(root).free
    criteria["projected_storage_with_headroom_available"] = (
        free_bytes >= required_with_headroom
    )
    readiness_status = "READY" if all(criteria.values()) else "NOT READY"
    mean_episode_runtime = float(
        combined.get("runtime_s", {}).get("total_episode_s", {}).get("mean", 0.0)
    )
    projected_serial_runtime_s = mean_episode_runtime * full_episodes

    report = {
        "status": "pass",
        "profile": config["profile"],
        "full_production_was_started": False,
        "episode_count": episode_count,
        "expected_episode_count": expected_episodes,
        "per_scene": per_scene,
        "acceptance_efficiency": combined["acceptance_efficiency"],
        "gt_candidate_efficiency": combined["gt_candidate_efficiency"],
        "overlap": {
            **combined["overlap"],
            "target_regime_distribution": manifest["target_regime_distribution"],
            "realized_regime_counts": regime_counts,
            "realized_regime_fractions": regime_fractions,
        },
        "spatial": combined["spatial"],
        "trajectory": combined["trajectory"],
        "dynamic_configuration": combined["configuration"],
        "intervention": {
            **combined["intervention"],
            "type_fractions": intervention_fractions,
        },
        "runtime_s": combined["runtime_s"],
        "storage": {
            **combined["storage"],
            "projected_full_episode_bytes": projected_episode_bytes,
            "projected_full_configuration_bytes": projected_configuration_bytes,
            "projected_full_total_bytes": projected_total_bytes,
            "recommended_headroom_fraction": headroom_fraction,
            "required_bytes_with_headroom": required_with_headroom,
            "filesystem_free_bytes_at_report_time": free_bytes,
            "sufficient_free_space_at_report_time": free_bytes >= required_with_headroom,
        },
        "full_scale_projection": {
            "eligible_scene_count": eligible_scene_count,
            "configurations": full_configurations,
            "episodes": full_episodes,
            "robot_view_frames": robot_frames,
            "world_bev_frames": world_bev_frames,
            "projected_serial_episode_runtime_s": projected_serial_runtime_s,
        },
        "production_readiness": {
            "status": readiness_status,
            "criteria": criteria,
        },
        "collapse_warnings": combined["collapse_warnings"],
    }
    output = report_root / "pilot_report.json"
    dump_json(output, report)
    plots = report_root / "plots"
    _bar_svg(plots / "overlap_regimes.svg", "Realized overlap regimes", regime_counts)
    _bar_svg(plots / "gt_validation_batches.svg", "GT validation batches", combined[
        "gt_candidate_efficiency"
    ]["accepted_batch_counts"])
    _bar_svg(plots / "intervention_types.svg", "Accepted intervention types", intervention_counts)
    _bar_svg(plots / "changed_object_counts.svg", "DynamicConfiguration changed objects", changed_distribution)
    lines = [
        "# MultiViewWorldDataset production-readiness report",
        "",
        f"- Readiness: **{readiness_status}**",
        f"- Pilot episodes: {episode_count} / {expected_episodes}",
        f"- Full production started: **NO**",
        f"- Eligible scenes: {eligible_scene_count}",
        f"- Projected full configurations: {full_configurations:,}",
        f"- Projected full episodes: {full_episodes:,}",
        f"- Projected robot-view frames: {robot_frames:,}",
        f"- Projected world-BEV frames: {world_bev_frames:,}",
        "",
        "## Readiness criteria",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'}: {name}"
        for name, passed in criteria.items()
    )
    lines.extend([
        "",
        "## Distribution plots",
        "",
        "- [Overlap regimes](plots/overlap_regimes.svg)",
        "- [GT validation batches](plots/gt_validation_batches.svg)",
        "- [Intervention types](plots/intervention_types.svg)",
        "- [Changed object counts](plots/changed_object_counts.svg)",
        "",
        "Full production was NOT started.",
    ])
    (report_root / "pilot_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output, report
