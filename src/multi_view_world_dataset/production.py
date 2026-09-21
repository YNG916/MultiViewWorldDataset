from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from multi_view_world_dataset.assets import robot_asset_fingerprint
from multi_view_world_dataset.errors import ConfigurationError
from multi_view_world_dataset.scene_eligibility import (
    load_scene_eligibility,
    resolve_scene_eligibility_path,
)
from multi_view_world_dataset.sampling.splits import (
    assign_scene_family_splits,
    infer_scene_family,
)
from multi_view_world_dataset.utils.provenance import (
    default_taxonomy,
    generator_source_fingerprint,
    resolved_dataset_fingerprint,
)
from multi_view_world_dataset.utils.runtime import RuntimePaths, generator_git_commit
from multi_view_world_dataset.utils.serialization import dump_json, to_jsonable

_SCENE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def scene_shard_path(dataset_root: str | Path, scene_id: str) -> Path:
    if not _SCENE_ID.fullmatch(scene_id):
        raise ConfigurationError(f"Unsafe scene ID for shard path: {scene_id!r}")
    return Path(dataset_root).expanduser().resolve() / "shards" / scene_id


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _target_fingerprint(config: Mapping[str, Any], repository_root: Path) -> str:
    source = generator_source_fingerprint(repository_root)
    return resolved_dataset_fingerprint(dict(config), source)


def _selected_scenes(config: Mapping[str, Any], repository_root: Path) -> tuple[str, ...]:
    manifest = load_scene_eligibility(
        resolve_scene_eligibility_path(repository_root, config)
    )
    configured = config.get("production", {}).get("selected_scenes")
    selected = (
        tuple(sorted(map(str, configured)))
        if configured
        else manifest.eligible_scene_ids
    )
    unknown = sorted(set(selected) - set(manifest.by_scene))
    excluded = sorted(
        scene for scene in selected
        if scene in manifest.by_scene and not manifest.by_scene[scene].eligible
    )
    if unknown or excluded:
        raise ConfigurationError(
            f"Production scene selection is invalid; unknown={unknown}, excluded={excluded}"
        )
    return selected


def initialize_production_root(
    dataset_root: str | Path,
    config: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    global_root = root / "global"
    global_root.mkdir(parents=True, exist_ok=True)
    (root / "shards").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    eligibility_path = resolve_scene_eligibility_path(repository_root, config)
    eligibility = load_scene_eligibility(eligibility_path)
    selected = _selected_scenes(config, repository_root)
    splits = assign_scene_family_splits(
        list(eligibility.eligible_scene_ids),
        dict(config["dataset"]["splits"]),
        int(config["dataset"]["scene_family_split_seed"]),
    )
    manifest = {
        "schema_version": config["dataset"]["schema_version"],
        "dataset_semantics": "Dataset-v1.1",
        "profile": config["profile"],
        "configuration_fingerprint": _target_fingerprint(config, repository_root),
        "generator_source_fingerprint": generator_source_fingerprint(repository_root),
        "generator_git_commit": generator_git_commit(repository_root),
        "robot_asset_fingerprint": robot_asset_fingerprint(repository_root),
        "scene_eligibility_manifest": str(eligibility_path),
        "scene_eligibility_version": eligibility.version,
        "all_manifest_scenes": sorted(eligibility.by_scene),
        "eligible_scenes": list(eligibility.eligible_scene_ids),
        "scene_eligibility_records": {
            record.scene_id: {
                "eligible": record.eligible,
                "feasibility_class": record.feasibility_class,
                "reason": record.reason,
                "robot_asset": record.robot_asset,
                "robot_asset_version": record.robot_asset_version,
                "footprint_profile": record.footprint_profile,
            }
            for record in eligibility.records
        },
        "excluded_scenes": [
            {
                "scene_id": record.scene_id,
                "reason": record.reason,
                "feasibility_class": record.feasibility_class,
            }
            for record in eligibility.records if not record.eligible
        ],
        "selected_scenes": list(selected),
        "split_mapping": splits,
        "scene_family_mapping": {
            scene: infer_scene_family(scene) for scene in eligibility.eligible_scene_ids
        },
        "target_regime_distribution": dict(
            config["placement"]["observation_regime_weights"]
        ),
        "one_fresh_process_per_scene": True,
        "shard_layout": "shards/<scene_id>",
    }
    path = global_root / "production_manifest.json"
    existing = _read_json(path)
    if existing is not None and existing.get("configuration_fingerprint") != manifest["configuration_fingerprint"]:
        raise ConfigurationError(
            "Refusing production resume because global configuration fingerprint differs"
        )
    dump_json(path, manifest)
    resolved = global_root / "resolved_config.yaml"
    if not resolved.is_file():
        resolved.write_text(
            yaml.safe_dump(to_jsonable(dict(config)), sort_keys=True),
            encoding="utf-8",
        )
    return manifest


def _initial_scene_states(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    states = {
        str(item["scene_id"]): {
            "status": "excluded",
            "reason": item["reason"],
        }
        for item in manifest["excluded_scenes"]
    }
    for scene in manifest["selected_scenes"]:
        states.setdefault(str(scene), {"status": "pending"})
    return states


def _should_launch_scene(status: str | None, retry_failed: bool) -> bool:
    if status == "complete":
        return False
    if status == "failed":
        return retry_failed
    return status in {None, "pending", "running"}


def run_scene_worker(
    runtime: RuntimePaths,
    config: dict[str, Any],
    scene_id: str,
    *,
    allow_large: bool,
) -> tuple[Path, dict[str, Any]]:
    from multi_view_world_dataset.generator import generate_dataset

    production_root = runtime.require_output()
    repository_root = Path(__file__).resolve().parents[2]
    shard = scene_shard_path(production_root, scene_id)
    shard.mkdir(parents=True, exist_ok=True)
    fingerprint = _target_fingerprint(config, repository_root)
    prior = _read_json(shard / "shard_status.json", {})
    prior_fingerprint = prior.get("configuration_fingerprint")
    if prior_fingerprint is not None and prior_fingerprint != fingerprint:
        raise ConfigurationError(
            f"Refusing resume for shard {scene_id}: fingerprint mismatch"
        )
    started = time.time()
    dump_json(shard / "shard_status.json", {
        "scene_id": scene_id,
        "status": "running",
        "configuration_fingerprint": fingerprint,
        "started_unix_s": started,
        "worker_pid": os.getpid(),
    })
    shard_runtime = RuntimePaths(
        behavior_root=runtime.behavior_root,
        output_root=shard,
        cache_root=runtime.cache_root,
    )
    try:
        _, result = generate_dataset(
            shard_runtime, config, scene_id=scene_id, allow_large=allow_large
        )
        if result.get("status") != "pass":
            reason = result.get("error", "scene generation returned a non-pass status")
            raise RuntimeError(
                f"scene generation failed for {scene_id}: {reason}"
            )
    except BaseException as error:
        finished = time.time()
        failure = {
            "scene_id": scene_id,
            "status": "failed",
            "configuration_fingerprint": fingerprint,
            "started_unix_s": started,
            "finished_unix_s": finished,
            "elapsed_s": finished - started,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        dump_json(shard / "shard_status.json", failure)
        dump_json(shard / "timing.json", {"total_scene_worker_s": finished - started})
        raise
    finished = time.time()
    status = {
        "scene_id": scene_id,
        "status": "complete",
        "configuration_fingerprint": fingerprint,
        "started_unix_s": started,
        "finished_unix_s": finished,
        "elapsed_s": finished - started,
        "accepted_configurations": int(result["accepted_configurations"]),
        "accepted_episodes": int(result["accepted_episodes"]),
    }
    dump_json(shard / "shard_status.json", status)
    dump_json(shard / "timing.json", {"total_scene_worker_s": finished - started})
    return shard, status


def _worker_command(
    config_path: Path,
    runtime: RuntimePaths,
    scene_id: str,
    *,
    allow_large: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "multi_view_world_dataset.cli",
        "scene-worker",
        "--config",
        str(config_path),
        "--scene",
        scene_id,
        "--behavior-root",
        str(runtime.behavior_root),
        "--output-root",
        str(runtime.require_output()),
    ]
    if runtime.cache_root is not None:
        command.extend(("--cache-root", str(runtime.cache_root)))
    if allow_large:
        command.append("--allow-large")
    return command


def launch_scene_shards(
    runtime: RuntimePaths,
    config: dict[str, Any],
    config_path: str | Path,
    *,
    gpus: Sequence[str],
    max_workers: int,
    allow_large: bool,
    retry_failed: bool = False,
) -> tuple[Path, dict[str, Any]]:
    if not gpus:
        raise ConfigurationError("At least one GPU ID is required")
    if max_workers < 1:
        raise ConfigurationError("max_workers must be positive")
    root = runtime.require_output()
    repository_root = Path(__file__).resolve().parents[2]
    manifest = initialize_production_root(root, config, repository_root)
    states = _initial_scene_states(manifest)
    previous = _read_json(root / "production_status.json", {})
    for scene, record in previous.get("scenes", {}).items():
        if scene in states and record.get("status") in {"complete", "failed"}:
            states[scene] = record
    selected = [
        scene for scene in manifest["selected_scenes"]
        if _should_launch_scene(states.get(scene, {}).get("status"), retry_failed)
    ]
    for index, scene in enumerate(selected):
        states[scene] = {"status": "pending", "assigned_gpu": str(gpus[index % len(gpus)])}
    status = {
        "status": "running",
        "profile": config["profile"],
        "configuration_fingerprint": manifest["configuration_fingerprint"],
        "full_production_started": config["profile"] == "production",
        "scenes": states,
    }
    dump_json(root / "production_status.json", status)

    def invoke(scene: str, gpu: str) -> dict[str, Any]:
        log_path = root / "logs" / f"{scene}.log"
        env = dict(os.environ)
        # OmniGibson passes this physical ordinal to both Isaac renderer and
        # PhysX. CUDA_VISIBLE_DEVICES remaps CUDA ordinals but not Vulkan and
        # therefore breaks nonzero-GPU rendering in this installed runtime.
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env["OMNIGIBSON_GPU_ID"] = str(gpu)
        started = time.time()
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[mvwd-parent] launch scene={scene} gpu={gpu}\n")
            completed = subprocess.run(
                _worker_command(
                    Path(config_path).expanduser().resolve(), runtime, scene,
                    allow_large=allow_large,
                ),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
            )
        worker_status = _read_json(
            scene_shard_path(root, scene) / "shard_status.json", {}
        )
        generation_status = _read_json(
            scene_shard_path(root, scene) / "generation_status.json", {}
        )
        # Isaac Sim's configured fast shutdown may terminate the interpreter
        # from adapter.close() before run_scene_worker regains control. Reconcile
        # the generator's already-atomic terminal record in the parent instead
        # of leaving a dead worker marked as running forever.
        if worker_status.get("status") not in {"complete", "failed"}:
            terminal = generation_status.get("status")
            if terminal in {"pass", "error"}:
                finished = time.time()
                generator_passed = terminal == "pass"
                expected_configurations = int(
                    config["dataset"]["accepted_configurations_per_scene"]
                )
                expected_episodes = expected_configurations * int(
                    config["dataset"]["accepted_episodes_per_configuration"]
                )
                accepted_configurations = int(
                    generation_status.get("accepted_configurations", 0)
                )
                accepted_episodes = int(
                    generation_status.get("accepted_episodes", 0)
                )
                complete_counts = bool(
                    accepted_configurations == expected_configurations
                    and accepted_episodes == expected_episodes
                )
                reconciled_complete = bool(
                    completed.returncode == 0
                    and generator_passed
                    and complete_counts
                )
                worker_status = {
                    **worker_status,
                    "scene_id": scene,
                    "status": "complete" if reconciled_complete else "failed",
                    "finished_unix_s": finished,
                    "elapsed_s": finished - started,
                    "accepted_configurations": accepted_configurations,
                    "accepted_episodes": accepted_episodes,
                    "terminal_status_source": "generation_status_parent_reconciliation",
                }
                if not reconciled_complete:
                    worker_status.update({
                        "error_type": str(
                            generation_status.get("error_type", "WorkerStatusError")
                        ),
                        "error": str(generation_status.get(
                            "error",
                            "generator terminal counts did not match the scene target",
                        )),
                    })
                dump_json(
                    scene_shard_path(root, scene) / "shard_status.json",
                    worker_status,
                )
                dump_json(
                    scene_shard_path(root, scene) / "timing.json",
                    {"total_scene_worker_s": finished - started},
                )
        worker_complete = bool(
            completed.returncode == 0
            and worker_status.get("status") == "complete"
        )
        status_error = None
        if completed.returncode == 0 and not worker_complete:
            status_error = (
                "scene worker exited successfully without a complete shard status"
            )
        return {
            **worker_status,
            "scene_id": scene,
            "status": "complete" if worker_complete else "failed",
            "returncode": completed.returncode,
            "assigned_gpu": gpu,
            "log": str(log_path),
            "parent_elapsed_s": time.time() - started,
            **({"parent_error": status_error} if status_error else {}),
        }

    worker_count = min(max_workers, len(gpus), max(1, len(selected)))
    if selected:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(invoke, scene, str(gpus[index % len(gpus)])): scene
                for index, scene in enumerate(selected)
            }
            for future in as_completed(futures):
                scene = futures[future]
                try:
                    states[scene] = future.result()
                except BaseException as error:
                    states[scene] = {
                        "scene_id": scene,
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                dump_json(root / "production_status.json", {**status, "scenes": states})
    failed = sorted(scene for scene, record in states.items() if record["status"] == "failed")
    pending = sorted(scene for scene in manifest["selected_scenes"] if states[scene]["status"] != "complete")
    result = {
        **status,
        "status": "pass" if not failed and not pending else "error",
        "scenes": states,
        "complete_scenes": sorted(
            scene for scene, record in states.items() if record["status"] == "complete"
        ),
        "failed_scenes": failed,
        "pending_scenes": pending,
    }
    dump_json(root / "production_status.json", result)
    return root, result


def merge_taxonomies(taxonomies: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    merged = default_taxonomy()
    semantic = merged["semantic_labels"]
    names = {str(record["name"]): str(key) for key, record in semantic.items()}
    catalogs = merged["instance_catalogs"]
    for taxonomy in taxonomies:
        for semantic_id, record in sorted(
            taxonomy.get("semantic_labels", {}).items(), key=lambda item: int(item[0])
        ):
            semantic_id = str(semantic_id)
            name = str(record["name"])
            if semantic_id in semantic and semantic[semantic_id]["name"] != name:
                raise ConfigurationError(
                    f"Semantic ID collision at {semantic_id}: {semantic[semantic_id]['name']} != {name}"
                )
            if name in names and names[name] != semantic_id:
                raise ConfigurationError(
                    f"Semantic category {name} has inconsistent IDs {names[name]} and {semantic_id}"
                )
            semantic[semantic_id] = dict(record)
            names[name] = semantic_id
        for scene_id, entries in sorted(taxonomy.get("instance_catalogs", {}).items()):
            if scene_id in catalogs and catalogs[scene_id] != entries:
                raise ConfigurationError(f"Conflicting instance catalog for scene {scene_id}")
            catalogs[scene_id] = entries
    return merged


def finalize_dataset(dataset_root: str | Path) -> tuple[Path, dict[str, Any]]:
    root = Path(dataset_root).expanduser().resolve()
    manifest = _read_json(root / "global" / "production_manifest.json")
    if manifest is None:
        raise ConfigurationError(f"Missing production manifest under {root}")
    expected_fingerprint = manifest["configuration_fingerprint"]
    selected = tuple(map(str, manifest["selected_scenes"]))
    shard_records: list[dict[str, Any]] = []
    taxonomies: list[dict[str, Any]] = []
    episode_index: list[dict[str, str]] = []
    configuration_index: list[dict[str, str]] = []
    rejection_reasons: Counter[str] = Counter()
    requested_regimes: Counter[str] = Counter()
    realized_regimes: Counter[str] = Counter()
    intervention_types: Counter[str] = Counter()
    total_bytes = 0
    for scene_id in sorted(selected):
        shard = scene_shard_path(root, scene_id)
        status = _read_json(shard / "shard_status.json", {})
        if status.get("status") != "complete":
            raise ConfigurationError(f"Cannot finalize incomplete shard {scene_id}")
        metadata = _read_json(shard / "dataset_meta.json", {})
        if metadata.get("configuration_fingerprint") != expected_fingerprint:
            raise ConfigurationError(f"Shard fingerprint mismatch for {scene_id}")
        if metadata.get("schema_version") != manifest["schema_version"]:
            raise ConfigurationError(f"Shard schema mismatch for {scene_id}")
        taxonomy = _read_json(shard / "taxonomy.json", {})
        taxonomies.append(taxonomy)
        configurations = sorted(
            path.parent for path in (shard / "configurations").glob("*/*/config_meta.json")
        )
        episodes = sorted(
            path.parent for path in (shard / "episodes").glob("*/*/episode_*/meta.json")
        )
        for path in configurations:
            configuration_index.append({
                "scene_id": scene_id,
                "configuration_id": path.name,
                "path": str(path.relative_to(root)),
            })
        for path in episodes:
            episode_index.append({
                "scene_id": scene_id,
                "configuration_id": path.parent.name,
                "episode_id": path.name,
                "path": str(path.relative_to(root)),
            })
            events = _read_json(path / "events.json", [])
            if events:
                intervention_types[str(events[0].get("intervention_type", "unknown"))] += 1
        for line in (shard / "rejects.jsonl").read_text(encoding="utf-8").splitlines() if (shard / "rejects.jsonl").is_file() else ():
            try:
                rejection_reasons[str(json.loads(line).get("reason", "unknown"))] += 1
            except json.JSONDecodeError:
                continue
        generation = _read_json(shard / "generation_result.json", {})
        requested_regimes.update(generation.get("requested_overlap_regime_counts", {}))
        realized_regimes.update(generation.get("realized_overlap_regime_counts", {}))
        shard_bytes = sum(path.stat().st_size for path in shard.rglob("*") if path.is_file())
        total_bytes += shard_bytes
        shard_records.append({
            "scene_id": scene_id,
            "path": str(shard.relative_to(root)),
            "configuration_count": len(configurations),
            "episode_count": len(episodes),
            "bytes": shard_bytes,
            "configuration_fingerprint": expected_fingerprint,
        })
    taxonomy = merge_taxonomies(taxonomies)
    global_root = root / "global"
    dump_json(global_root / "taxonomy.json", taxonomy)
    dump_json(root / "taxonomy.json", taxonomy)
    index = {
        "shards": shard_records,
        "configurations": configuration_index,
        "episodes": episode_index,
    }
    dump_json(global_root / "dataset_index.json", index)
    total_realized = sum(realized_regimes.values())
    finalized = {
        **manifest,
        "status": "complete",
        "shard_count": len(shard_records),
        "configuration_count": len(configuration_index),
        "episode_count": len(episode_index),
        "requested_overlap_regime_counts": dict(requested_regimes),
        "realized_overlap_regime_counts": dict(realized_regimes),
        "realized_overlap_regime_fractions": {
            key: value / max(1, total_realized)
            for key, value in sorted(realized_regimes.items())
        },
        "intervention_type_counts": dict(intervention_types),
        "rejection_reason_counts": dict(rejection_reasons),
        "total_shard_bytes": total_bytes,
        "dense_data_duplicated_by_merge": False,
    }
    dump_json(global_root / "dataset_meta.json", finalized)
    dump_json(root / "dataset_meta.json", finalized)
    source_config = global_root / "resolved_config.yaml"
    destination_config = root / "resolved_config.yaml"
    if source_config.is_file() and not destination_config.is_file():
        destination_config.write_text(source_config.read_text(encoding="utf-8"), encoding="utf-8")
    dump_json(root / "generation_status.json", finalized)
    return global_root, finalized
