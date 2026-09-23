from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import pytest
import numpy as np

from multi_view_world_dataset.errors import ConfigurationError, SampleRejected
from multi_view_world_dataset.generator import (
    _adaptive_exact_validation,
    _candidate_accounting,
    _configuration_navigation_seed,
    _gt_rescue_candidates,
    _gt_valid_candidate_soft_score,
    _temporal_overlap_preflight,
)
import multi_view_world_dataset.pilot_report as pilot_report_module
import multi_view_world_dataset.production as production_module
from multi_view_world_dataset.pilot_report import generate_pilot_report
from multi_view_world_dataset.production import (
    _exclusive_production_root,
    _quarantine_exhausted_empty_configuration,
    _should_launch_scene,
    finalize_dataset,
    initialize_production_root,
    launch_scene_shards,
    merge_taxonomies,
    scene_shard_path,
)
from multi_view_world_dataset.sampling.diversity import stable_seed, temporal_overlap_acceptance
from multi_view_world_dataset.sampling.splits import assign_scene_family_splits
from multi_view_world_dataset.scene_eligibility import (
    load_scene_eligibility,
    reconcile_scene_eligibility,
    validate_scene_family_split_disjointness,
)
from multi_view_world_dataset.storage.writer import DatasetWriter
from multi_view_world_dataset.utils.config import load_yaml_config
from multi_view_world_dataset.utils.runtime import RuntimePaths
from multi_view_world_dataset.utils.serialization import dump_json
import multi_view_world_dataset.utils.serialization as serialization_module


def _overlap(frames):
    return temporal_overlap_acceptance(
        ("a", "b", "c"), frames, regime="unclassified",
        regime_connected_fraction_target={
            "dense_shared": 0.60, "partial_chain": 0.30, "exploratory": 0.15,
        },
        regime_shared_keyframe_fraction_target={
            "dense_shared": 0.60, "partial_chain": 0.30, "exploratory": 0.15,
        },
        regime_maximum_consecutive_isolated_keyframes={
            "dense_shared": 10, "partial_chain": 10, "exploratory": 10,
        },
    )


def test_realized_overlap_regimes_include_temporal_three_edge_partial_chain():
    dense = _overlap([
        {"connected": True, "edges": [["a", "b"], ["b", "c"]]},
        {"connected": True, "edges": [["a", "c"], ["b", "c"]]},
        {"connected": False, "edges": [["a", "b"]]},
    ])
    assert dense["passed"] and dense["realized_regime"] == "dense_shared"

    true_chain = _overlap([
        {"connected": False, "edges": [["a", "b"]]},
        {"connected": False, "edges": [["b", "c"]]},
        {"connected": False, "edges": []},
    ])
    assert true_chain["passed"] and true_chain["realized_regime"] == "exploratory"
    assert true_chain["union_graph_is_tree"]

    temporal_triangle = _overlap([
        {"connected": False, "edges": [["a", "b"]]},
        {"connected": False, "edges": [["b", "c"]]},
        {"connected": False, "edges": [["a", "c"]]},
    ])
    assert temporal_triangle["passed"]
    assert temporal_triangle["realized_regime"] == "partial_chain"
    assert not temporal_triangle["union_graph_is_tree"]

    exploratory = _overlap([
        {"connected": False, "edges": [["a", "b"]]},
        {"connected": False, "edges": []},
        {"connected": False, "edges": [["b", "c"]]},
        {"connected": False, "edges": []},
        {"connected": False, "edges": []},
        {"connected": False, "edges": []},
        {"connected": False, "edges": []},
    ])
    assert exploratory["passed"]
    assert exploratory["realized_regime"] == "exploratory"

    disconnected = _overlap([
        {"connected": False, "edges": [["a", "b"]]},
        {"connected": False, "edges": [["a", "b"]]},
    ])
    assert not disconnected["passed"]
    assert not disconnected["checks"]["union_graph_connected"]


def _v11_overlap(frames, *, mode="sampled_keyframes", frame_count=60):
    return temporal_overlap_acceptance(
        ("a", "b", "c"), frames, regime="unclassified",
        regime_connected_fraction_target={
            "dense_shared": 0.60, "partial_chain": 0.30, "exploratory": 0.15,
        },
        regime_shared_keyframe_fraction_target={
            "dense_shared": 0.60, "partial_chain": 0.30, "exploratory": 0.15,
        },
        regime_maximum_consecutive_isolated_keyframes={
            "dense_shared": 2, "partial_chain": 5, "exploratory": 6,
        },
        regime_minimum_participating_keyframes={
            "dense_shared": 1, "partial_chain": 2, "exploratory": 1,
        },
        regime_maximum_isolation_fraction={
            "dense_shared": 1.0,
            "partial_chain": 5 / 7,
            "exploratory": 6 / 7,
        },
        isolation_decision_mode=mode,
        episode_frame_count=frame_count,
    )


def test_v11_partial_chain_accepts_review_borderline_semantics():
    frames = [
        {"frame_index": 0, "connected": True,
         "edges": [["a", "b"], ["b", "c"]]},
        {"frame_index": 10, "connected": False, "edges": [["a", "c"]]},
        {"frame_index": 20, "connected": False, "edges": [["a", "b"]]},
        *[
            {"frame_index": frame, "connected": False, "edges": []}
            for frame in (30, 39, 49, 59)
        ],
    ]
    result = _v11_overlap(frames)
    assert result["passed"]
    assert result["realized_regime"] == "partial_chain"
    assert result["participating_keyframe_count"] == {"a": 3, "b": 2, "c": 2}
    assert result["maximum_consecutive_isolated_keyframes"] == {
        "a": 4, "b": 4, "c": 5,
    }
    assert result["checks"]["union_graph_connected"]
    assert result["isolation_confirmation_recommended"]


def test_v11_exploratory_allows_one_anchor_but_not_zero_participation():
    frames = [
        {"frame_index": 0, "connected": False, "edges": [["a", "b"]]},
        *[
            {"frame_index": frame, "connected": False, "edges": []}
            for frame in (10, 20, 30, 39, 49)
        ],
        {"frame_index": 59, "connected": False, "edges": [["b", "c"]]},
    ]
    result = _v11_overlap(frames)
    assert result["passed"] and result["realized_regime"] == "exploratory"
    assert max(result["maximum_consecutive_isolated_keyframes"].values()) == 6

    never = _v11_overlap([
        {**frame, "edges": [["a", "b"]], "connected": False}
        for frame in frames
    ])
    assert not never["passed"]
    assert not never["checks"]["every_robot_participates"]
    assert "robot_never_participates" in never["failure_reasons"]


def test_dense_confirmation_uses_normalized_duration_not_sparse_raw_count():
    frames = []
    for index, frame in enumerate(np.rint(np.linspace(0, 59, 13)).astype(int)):
        edges = [["a", "b"]] if index < 8 else [["a", "c"]]
        frames.append({"frame_index": int(frame), "connected": False, "edges": edges})
    result = _v11_overlap(frames, mode="normalized_duration")
    assert result["realized_regime"] == "partial_chain"
    assert result["maximum_consecutive_isolated_keyframes"]["c"] == 8
    assert result["maximum_isolation_fraction"]["c"] <= 5 / 7
    assert result["passed"]


def test_dense_confirmation_falls_back_to_exploratory_instead_of_penalizing_overlap():
    frames = []
    for index, frame in enumerate(np.rint(np.linspace(0, 59, 13)).astype(int)):
        edges = [["a", "b"]]
        if index in {0, 1, 12}:
            edges.append(["a", "c"])
        frames.append({
            "frame_index": int(frame),
            "connected": len(edges) == 2,
            "edges": edges,
        })
    result = _v11_overlap(frames, mode="normalized_duration")
    assert result["partial_chain_topology_and_participation_met"]
    assert not result["partial_chain_isolation_met"]
    assert result["fell_back_to_exploratory_for_isolation"]
    assert result["realized_regime"] == "exploratory"
    assert result["maximum_isolation_fraction"]["c"] <= 6 / 7
    assert result["passed"]


def test_dense_confirmation_accepts_one_early_anchor_with_measured_span():
    keyframes = np.rint(np.linspace(0, 59, 13)).astype(int)
    frames = []
    for index, frame in enumerate(keyframes):
        edges = []
        if index == 1:
            edges = [["a", "b"], ["b", "c"]]
        elif index == 4:
            edges = [["a", "b"]]
        frames.append({
            "frame_index": int(frame),
            "connected": len(edges) == 2,
            "edges": edges,
        })
    result = _v11_overlap(frames, mode="normalized_duration")
    assert result["passed"]
    assert result["realized_regime"] == "exploratory"
    assert result["participating_keyframe_count"] == {"a": 2, "b": 2, "c": 1}
    assert result["maximum_consecutive_isolated_keyframes"]["c"] == 11
    assert result["maximum_isolation_fraction"]["c"] == pytest.approx(49 / 59)
    assert result["maximum_isolation_fraction"]["c"] <= 6 / 7


def test_sparse_isolation_boundary_triggers_dense_confirmation(monkeypatch):
    import multi_view_world_dataset.generator as generator_module

    edges_by_frame = {
        0: (("robot_00", "robot_01"), ("robot_01", "robot_02")),
        10: (("robot_00", "robot_02"),),
        20: (("robot_00", "robot_01"),),
        25: (("robot_01", "robot_02"),),
    }

    class Adapter:
        def __init__(self):
            self.frame = 0
            self.placed = []

        def place_robots_at_trajectory_frame(self, trajectories, frame_index):
            self.frame = int(frame_index)
            self.placed.append(self.frame)

        def robot_depth_observations(self):
            return {
                robot_id: {"depth_linear": np.ones((8, 16)), "camera_to_world": np.eye(4)}
                for robot_id in ("robot_00", "robot_01", "robot_02")
            }

    adapter = Adapter()

    def graph(*args, **kwargs):
        edges = edges_by_frame.get(adapter.frame, ())
        return SimpleNamespace(
            edges=edges,
            connected=len({node for edge in edges for node in edge}) == 3,
            near_duplicate_pairs=(),
            overlaps={tuple(sorted(edge)): 0.25 for edge in edges},
        )

    monkeypatch.setattr(generator_module, "build_overlap_graph", graph)
    monkeypatch.setattr(
        generator_module, "pairwise_shared_surface_centroid", lambda *args, **kwargs: None
    )
    config = load_yaml_config("configs/default.yaml")
    trajectories = tuple(
        SimpleNamespace(
            robot_id=robot_id, frames=60,
            metadata={"observation_regime": "partial_chain"},
        )
        for robot_id in ("robot_00", "robot_01", "robot_02")
    )
    result = _temporal_overlap_preflight(adapter, config, trajectories)
    assert result["acceptance_source"] == "dense_13_confirmation"
    assert result["gt_validation_accounting"] == {
        "sparse_7_validations": 1, "dense_13_confirmations": 1,
    }
    assert "sparse_validation" in result


def test_union_disconnected_does_not_trigger_dense_confirmation(monkeypatch):
    import multi_view_world_dataset.generator as generator_module

    class Adapter:
        def __init__(self): self.placed = []
        def place_robots_at_trajectory_frame(self, trajectories, frame_index):
            self.placed.append(int(frame_index))
        def robot_depth_observations(self):
            return {
                robot_id: {"depth_linear": np.ones((8, 16)), "camera_to_world": np.eye(4)}
                for robot_id in ("robot_00", "robot_01", "robot_02")
            }

    monkeypatch.setattr(generator_module, "build_overlap_graph", lambda *args, **kwargs: SimpleNamespace(
        edges=(("robot_00", "robot_01"),), connected=False,
        near_duplicate_pairs=(), overlaps={("robot_00", "robot_01"): 0.25},
    ))
    monkeypatch.setattr(
        generator_module, "pairwise_shared_surface_centroid", lambda *args, **kwargs: None
    )
    config = load_yaml_config("configs/default.yaml")
    trajectories = tuple(
        SimpleNamespace(robot_id=robot_id, frames=60, metadata={})
        for robot_id in ("robot_00", "robot_01", "robot_02")
    )
    adapter = Adapter()
    with pytest.raises(SampleRejected) as caught:
        _temporal_overlap_preflight(adapter, config, trajectories)
    assert "union_disconnected" in caught.value.details["failure_reasons"]
    assert caught.value.details["gt_validation_accounting"]["dense_13_confirmations"] == 0
    assert len(adapter.placed) == 8


def test_adaptive_exact_validation_expands_batches_without_duplicates():
    attempted = []

    def validate(candidate, rank, batch):
        attempted.append((rank, batch))
        if rank not in {15, 20}:
            raise SampleRejected("no_gt_overlap", {"rank": rank})
        return candidate

    valid, diagnostics = _adaptive_exact_validation(
        list(range(48)), [12, 24, 48], validate
    )
    assert valid == [15, 20]
    assert diagnostics["accepted_batch_limit"] == 24
    assert diagnostics["total_exact_candidates_tested"] == 24
    assert len({rank for rank, _ in attempted}) == len(attempted) == 24
    assert all(batch == 12 for _, batch in attempted[:12])
    assert all(batch == 24 for _, batch in attempted[12:])


def test_adaptive_exact_validation_stops_after_first_valid_batch():
    attempted = []

    def validate(candidate, rank, batch):
        attempted.append(rank)
        if rank != 3:
            raise SampleRejected("invalid")
        return candidate

    valid, diagnostics = _adaptive_exact_validation(
        list(range(48)), [12, 24, 48], validate
    )
    assert valid == [3]
    assert diagnostics["accepted_batch_limit"] == 12
    assert attempted == list(range(12))


def test_candidate_accounting_matches_raw_candidate_records():
    records = [
        {"validation_stage": "adaptive_base", "sparse_7_validations": 1,
         "dense_13_confirmations": 0},
        {"validation_stage": "adaptive_base", "sparse_7_validations": 1,
         "dense_13_confirmations": 1},
        {"validation_stage": "gt_rescue", "sparse_7_validations": 1,
         "dense_13_confirmations": 1},
    ]
    result = _candidate_accounting(
        base_candidates_generated=48,
        candidate_records=records,
        rescue_candidate_counts={"measured_overlap_route_mutation": 1},
        duplicate_candidates_removed=4,
        accepted_candidate_source="measured_overlap_route_mutation:gt_rescue",
    )
    assert result["base_candidates_exact_gt_validated"] == 2
    assert result["rescue_candidates_exact_gt_validated"] == 1
    assert result["sparse_7_validations"] == 3
    assert result["dense_13_confirmations"] == 2
    assert result["exact_gt_candidate_record_count"] == 3
    assert result["totals_consistent"]



def test_gt_rescue_candidates_merge_bounded_builders_and_isolate_failure():
    calls = []

    class Adapter:
        def complementary_trajectory_hybrids(self, candidates, failures):
            calls.append(("complementary", candidates, failures))
            return ("hybrid",)

        def measured_overlap_bridge_trajectories(self, candidates, failures, seed):
            calls.append(("bridge", candidates, failures, seed))
            raise SampleRejected("bridge_generation_failed", {"seed": seed})

    candidates = (("base_0", {}), ("base_1", {}))
    failures = [{"candidate_rank": 0, "reason": "temporal"}]
    generated, generation_failures = _gt_rescue_candidates(
        Adapter(), candidates, failures, 123
    )
    assert generated == ("hybrid",)
    assert [item[0] for item in calls] == ["complementary", "bridge"]
    assert calls[1][3] == 123
    assert generation_failures == [{
        "candidate_kind": "measured_overlap_bridge",
        "reason": "bridge_generation_failed",
        "details": {"seed": 123},
    }]


def test_soft_regime_preference_is_a_bonus_not_a_gate():
    weights = {"dense_shared": 0.30, "partial_chain": 0.50, "exploratory": 0.20}
    counts = Counter({"dense_shared": 7, "exploratory": 3})
    partial = _gt_valid_candidate_soft_score(
        {"cheap_scene_visibility": {"score": 0.0}}, 1, 3,
        "partial_chain", weights, counts, Counter(counts), lambda_regime=1.0,
    )
    dense = _gt_valid_candidate_soft_score(
        {"cheap_scene_visibility": {"score": 0.0}}, 0, 3,
        "dense_shared", weights, counts, Counter(counts), lambda_regime=1.0,
    )
    assert partial["final_score"] > dense["final_score"]
    assert dense["quality_score"] > partial["quality_score"]


def test_scene_eligibility_filters_and_reconciles_installed_catalog():
    manifest = load_scene_eligibility("configs/scene_eligibility.yaml")
    assert len(manifest.records) == 51
    assert len(manifest.eligible_scene_ids) == 50
    assert manifest.excluded_scene_ids == ("Wainscott_0_garden",)
    excluded = manifest.by_scene["Wainscott_0_garden"]
    assert "no footprint-safe navigable state" in excluded.reason
    result = reconcile_scene_eligibility(sorted(manifest.by_scene), manifest)
    assert "Wainscott_0_garden" not in result["eligible_scenes"]
    assert result["excluded_scenes"][0]["scene_id"] == "Wainscott_0_garden"
    with pytest.raises(ConfigurationError, match="missing_from_manifest"):
        reconcile_scene_eligibility([*manifest.by_scene, "new_scene"], manifest)
    with pytest.raises(ConfigurationError, match="not_installed"):
        reconcile_scene_eligibility(list(manifest.by_scene)[:-1], manifest)


def test_primary_split_is_built_only_from_eligible_scenes_and_is_family_disjoint():
    manifest = load_scene_eligibility("configs/scene_eligibility.yaml")
    splits = assign_scene_family_splits(
        list(manifest.eligible_scene_ids), {"train": 0.8, "val": 0.1, "test": 0.1}, 1907
    )
    assert set(splits) == set(manifest.eligible_scene_ids)
    assert "Wainscott_0_garden" not in splits
    validate_scene_family_split_disjointness(splits)
    with pytest.raises(ConfigurationError, match="leak"):
        validate_scene_family_split_disjointness({
            "Beechwood_0_int": "train", "Beechwood_1_int": "test",
        })


def test_scene_shard_paths_and_parent_resume_policy(tmp_path):
    assert scene_shard_path(tmp_path, "Rs_int") == tmp_path / "shards" / "Rs_int"
    with pytest.raises(ConfigurationError, match="Unsafe"):
        scene_shard_path(tmp_path, "../escape")
    assert not _should_launch_scene("complete", retry_failed=True)
    assert not _should_launch_scene("failed", retry_failed=False)
    assert _should_launch_scene("failed", retry_failed=True)
    assert _should_launch_scene("pending", retry_failed=False)


def _taxonomy(scene, semantic_id, category):
    return {
        "version": "Dataset-v1.1",
        "semantic_labels": {
            "0": {"name": "background", "reserved": True},
            "1": {"name": "unknown", "reserved": True},
            "2": {"name": "robot", "reserved": True},
            str(semantic_id): {"name": category, "reserved": False},
        },
        "instance_id_convention": {"scene_objects_start_at": 4},
        "instance_catalogs": {scene: [{"category": category}]},
    }


def test_taxonomy_merge_is_deterministic_and_detects_collisions():
    left = _taxonomy("A", 101, "chair")
    right = _taxonomy("B", 202, "table")
    assert merge_taxonomies([left, right]) == merge_taxonomies([right, left])
    with pytest.raises(ConfigurationError, match="collision"):
        merge_taxonomies([left, _taxonomy("B", 101, "table")])
    with pytest.raises(ConfigurationError, match="inconsistent IDs"):
        merge_taxonomies([left, _taxonomy("B", 999, "chair")])


def test_finalize_merges_metadata_without_copying_dense_episode_data(tmp_path):
    root = tmp_path / "dataset"
    fingerprint = "same-fingerprint"
    dump_json(root / "global" / "production_manifest.json", {
        "schema_version": "1.1.0",
        "configuration_fingerprint": fingerprint,
        "selected_scenes": ["A", "B"],
        "excluded_scenes": [{"scene_id": "X", "reason": "infeasible"}],
        "eligible_scenes": ["A", "B"],
    })
    (root / "global" / "resolved_config.yaml").write_text(
        "sampling_diagnostics: {}\n", encoding="utf-8"
    )
    for index, scene in enumerate(("A", "B"), start=1):
        shard = scene_shard_path(root, scene)
        dump_json(shard / "shard_status.json", {"status": "complete"})
        dump_json(shard / "dataset_meta.json", {
            "configuration_fingerprint": fingerprint, "schema_version": "1.1.0",
        })
        dump_json(shard / "taxonomy.json", _taxonomy(scene, 100 + index, f"cat_{scene}"))
        dump_json(shard / "configurations" / scene / "config_000" / "config_meta.json", {})
        episode = shard / "episodes" / scene / "config_000" / "episode_000"
        dump_json(episode / "meta.json", {})
        dump_json(episode / "events.json", [{"intervention_type": "rigid_relocation"}])
        (episode / "dense.bin").write_bytes(b"dense-data")
        dump_json(shard / "generation_result.json", {
            "requested_overlap_regime_counts": {"partial_chain": 1},
            "realized_overlap_regime_counts": {"partial_chain": 1},
        })
    _, result = finalize_dataset(root)
    assert result["shard_count"] == 2
    assert result["episode_count"] == 2
    assert result["dense_data_duplicated_by_merge"] is False
    assert len(json.loads((root / "global" / "dataset_index.json").read_text())["episodes"]) == 2
    assert not (root / "episodes").exists()
    first = json.loads((root / "global" / "dataset_meta.json").read_text())
    finalize_dataset(root)
    second = json.loads((root / "global" / "dataset_meta.json").read_text())
    assert first == second


def test_shard_finalize_refuses_fingerprint_mismatch(tmp_path):
    root = tmp_path / "dataset"
    dump_json(root / "global" / "production_manifest.json", {
        "schema_version": "1.1.0",
        "configuration_fingerprint": "expected",
        "selected_scenes": ["A"],
        "excluded_scenes": [],
        "eligible_scenes": ["A"],
    })
    shard = scene_shard_path(root, "A")
    dump_json(shard / "shard_status.json", {"status": "complete"})
    dump_json(shard / "dataset_meta.json", {
        "configuration_fingerprint": "wrong", "schema_version": "1.1.0",
    })
    with pytest.raises(ConfigurationError, match="fingerprint mismatch"):
        finalize_dataset(root)


def test_writer_recovers_only_atomic_partial_directories_after_fingerprint_check(tmp_path):
    writer = DatasetWriter(tmp_path / "dataset")
    partial = writer.root / "episodes" / "A" / "config_000" / ".episode_000.dead"
    partial.mkdir(parents=True)
    preserved = writer.root / "episodes" / "A" / "config_000" / "notes"
    preserved.mkdir()
    writer.initialize({"schema_version": "1.1.0"})
    assert not partial.exists()
    assert preserved.exists()


def test_production_scale_configs_share_research_semantics():
    production = load_yaml_config("configs/production_v1.yaml")
    integration = load_yaml_config("configs/integration_final.yaml")
    pilot = load_yaml_config("configs/pilot_production.yaml")
    for config in (production, integration, pilot):
        assert config["dataset"]["robots"] == 3
        assert config["dataset"]["frames"] == 60
        assert config["dataset"]["fps"] == 10
        assert config["robot"]["final_model"] == "mobile_sensor_robot_v1"
        assert config["robot"]["use_final_robot"]
        assert config["configuration_sampling"]["minimum_changed_objects"] == 2
        assert config["configuration_sampling"]["maximum_changed_objects"] == 6
        assert config["navigation"]["exact_validation_batches"] == [12, 24, 48]
        assert config["overlap"]["edge_threshold"] == 0.20
    assert integration["production"]["selected_scenes"] == ["Beechwood_0_int"]
    assert pilot["production"]["selected_scenes"] == ["Beechwood_0_int", "Rs_int"]


def test_production_manifest_persists_excluded_scene_and_split_metadata(tmp_path):
    config = load_yaml_config("configs/integration_final.yaml")
    repository_root = Path(__file__).resolve().parents[1]
    manifest = initialize_production_root(tmp_path, config, repository_root)
    assert len(manifest["eligible_scenes"]) == 50
    assert manifest["selected_scenes"] == ["Beechwood_0_int"]
    assert manifest["excluded_scenes"] == [{
        "scene_id": "Wainscott_0_garden",
        "reason": "no footprint-safe navigable state for mobile_sensor_robot_v1 under final navigation semantics",
        "feasibility_class": "infeasible",
    }]
    assert set(manifest["split_mapping"]) == set(manifest["eligible_scenes"])
    assert "Wainscott_0_garden" not in manifest["split_mapping"]


def test_parent_launcher_is_only_global_status_writer_and_records_worker_result(
    tmp_path, monkeypatch,
):
    config = load_yaml_config("configs/integration_final.yaml")
    runtime = RuntimePaths(tmp_path / "behavior", tmp_path / "dataset", tmp_path / "cache")

    def fake_run(command, **kwargs):
        assert kwargs["env"]["OMNIGIBSON_GPU_ID"] == "7"
        assert "CUDA_VISIBLE_DEVICES" not in kwargs["env"]
        scene = command[command.index("--scene") + 1]
        dump_json(scene_shard_path(runtime.output_root, scene) / "shard_status.json", {
            "scene_id": scene, "status": "complete", "accepted_episodes": 15,
        })
        dump_json(scene_shard_path(runtime.output_root, scene) / "generation_status.json", {
            "status": "pass", "accepted_configurations": 5,
            "accepted_episodes": 15,
        })
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(production_module.subprocess, "run", fake_run)
    _, result = launch_scene_shards(
        runtime, config, "configs/integration_final.yaml",
        gpus=("7",), max_workers=1, allow_large=False,
    )
    assert result["status"] == "pass"
    assert result["complete_scenes"] == ["Beechwood_0_int"]
    status = json.loads((runtime.output_root / "production_status.json").read_text())
    assert status["scenes"]["Beechwood_0_int"]["assigned_gpu"] == "7"
    assert status["full_production_started"] is False


def test_launcher_does_not_retry_worker_that_wrote_no_new_status(tmp_path, monkeypatch):
    config = load_yaml_config("configs/integration_final.yaml")
    config["generation"]["maximum_scene_sampling_restarts"] = 3
    runtime = RuntimePaths(tmp_path / "behavior", tmp_path / "dataset", tmp_path / "cache")
    shard = scene_shard_path(runtime.output_root, "Beechwood_0_int")
    dump_json(shard / "generation_status.json", {
        "status": "error", "error": "worker_progress_stalled",
        "accepted_episodes": 19,
    })
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(production_module.subprocess, "run", fake_run)
    _, result = launch_scene_shards(
        runtime, config, "configs/integration_final.yaml",
        gpus=("7",), max_workers=1, allow_large=False,
    )
    assert result["status"] == "error"
    assert len(calls) == 1
    status = json.loads((shard / "generation_status.json").read_text())
    assert status["error"] == "worker_exited_without_status_update"
    assert len(json.loads((shard / "sampling_recovery.json").read_text())["attempts"]) == 1


def test_parent_refuses_stale_complete_marker_after_worker_exits(tmp_path, monkeypatch):
    config = load_yaml_config("configs/integration_final.yaml")
    config["generation"]["maximum_scene_sampling_restarts"] = 0
    runtime = RuntimePaths(tmp_path / "behavior", tmp_path / "dataset", tmp_path / "cache")
    shard = scene_shard_path(runtime.output_root, "Beechwood_0_int")
    dump_json(shard / "shard_status.json", {
        "status": "complete", "accepted_configurations": 5, "accepted_episodes": 15,
    })
    dump_json(shard / "generation_status.json", {
        "status": "pass", "accepted_configurations": 5, "accepted_episodes": 15,
    })
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(production_module.subprocess, "run", fake_run)
    _, result = launch_scene_shards(
        runtime, config, "configs/integration_final.yaml",
        gpus=("7",), max_workers=1, allow_large=False,
    )
    assert len(calls) == 1
    assert result["status"] == "error"
    assert json.loads((shard / "shard_status.json").read_text())["status"] == "failed"
    assert json.loads((shard / "generation_status.json").read_text())["error"] == (
        "worker_exited_without_status_update"
    )


def test_launcher_stops_retry_when_generator_source_changes(tmp_path, monkeypatch):
    config = load_yaml_config("configs/integration_final.yaml")
    config["generation"]["maximum_scene_sampling_restarts"] = 3
    runtime = RuntimePaths(tmp_path / "behavior", tmp_path / "dataset", tmp_path / "cache")
    shard = scene_shard_path(runtime.output_root, "Beechwood_0_int")
    original_fingerprint = production_module._target_fingerprint
    source_changed = {"value": False}

    def fingerprint(*args, **kwargs):
        if source_changed["value"]:
            return "changed-source-fingerprint"
        return original_fingerprint(*args, **kwargs)

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        dump_json(shard / "shard_status.json", {
            "status": "failed", "error": "episode_before_attempts_exhausted",
        })
        dump_json(shard / "generation_status.json", {
            "status": "error", "error": "episode_before_attempts_exhausted",
        })
        source_changed["value"] = True
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(production_module, "_target_fingerprint", fingerprint)
    monkeypatch.setattr(production_module.subprocess, "run", fake_run)
    _, result = launch_scene_shards(
        runtime, config, "configs/integration_final.yaml",
        gpus=("7",), max_workers=1, allow_large=False,
    )
    assert result["status"] == "error"
    assert len(calls) == 1
    scene = result["scenes"]["Beechwood_0_int"]
    assert scene["retry_blocked"] == "generator_source_changed_during_run"
    assert len(scene["sampling_recovery_history"]) == 1


def test_scene_worker_turns_generator_error_result_into_failed_shard(
    tmp_path, monkeypatch,
):
    config = load_yaml_config("configs/integration_final.yaml")
    runtime = RuntimePaths(tmp_path / "behavior", tmp_path / "dataset", tmp_path / "cache")

    def fake_generate(*args, **kwargs):
        return tmp_path / "dataset", {
            "status": "error",
            "error": "episode_before_attempts_exhausted",
        }

    monkeypatch.setattr(
        "multi_view_world_dataset.generator.generate_dataset", fake_generate
    )
    with pytest.raises(RuntimeError, match="episode_before_attempts_exhausted"):
        production_module.run_scene_worker(
            runtime, config, "Beechwood_0_int", allow_large=False
        )
    status = json.loads(
        (scene_shard_path(runtime.output_root, "Beechwood_0_int")
         / "shard_status.json").read_text()
    )
    assert status["status"] == "failed"
def test_parent_refuses_zero_exit_without_complete_shard_status(tmp_path, monkeypatch):
    config = load_yaml_config("configs/integration_final.yaml")
    runtime = RuntimePaths(tmp_path / "behavior", tmp_path / "dataset", tmp_path / "cache")

    def fake_run(command, **kwargs):
        scene = command[command.index("--scene") + 1]
        dump_json(scene_shard_path(runtime.output_root, scene) / "shard_status.json", {
            "scene_id": scene, "status": "running",
        })
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(production_module.subprocess, "run", fake_run)
    _, result = launch_scene_shards(
        runtime, config, "configs/integration_final.yaml",
        gpus=("7",), max_workers=1, allow_large=False,
    )
    assert result["status"] == "error"
    assert result["failed_scenes"] == ["Beechwood_0_int"]
    assert "without a complete shard status" in (
        result["scenes"]["Beechwood_0_int"]["parent_error"]
    )


def test_parent_reconciles_fast_shutdown_generator_failure(tmp_path, monkeypatch):
    config = load_yaml_config("configs/integration_final.yaml")
    config["generation"]["maximum_scene_sampling_restarts"] = 0
    runtime = RuntimePaths(tmp_path / "behavior", tmp_path / "dataset", tmp_path / "cache")

    def fake_run(command, **kwargs):
        scene = command[command.index("--scene") + 1]
        shard = scene_shard_path(runtime.output_root, scene)
        dump_json(shard / "shard_status.json", {
            "scene_id": scene,
            "status": "running",
            "configuration_fingerprint": production_module._target_fingerprint(
                config, Path(__file__).resolve().parents[1]
            ),
        })
        dump_json(shard / "generation_status.json", {
            "status": "error",
            "error_type": "SampleRejected",
            "error": "episode_before_attempts_exhausted",
            "accepted_configurations": 1,
            "accepted_episodes": 1,
        })
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(production_module.subprocess, "run", fake_run)
    _, result = launch_scene_shards(
        runtime, config, "configs/integration_final.yaml",
        gpus=("7",), max_workers=1, allow_large=False,
    )
    assert result["status"] == "error"
    shard_status = json.loads(
        (scene_shard_path(runtime.output_root, "Beechwood_0_int")
         / "shard_status.json").read_text()
    )
    assert shard_status["status"] == "failed"
    assert shard_status["terminal_status_source"] == (
        "generation_status_parent_reconciliation"
    )
    assert shard_status["error"] == "episode_before_attempts_exhausted"


def test_parent_automatically_recovers_sampling_exhaustion(tmp_path, monkeypatch):
    config = load_yaml_config("configs/integration_final.yaml")
    config["generation"]["maximum_scene_sampling_restarts"] = 2
    runtime = RuntimePaths(
        tmp_path / "behavior", tmp_path / "dataset", tmp_path / "cache"
    )
    observed_epochs = []

    def fake_run(command, **kwargs):
        scene = command[command.index("--scene") + 1]
        epoch = int(command[command.index("--sampling-retry-epoch") + 1])
        observed_epochs.append(epoch)
        shard = scene_shard_path(runtime.output_root, scene)
        if epoch == 0:
            dump_json(shard / "shard_status.json", {
                "scene_id": scene,
                "status": "running",
            })
            dump_json(shard / "generation_status.json", {
                "status": "error",
                "error_type": "SampleRejected",
                "error": "episode_before_attempts_exhausted",
                "accepted_configurations": 1,
                "accepted_episodes": 1,
                "sampling_retry_epoch": epoch,
            })
        else:
            dump_json(shard / "shard_status.json", {
                "scene_id": scene,
                "status": "complete",
                "accepted_configurations": 5,
                "accepted_episodes": 15,
                "sampling_retry_epoch": epoch,
            })
            dump_json(shard / "generation_status.json", {
                "status": "pass",
                "accepted_configurations": 5,
                "accepted_episodes": 15,
                "sampling_retry_epoch": epoch,
            })
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(production_module.subprocess, "run", fake_run)
    _, result = launch_scene_shards(
        runtime,
        config,
        "configs/integration_final.yaml",
        gpus=("7",),
        max_workers=1,
        allow_large=False,
    )

    assert result["status"] == "pass"
    assert observed_epochs == [0, 1]
    scene_result = result["scenes"]["Beechwood_0_int"]
    assert scene_result["sampling_restart_count"] == 1
    assert [
        item["error"] for item in scene_result["sampling_recovery_history"]
    ] == ["episode_before_attempts_exhausted", None]
    recovery = json.loads(
        (scene_shard_path(runtime.output_root, "Beechwood_0_int")
         / "sampling_recovery.json").read_text()
    )
    assert recovery["complete"] is True
    assert len(recovery["attempts"]) == 2


def test_configuration_navigation_seed_survives_resume():
    saved_configuration = {"seed": 4114367871, "accepted_attempt": 3}
    assert _configuration_navigation_seed(saved_configuration) == stable_seed(
        saved_configuration["seed"], "configuration-navigation-context"
    )


def test_retry_failed_continues_after_persisted_epoch(tmp_path, monkeypatch):
    config = load_yaml_config("configs/integration_final.yaml")
    config["generation"]["maximum_scene_sampling_restarts"] = 0
    runtime = RuntimePaths(tmp_path / "behavior", tmp_path / "dataset", tmp_path / "cache")
    initialize_production_root(runtime.output_root, config, Path(__file__).resolve().parents[1])
    shard = scene_shard_path(runtime.output_root, "Beechwood_0_int")
    dump_json(shard / "shard_status.json", {"status": "failed"})
    dump_json(shard / "sampling_recovery.json", {
        "attempts": [{"sampling_retry_epoch": 3, "status": "failed"}],
    })
    observed_epochs = []

    def fake_run(command, **kwargs):
        epoch = int(command[command.index("--sampling-retry-epoch") + 1])
        observed_epochs.append(epoch)
        dump_json(shard / "shard_status.json", {
            "status": "complete", "accepted_configurations": 5,
            "accepted_episodes": 15,
        })
        dump_json(shard / "generation_status.json", {
            "status": "pass", "accepted_configurations": 5,
            "accepted_episodes": 15,
        })
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(production_module.subprocess, "run", fake_run)
    _, result = launch_scene_shards(
        runtime, config, "configs/integration_final.yaml",
        gpus=("7",), max_workers=1, allow_large=False, retry_failed=True,
    )
    assert result["status"] == "pass"
    assert observed_epochs == [4]
    assert [item["sampling_retry_epoch"] for item in json.loads(
        (shard / "sampling_recovery.json").read_text()
    )["attempts"]] == [3, 4]


def test_scene_workers_are_serial_on_each_gpu(tmp_path, monkeypatch):
    config = load_yaml_config("configs/integration_final.yaml")
    config["production"]["selected_scenes"] = [
        "Beechwood_0_int", "Beechwood_1_int", "Rs_int",
    ]
    runtime = RuntimePaths(tmp_path / "behavior", tmp_path / "dataset", tmp_path / "cache")
    lock = Lock()
    active = {"0": 0, "1": 0}
    maximum_active = {"0": 0, "1": 0}
    assignments = {}

    def fake_run(command, **kwargs):
        scene = command[command.index("--scene") + 1]
        gpu = kwargs["env"]["OMNIGIBSON_GPU_ID"]
        with lock:
            active[gpu] += 1
            maximum_active[gpu] = max(maximum_active[gpu], active[gpu])
            assignments[scene] = gpu
        time.sleep(0.02)
        shard = scene_shard_path(runtime.output_root, scene)
        dump_json(shard / "shard_status.json", {
            "status": "complete", "accepted_configurations": 5,
            "accepted_episodes": 15,
        })
        dump_json(shard / "generation_status.json", {
            "status": "pass", "accepted_configurations": 5,
            "accepted_episodes": 15,
        })
        with lock:
            active[gpu] -= 1
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(production_module.subprocess, "run", fake_run)
    _, result = launch_scene_shards(
        runtime, config, "configs/integration_final.yaml",
        gpus=("0", "1"), max_workers=2, allow_large=False,
    )
    assert result["status"] == "pass"
    assert maximum_active == {"0": 1, "1": 1}
    assert assignments == {
        "Beechwood_0_int": "0", "Beechwood_1_int": "1", "Rs_int": "0",
    }


def test_full_production_batches_keep_one_manifest_and_completed_shards(tmp_path, monkeypatch):
    config = load_yaml_config("configs/production_v1.yaml")
    runtime = RuntimePaths(tmp_path / "behavior", tmp_path / "dataset", tmp_path / "cache")
    calls = []

    def fake_run(command, **kwargs):
        scene = command[command.index("--scene") + 1]
        calls.append(scene)
        shard = scene_shard_path(runtime.output_root, scene)
        dump_json(shard / "shard_status.json", {
            "status": "complete", "accepted_configurations": 150,
            "accepted_episodes": 450,
        })
        dump_json(shard / "generation_status.json", {
            "status": "pass", "accepted_configurations": 150,
            "accepted_episodes": 450,
        })
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(production_module.subprocess, "run", fake_run)
    with pytest.raises(ConfigurationError, match="explicit --scenes"):
        launch_scene_shards(
            runtime, config, "configs/production_v1.yaml",
            gpus=("0",), max_workers=1, allow_large=True,
        )
    _, first = launch_scene_shards(
        runtime, config, "configs/production_v1.yaml",
        gpus=("0",), max_workers=1, allow_large=True,
        scene_ids=("Beechwood_0_int",),
    )
    assert first["status"] == "partial"
    assert first["complete_scenes"] == ["Beechwood_0_int"]
    _, second = launch_scene_shards(
        runtime, config, "configs/production_v1.yaml",
        gpus=("1",), max_workers=1, allow_large=True,
        scene_ids=("Rs_int",),
    )
    assert second["status"] == "partial"
    assert second["complete_scenes"] == ["Beechwood_0_int", "Rs_int"]
    assert calls == ["Beechwood_0_int", "Rs_int"]
    manifest = json.loads(
        (runtime.output_root / "global" / "production_manifest.json").read_text()
    )
    assert len(manifest["selected_scenes"]) == 50
    assert manifest["configuration_fingerprint"] == second["configuration_fingerprint"]
    with pytest.raises(ConfigurationError, match="not selected"):
        launch_scene_shards(
            runtime, config, "configs/production_v1.yaml",
            gpus=("0",), max_workers=1, allow_large=True,
            scene_ids=("Wainscott_0_garden",),
        )


def test_partial_finalize_indexes_completed_scene_only(tmp_path):
    root = tmp_path / "dataset"
    config = load_yaml_config("configs/production_v1.yaml")
    repository_root = Path(__file__).resolve().parents[1]
    manifest = initialize_production_root(root, config, repository_root)
    scene = "Beechwood_0_int"
    shard = scene_shard_path(root, scene)
    dump_json(shard / "shard_status.json", {"status": "complete"})
    dump_json(shard / "dataset_meta.json", {
        "configuration_fingerprint": manifest["configuration_fingerprint"],
        "schema_version": manifest["schema_version"],
    })
    dump_json(shard / "taxonomy.json", _taxonomy(scene, 101, "chair"))
    dump_json(shard / "configurations" / scene / "config_000" / "config_meta.json", {})
    dump_json(shard / "episodes" / scene / "config_000" / "episode_000" / "meta.json", {})
    with pytest.raises(ConfigurationError, match="incomplete shard"):
        finalize_dataset(root)
    _, result = finalize_dataset(root, allow_partial=True)
    assert result["status"] == "partial"
    assert result["indexed_completed_scenes"] == [scene]
    assert result["pending_scene_count"] == 49
    assert result["episode_count"] == 1
    index = json.loads((root / "global" / "dataset_index.json").read_text())
    assert len(index["episodes"]) == 1


def test_partial_finalize_preserves_committed_episodes_across_interrupted_batches(tmp_path):
    root = tmp_path / "dataset"
    config = load_yaml_config("configs/production_v1.yaml")
    manifest = initialize_production_root(root, config, Path(__file__).resolve().parents[1])
    scene = "Beechwood_0_int"
    shard = scene_shard_path(root, scene)
    dump_json(shard / "shard_status.json", {"status": "failed", "error": "disk full"})
    dump_json(shard / "dataset_meta.json", {
        "configuration_fingerprint": manifest["configuration_fingerprint"],
        "schema_version": manifest["schema_version"],
    })
    dump_json(shard / "taxonomy.json", _taxonomy(scene, 101, "chair"))
    dump_json(shard / "configurations" / scene / "config_000" / "config_meta.json", {})

    def committed_episode(index, passed=True):
        episode = shard / "episodes" / scene / "config_000" / f"episode_{index:03d}"
        dump_json(episode / "meta.json", {"episode_id": episode.name})
        dump_json(episode / "qa.json", [{"check": "paired_trajectory_equality", "passed": passed}])
        dump_json(episode / "generation_metrics.json", {})
        (episode / "trajectories.npz").write_bytes(b"trajectory data")
        return episode

    first = committed_episode(0)
    _, first_index = finalize_dataset(root, allow_partial=True)
    assert first_index["status"] == "partial"
    assert first_index["indexed_partial_scenes"] == [scene]
    assert first_index["episode_count"] == 1
    first_bytes = (first / "trajectories.npz").read_bytes()

    committed_episode(1, passed=False)
    with pytest.raises(ConfigurationError, match="passing QA"):
        finalize_dataset(root, allow_partial=True)
    dump_json(shard / "episodes" / scene / "config_000" / "episode_001" / "qa.json", [
        {"check": "paired_trajectory_equality", "passed": True},
    ])
    _, second_index = finalize_dataset(root, allow_partial=True)
    assert second_index["episode_count"] == 2
    assert (first / "trajectories.npz").read_bytes() == first_bytes
    episodes = json.loads((root / "global" / "dataset_index.json").read_text())["episodes"]
    assert [entry["episode_id"] for entry in episodes] == ["episode_000", "episode_001"]


def test_atomic_json_replace_failure_preserves_previous_status(tmp_path, monkeypatch):
    status_path = tmp_path / "generation_status.json"
    dump_json(status_path, {"accepted_episodes": 6})

    def interrupted_replace(source, target):
        raise OSError("interrupted before commit")

    monkeypatch.setattr(serialization_module.os, "replace", interrupted_replace)
    with pytest.raises(OSError, match="interrupted before commit"):
        dump_json(status_path, {"accepted_episodes": 7})
    assert json.loads(status_path.read_text()) == {"accepted_episodes": 6}
    assert not list(tmp_path.glob(".generation_status.json.*.tmp"))


def test_production_root_rejects_concurrent_launcher(tmp_path):
    root = tmp_path / "dataset"
    with _exclusive_production_root(root):
        with pytest.raises(ConfigurationError, match="already using"):
            with _exclusive_production_root(root):
                pass


def test_empty_configuration_is_archived_only_after_repeated_failure(tmp_path):
    shard = tmp_path / "shards" / "Rs_int"
    configuration = shard / "configurations" / "Rs_int" / "config_002"
    dump_json(configuration / "config_meta.json", {"seed": 4114367871})
    (configuration / "simulator_state.npy").write_bytes(b"saved snapshot")
    failure = {
        "error": "episode_before_attempts_exhausted",
        "rejection_details": {
            "scene_id": "Rs_int", "configuration_id": "config_002",
            "episode_id": "episode_000",
        },
    }
    attempts = [
        {
            "error": "episode_before_attempts_exhausted",
            "failure_configuration_id": "config_002",
            "failure_configuration_seed": 4114367871,
        }
        for _ in range(2)
    ]
    assert _quarantine_exhausted_empty_configuration(
        shard, failure, attempts[:1], minimum_failed_epochs=2,
        sampling_retry_epoch=0,
    ) is None
    assert configuration.is_dir()

    archive = _quarantine_exhausted_empty_configuration(
        shard, failure, attempts, minimum_failed_epochs=2,
        sampling_retry_epoch=1,
    )
    assert archive is not None
    assert not configuration.exists()
    assert (shard / archive / "simulator_state.npy").read_bytes() == b"saved snapshot"
    assert json.loads((shard / archive / "sampling_quarantine.json").read_text())[
        "failed_epochs"
    ] == 2


def test_geometry_mismatch_archives_only_episode_free_configuration(tmp_path):
    shard = tmp_path / "shards" / "Rs_int"
    configuration = shard / "configurations" / "Rs_int" / "config_002"
    dump_json(configuration / "config_meta.json", {"seed": 42})
    (configuration / "simulator_state.npy").write_bytes(b"saved snapshot")
    failure = {
        "error": "configuration_snapshot_geometry_mismatch",
        "rejection_details": {
            "scene_id": "Rs_int", "configuration_id": "config_002",
        },
    }
    attempts = [{
        "error": failure["error"],
        "failure_configuration_id": "config_002",
        "failure_configuration_seed": 42,
    }]
    archive = _quarantine_exhausted_empty_configuration(
        shard, failure, attempts, minimum_failed_epochs=2,
        sampling_retry_epoch=0,
    )
    assert archive is not None
    assert (shard / archive / "simulator_state.npy").read_bytes() == b"saved snapshot"
    assert not configuration.exists()


def test_configuration_with_completed_episode_is_never_archived(tmp_path):
    shard = tmp_path / "shards" / "Rs_int"
    configuration = shard / "configurations" / "Rs_int" / "config_002"
    dump_json(configuration / "config_meta.json", {"seed": 42})
    episode = shard / "episodes" / "Rs_int" / "config_002" / "episode_000"
    dump_json(episode / "meta.json", {"episode_id": "episode_000"})
    failure = {
        "error": "episode_before_attempts_exhausted",
        "rejection_details": {
            "scene_id": "Rs_int", "configuration_id": "config_002",
        },
    }
    attempts = [{
        "error": failure["error"],
        "failure_configuration_id": "config_002",
        "failure_configuration_seed": 42,
    }]
    assert _quarantine_exhausted_empty_configuration(
        shard, failure, attempts, minimum_failed_epochs=1,
        sampling_retry_epoch=0,
    ) is None
    assert configuration.is_dir()
    assert (episode / "meta.json").is_file()
    failure["error"] = "configuration_snapshot_geometry_mismatch"
    attempts[0]["error"] = failure["error"]
    assert _quarantine_exhausted_empty_configuration(
        shard, failure, attempts, minimum_failed_epochs=2,
        sampling_retry_epoch=1,
    ) is None
    assert configuration.is_dir()


def test_progress_watchdog_terminates_stalled_running_worker(
    tmp_path, monkeypatch,
):
    shard = tmp_path / "shard"
    dump_json(shard / "generation_status.json", {
        "status": "running",
        "stage": "sample_episode_intervention",
        "configuration_id": "config_002",
        "episode_id": "episode_001",
        "attempt": 0,
    })
    dump_json(shard / "shard_status.json", {
        "status": "running",
        "worker_pid": 424242,
    })
    killed = []
    monkeypatch.setattr(production_module.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    result = {}

    production_module._watch_worker_progress(
        shard,
        production_module.Event(),
        result,
        stall_timeout_s=0.005,
        poll_interval_s=0.001,
    )

    assert killed == [(424242, 15)]
    assert result["triggered"] is True
    assert result["stalled_stage"] == "sample_episode_intervention"
    assert result["stalled_configuration_id"] == "config_002"
    assert result["stalled_episode_id"] == "episode_001"


def test_pilot_report_projects_full_scale_and_explicitly_stops(tmp_path, monkeypatch):
    root = tmp_path / "pilot"
    config = load_yaml_config("configs/pilot_production.yaml")
    (root / "global").mkdir(parents=True)
    (root / "global" / "resolved_config.yaml").write_text(
        __import__("yaml").safe_dump(config), encoding="utf-8"
    )
    dump_json(root / "global" / "production_manifest.json", {
        "selected_scenes": ["Beechwood_0_int", "Rs_int"],
        "eligible_scenes": [f"scene_{index:02d}" for index in range(50)],
        "target_regime_distribution": {
            "dense_shared": 0.30, "partial_chain": 0.50, "exploratory": 0.20,
        },
        "scene_eligibility_records": {
            "Beechwood_0_int": {"feasibility_class": "healthy"},
            "Rs_int": {"feasibility_class": "constrained"},
        },
    })
    dump_json(root / "production_status.json", {"scenes": {
        "Beechwood_0_int": {"status": "complete"},
        "Rs_int": {"status": "complete"},
    }})
    for scene in ("Beechwood_0_int", "Rs_int"):
        dump_json(root / "shards" / scene / "shard_status.json", {
            "status": "complete", "accepted_episodes": 60,
        })

    def fake_summary(dataset_root, *, output_path=None):
        count = 120 if Path(dataset_root) == root else 60
        report = {
            "finalized_episode_count": count,
            "acceptance_efficiency": {
                "outer_motion_attempt_counts": {"0": count},
                "episode_before_attempts_exhausted_count": 0,
            },
            "gt_candidate_efficiency": {
                "accepted_batch_counts": {"12": count},
                "accepted_batch_fractions": {"12": 1.0},
                "candidate_pool_exhausted_reject_count": 0,
            },
            "trajectory": {"realized_regime_counts": {
                "dense_shared": count * 3 // 10,
                "partial_chain": count * 5 // 10,
                "exploratory": count * 2 // 10,
            }},
            "intervention": {"type_counts": {
                "rigid_relocation": count * 6 // 10,
                "articulation": count * 3 // 10,
                "state_change": count // 10,
            }},
            "configuration": {
                "changed_object_count_distribution": {
                    "2": 8, "3": 8, "4": 8, "5": 8, "6": 8,
                }
            },
            "rejections": {"reason_counts": {}},
            "overlap": {"union_connected_episode_count": count},
            "identity_and_calibration": {
                "complete_observation_metadata_episode_count": count,
                "complete_world_bev_calibration_episode_count": count,
            },
            "storage": {
                "bytes_per_episode": {"mean": 1.0, "p50": 1.0, "p90": 1.0},
                "bytes_per_configuration": {"mean": 1.0},
                "total_episode_bytes": count,
                "total_configuration_bytes": 40,
            },
            "runtime_s": {"total_episode_s": {"mean": 2.0}},
            "spatial": {},
            "collapse_warnings": [],
        }
        output = Path(output_path)
        dump_json(output, report)
        return output, report

    monkeypatch.setattr(pilot_report_module, "summarize_generated_dataset", fake_summary)
    output, report = generate_pilot_report(root)
    assert output.is_file()
    assert report["full_production_was_started"] is False
    assert report["full_scale_projection"]["episodes"] == 22500
    assert report["full_scale_projection"]["robot_view_frames"] == 8100000
    assert report["full_scale_projection"]["world_bev_frames"] == 2700000
    assert (root / "global" / "pilot_report.md").is_file()
    assert len(list((root / "global" / "plots").glob("*.svg"))) == 4
