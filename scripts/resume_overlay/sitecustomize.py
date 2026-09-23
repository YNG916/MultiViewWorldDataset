"""Audited resume-only overlay for the frozen Dataset-v1.1 producer."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from compat import can_skip_navigation_rebuild


def _install() -> None:
    root_value = os.environ.get("MVWD_RESUME_OVERLAY_ROOT")
    if not root_value:
        return
    root = Path(root_value).expanduser().resolve()
    frozen_root = root / "global" / "frozen_producer_469064fa"
    manifest = json.loads((root / "global" / "production_manifest.json").read_text(encoding="utf-8"))
    from multi_view_world_dataset.adapters.omnigibson import OmniGibsonAdapter
    from multi_view_world_dataset.utils.provenance import generator_source_fingerprint
    from multi_view_world_dataset.utils.serialization import dump_json

    module_path = Path(sys.modules[OmniGibsonAdapter.__module__].__file__).resolve()
    loaded_root = module_path.parents[3]
    if loaded_root != frozen_root:
        raise RuntimeError(f"Resume overlay requires frozen producer {frozen_root}, got {loaded_root}")
    source_hash = generator_source_fingerprint(loaded_root)
    if source_hash != manifest["generator_source_fingerprint"]:
        raise RuntimeError("Frozen producer source fingerprint differs from production manifest")
    overlay_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    original = OmniGibsonAdapter.prepare_navigation_context

    def prepare_navigation_context(self, configuration_token, seed, *, force=False):
        if force or self._scene_id is None:
            return original(self, configuration_token, seed, force=force)
        scene_id = str(self._scene_id)
        shard = Path(self.runtime.output_root).resolve()
        if shard != root / "shards" / scene_id:
            raise RuntimeError(f"Resume overlay received unexpected shard {shard}")
        token_map = getattr(self, "_mvwd_resume_token_map", None)
        if token_map is None:
            token_map = {}
            for meta_path in sorted((shard / "configurations" / scene_id).glob("config_*/config_meta.json")):
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                token_map[str(metadata["exact_state_hash"])] = meta_path.parent.name
            self._mvwd_resume_token_map = token_map
        configuration_id = token_map.get(str(configuration_token))
        requested_episodes = int(self.config["dataset"]["accepted_episodes_per_configuration"])
        if configuration_id is None or not can_skip_navigation_rebuild(
            shard, scene_id, configuration_id, requested_episodes
        ):
            return original(self, configuration_token, seed, force=force)

        # No episode in this configuration will execute; only skip its unused route bank.
        self._navigation_configuration_token = None
        self._navigation_contexts = {}
        audit_path = shard / "resume_overlay_audit.json"
        prior = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
        skipped = sorted(set(prior.get("skipped_configuration_ids", [])) | {configuration_id})
        dump_json(audit_path, {
            "mode": "skip_navigation_rebuild_for_committed_configurations_only",
            "overlay_sha256": overlay_hash,
            "generator_source_fingerprint": source_hash,
            "configuration_fingerprint": manifest["configuration_fingerprint"],
            "skipped_configuration_ids": skipped,
        })
        counts = getattr(self, "_mvwd_resume_counts", None)
        if counts is None:
            requested_configurations = int(self.config["dataset"]["accepted_configurations_per_scene"])
            configuration_ids = [f"config_{index:03d}" for index in range(requested_configurations)]
            accepted_configurations = sum(
                (shard / "configurations" / scene_id / item / "config_meta.json").is_file()
                for item in configuration_ids
            )
            accepted_episodes = sum(
                (shard / "episodes" / scene_id / item / f"episode_{index:03d}" / "meta.json").is_file()
                for item in configuration_ids for index in range(requested_episodes)
            )
            counts = (accepted_configurations, accepted_episodes)
            self._mvwd_resume_counts = counts
        dump_json(shard / "generation_status.json", {
            "status": "running",
            "stage": "resume_skip_completed_navigation_rebuild",
            "scene_id": scene_id,
            "configuration_id": configuration_id,
            "accepted_configurations": counts[0],
            "accepted_episodes": counts[1],
            "resume_overlay_sha256": overlay_hash,
        })
        print(f"[mvwd-resume-overlay] skipped {scene_id}/{configuration_id}", flush=True)
        return {}

    OmniGibsonAdapter.prepare_navigation_context = prepare_navigation_context
    print(f"[mvwd-resume-overlay] active frozen_source={source_hash} overlay={overlay_hash}", flush=True)


try:
    _install()
except Exception as error:
    print(f"[mvwd-resume-overlay] refusing startup: {error}", file=sys.stderr, flush=True)
    os._exit(78)
