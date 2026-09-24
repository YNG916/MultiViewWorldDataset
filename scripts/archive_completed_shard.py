"""Checksum-verified archive/restore of one completed production scene shard.

Never archive a live or partial scene. A coordinator lock prevents concurrent
production during transfer. Eviction is opt-in and happens only after a
verified archive copy and a durable archive record exist.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path


SCENE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@contextmanager
def _lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Active producer/archive lock: {path}") from error
        yield
    finally:
        os.close(descriptor)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_manifest(directory: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Shard contains a symlink: {path}")
        if path.is_file():
            result[str(path.relative_to(directory))] = _sha256(path)
        elif not path.is_dir():
            raise RuntimeError(f"Shard contains a special file: {path}")
    return result


def _rsync_copy(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-a", "--delete", f"{source}/", f"{destination}/"],
        check=True,
    )


def _rsync_verify(source: Path, destination: Path) -> None:
    result = subprocess.run(
        [
            "rsync", "-aicn", "--delete", "--out-format=%i %n",
            f"{source}/", f"{destination}/",
        ],
        check=True, capture_output=True, text=True,
    )
    if result.stdout.strip():
        raise RuntimeError(f"Archive copy differs from source:\n{result.stdout[:2000]}")


def _validate_complete_shard(dataset_root: Path, scene: str) -> tuple[Path, dict]:
    if not SCENE_ID.fullmatch(scene):
        raise ValueError(f"Invalid scene ID: {scene!r}")
    manifest = _read_json(dataset_root / "global" / "production_manifest.json")
    if scene not in manifest["selected_scenes"]:
        raise RuntimeError(f"Scene is not in the dataset manifest: {scene}")
    shard = dataset_root / "shards" / scene
    if not shard.is_dir() or shard.is_symlink():
        raise RuntimeError(f"Scene shard is absent or a symlink: {shard}")
    status = _read_json(shard / "shard_status.json")
    generation = _read_json(shard / "generation_status.json")
    metadata = _read_json(shard / "dataset_meta.json")
    if status.get("status") != "complete" or generation.get("status") != "pass":
        raise RuntimeError(f"Scene is not fully complete: {scene}")
    if metadata.get("configuration_fingerprint") != manifest["configuration_fingerprint"]:
        raise RuntimeError(f"Dataset fingerprint mismatch: {scene}")
    if metadata.get("schema_version") != manifest["schema_version"]:
        raise RuntimeError(f"Schema mismatch: {scene}")
    configs = list((shard / "configurations" / scene).glob("config_*/config_meta.json"))
    episodes = list((shard / "episodes" / scene).glob("config_*/episode_*/meta.json"))
    if len(configs) != int(status["accepted_configurations"]):
        raise RuntimeError(f"Configuration count mismatch: {scene}")
    if len(episodes) != int(status["accepted_episodes"]):
        raise RuntimeError(f"Episode count mismatch: {scene}")
    for metadata_path in episodes:
        episode = metadata_path.parent
        checks = _read_json(episode / "qa.json")
        if not isinstance(checks, list) or not checks or not all(
            check.get("passed") is True for check in checks
        ):
            raise RuntimeError(f"Episode QA failed or absent: {episode}")
        for name in (
            "generation_metrics.json", "trajectories.npz", "events.json",
            "state_before.json", "state_after.json",
            "bev/world_before.npz", "bev/world_after.npz",
            "bev/environment_after.npz",
        ):
            if not (episode / name).is_file():
                raise RuntimeError(f"Episode artifact missing: {episode / name}")
        for phase in ("before", "after"):
            observations = json.loads(
                (episode / f"observations_{phase}.json").read_text(encoding="utf-8")
            )
            if not isinstance(observations, list) or not observations:
                raise RuntimeError(f"Episode observations absent: {episode}/{phase}")
            referenced = {
                str(ref).split("::", 1)[0]
                for record in observations
                for ref in record["modality_refs"].values()
            }
            for relative in referenced:
                path = Path(relative)
                if path.is_absolute() or ".." in path.parts or not (episode / path).is_file():
                    raise RuntimeError(f"Episode modality archive missing: {episode / relative}")
    if any(path.name.startswith((".episode_", ".config_")) for path in shard.rglob(".*")):
        raise RuntimeError(f"Uncommitted staging directory in {shard}")
    return shard, {
        "scene_id": scene,
        "configuration_fingerprint": manifest["configuration_fingerprint"],
        "schema_version": manifest["schema_version"],
        "configuration_count": len(configs),
        "episode_count": len(episodes),
    }


def archive_scene(
    dataset_root: Path, archive_root: Path, scene: str, *,
    evict: bool = False, confirm_scene: str | None = None,
) -> dict:
    dataset_root = dataset_root.resolve()
    archive_root = archive_root.resolve()
    if archive_root == dataset_root or dataset_root in archive_root.parents:
        raise ValueError("Archive root must not be inside the active dataset root")
    if evict and confirm_scene != scene:
        raise ValueError("Eviction requires --confirm-scene matching --scene")
    with _lock(dataset_root / ".production.lock"), _lock(archive_root / ".archive.lock"):
        source, summary = _validate_complete_shard(dataset_root, scene)
        destination = archive_root / "shards" / scene
        staging = destination.with_name(f".{scene}.archive-incomplete")
        global_source = dataset_root / "global"
        global_destination = archive_root / "global"
        archived_manifest = global_destination / "production_manifest.json"
        if archived_manifest.is_file() and (
            _read_json(archived_manifest)["configuration_fingerprint"]
            != summary["configuration_fingerprint"]
        ):
            raise RuntimeError("Archive root contains a different dataset fingerprint")
        _rsync_copy(global_source, global_destination)
        _rsync_verify(global_source, global_destination)
        if destination.exists():
            if destination.is_symlink():
                raise RuntimeError(f"Archive target is a symlink: {destination}")
            _rsync_verify(source, destination)
        else:
            _rsync_copy(source, staging)
            _rsync_verify(source, staging)
            os.replace(staging, destination)
        files = _tree_manifest(destination)
        record = {
            **summary,
            "source_root": str(dataset_root),
            "archive_root": str(archive_root),
            "relative_shard": f"shards/{scene}",
            "files_sha256": files,
        }
        _write_json(archive_root / "records" / f"{scene}.json", record)
        if evict:
            _rsync_verify(source, destination)
            if _tree_manifest(destination) != files:
                raise RuntimeError("Archive changed during verification")
            _write_json(dataset_root / "global" / "archived_shards" / f"{scene}.json", record)
            evicting = source.with_name(f".{scene}.evicting")
            if evicting.exists():
                raise RuntimeError(f"Previous eviction still needs inspection: {evicting}")
            os.replace(source, evicting)
            shutil.rmtree(evicting)
        return {key: value for key, value in record.items() if key != "files_sha256"} | {
            "file_count": len(files), "evicted": evict,
        }


def restore_scene(dataset_root: Path, archive_root: Path, scene: str) -> dict:
    dataset_root = dataset_root.resolve()
    archive_root = archive_root.resolve()
    if not SCENE_ID.fullmatch(scene):
        raise ValueError(f"Invalid scene ID: {scene!r}")
    with _lock(dataset_root / ".production.lock"), _lock(archive_root / ".archive.lock"):
        if not (dataset_root / "global" / "production_manifest.json").is_file():
            _rsync_copy(archive_root / "global", dataset_root / "global")
            _rsync_verify(archive_root / "global", dataset_root / "global")
        record = _read_json(archive_root / "records" / f"{scene}.json")
        manifest = _read_json(dataset_root / "global" / "production_manifest.json")
        if record["scene_id"] != scene or record["configuration_fingerprint"] != manifest["configuration_fingerprint"]:
            raise RuntimeError("Archive record and dataset manifest differ")
        source = archive_root / "shards" / scene
        if _tree_manifest(source) != record["files_sha256"]:
            raise RuntimeError("Archive checksum verification failed")
        destination = dataset_root / "shards" / scene
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(f"Restore target already exists: {destination}")
        staging = destination.with_name(f".{scene}.restore-incomplete")
        _rsync_copy(source, staging)
        _rsync_verify(source, staging)
        os.replace(staging, destination)
        _write_json(dataset_root / "global" / "archived_shards" / f"{scene}.json", {
            **record, "restored": True,
        })
        return {"scene_id": scene, "restored": True, "file_count": len(record["files_sha256"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("archive", "restore"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--evict-after-verify", action="store_true")
    parser.add_argument("--confirm-scene")
    arguments = parser.parse_args()
    if arguments.action == "archive":
        result = archive_scene(
            arguments.dataset_root, arguments.archive_root, arguments.scene,
            evict=arguments.evict_after_verify,
            confirm_scene=arguments.confirm_scene,
        )
    else:
        if arguments.evict_after_verify:
            parser.error("--evict-after-verify applies only to archive")
        result = restore_scene(arguments.dataset_root, arguments.archive_root, arguments.scene)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
