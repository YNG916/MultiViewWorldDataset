from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from multi_view_world_dataset.schema.records import DynamicConfiguration, ObjectState, Trajectory, WorldEpisode
from multi_view_world_dataset.utils.serialization import dump_json, to_jsonable


def _write_catalog(path: Path, catalog: tuple[ObjectState, ...]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Catalog persistence requires the 'storage' extra with pyarrow") from error
    rows = [to_jsonable(obj) for obj in catalog]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


@dataclass
class EpisodeTransaction:
    target: Path
    staging: Path
    _finalized: bool = False

    def write_json(self, relative_path: str, value: Any) -> None:
        dump_json(self.staging / relative_path, value)

    def write_trajectories(self, trajectories: tuple[Trajectory, ...]) -> None:
        arrays: dict[str, np.ndarray] = {}
        metadata = {}
        for trajectory in trajectories:
            arrays[f"{trajectory.robot_id}_base_to_world"] = trajectory.base_to_world
            arrays[f"{trajectory.robot_id}_camera_to_world"] = trajectory.camera_to_world
            metadata[trajectory.robot_id] = {"fps": trajectory.fps, "frames": trajectory.frames}
        np.savez_compressed(self.staging / "trajectories.npz", **arrays)
        self.write_json("trajectories_meta.json", metadata)

    def write_dense_group(self, relative_path: str, arrays: dict[str, np.ndarray]) -> str:
        try:
            import zarr
        except ImportError:
            fallback = self.staging / f"{relative_path}.npz"
            fallback.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(fallback, **{name: np.asarray(array) for name, array in arrays.items()})
            return f"{relative_path}.npz"
        group = zarr.open_group(str(self.staging / relative_path), mode="w")
        for name, array in arrays.items():
            group.create_array(name, data=np.asarray(array), overwrite=False)
        return relative_path

    def finalize(self) -> Path:
        if self._finalized:
            return self.target
        if self.target.exists():
            raise FileExistsError(f"Refusing to overwrite finalized episode: {self.target}")
        self.target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.staging, self.target)
        self._finalized = True
        return self.target

    def abort(self) -> None:
        if not self._finalized and self.staging.exists():
            shutil.rmtree(self.staging)

    def __enter__(self) -> "EpisodeTransaction":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is not None or not self._finalized:
            self.abort()


class DatasetWriter:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def initialize(self, dataset_meta: dict[str, Any], taxonomy: dict[str, Any] | None = None) -> None:
        dump_json(self.root / "dataset_meta.json", dataset_meta)
        dump_json(self.root / "taxonomy.json", taxonomy or {})

    def write_scene(self, scene_id: str, scene_meta: Any, catalog: tuple[ObjectState, ...]) -> Path:
        target = self.root / "scenes" / scene_id
        dump_json(target / "scene_meta.json", scene_meta)
        _write_catalog(target / "base_object_catalog.parquet", catalog)
        return target

    def write_configuration(
        self,
        configuration: DynamicConfiguration,
        catalog: tuple[ObjectState, ...],
        *,
        snapshot: np.ndarray | None = None,
        environment_bev: dict[str, np.ndarray] | None = None,
    ) -> Path:
        target = self.root / "configurations" / configuration.scene_id / configuration.configuration_id
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite configuration: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{configuration.configuration_id}.", dir=target.parent))
        try:
            dump_json(staging / "config_meta.json", configuration)
            _write_catalog(staging / "object_catalog.parquet", catalog)
            if snapshot is not None:
                np.save(staging / "simulator_state.npy", np.asarray(snapshot), allow_pickle=False)
            if environment_bev is not None:
                bev_path = staging / "bev" / "environment_base.npz"
                bev_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    bev_path, **{name: np.asarray(value) for name, value in environment_bev.items()}
                )
            os.replace(staging, target)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return target

    def begin_episode(self, scene_id: str, configuration_id: str, episode_id: str) -> EpisodeTransaction:
        target = self.root / "episodes" / scene_id / configuration_id / episode_id
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{episode_id}.", dir=target.parent))
        return EpisodeTransaction(target, staging)

    def completed_configuration_ids(self, scene_id: str) -> tuple[str, ...]:
        root = self.root / "configurations" / scene_id
        if not root.is_dir():
            return ()
        return tuple(
            sorted(
                path.name
                for path in root.iterdir()
                if path.is_dir() and not path.name.startswith(".") and (path / "config_meta.json").is_file()
            )
        )

    def completed_episode_ids(self, scene_id: str, configuration_id: str) -> tuple[str, ...]:
        root = self.root / "episodes" / scene_id / configuration_id
        if not root.is_dir():
            return ()
        return tuple(
            sorted(
                path.name
                for path in root.iterdir()
                if path.is_dir() and not path.name.startswith(".") and (path / "meta.json").is_file()
            )
        )

    def record_reject(self, scope: str, reason: str, details: dict[str, Any] | None = None) -> None:
        record = json.dumps(
            to_jsonable({"scope": scope, "reason": reason, "details": details or {}}),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        path = self.root / "rejects.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(descriptor, (record + "\n").encode("utf-8"))
        finally:
            os.close(descriptor)

    @staticmethod
    def write_episode_metadata(transaction: EpisodeTransaction, episode: WorldEpisode) -> None:
        transaction.write_json("meta.json", {
            "episode_id": episode.episode_id,
            "scene_id": episode.scene_id,
            "configuration_id": episode.configuration_id,
            "seed": episode.seed,
            "simulator_versions": episode.simulator_versions,
            "generator_git_commit": episode.generator_git_commit,
        })
        transaction.write_json("events.json", [episode.intervention])
        transaction.write_json("state_before.json", episode.state_before)
        transaction.write_json("state_after.json", episode.state_after)
        transaction.write_json("qa.json", episode.qa)
        transaction.write_trajectories(episode.trajectories)

