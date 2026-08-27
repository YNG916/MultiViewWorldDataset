import json

import numpy as np
import pytest

from multi_view_world_dataset.sampling.trajectories import generate_smooth_trajectory
from multi_view_world_dataset.storage.writer import DatasetWriter
from multi_view_world_dataset.utils.serialization import dump_json


def test_episode_transaction_is_atomic_and_resume_safe(tmp_path):
    writer = DatasetWriter(tmp_path / "dataset")
    writer.initialize({"schema_version": "1.0.0"})
    camera = np.eye(4)
    trajectory = generate_smooth_trajectory("robot_00", (0, 0, 0), (1, 0, 0), 0, camera, frames=3, fps=10)
    with writer.begin_episode("scene", "config", "episode") as transaction:
        transaction.write_json("meta.json", {"accepted": True})
        transaction.write_trajectories((trajectory,))
        target = transaction.finalize()
    assert target.is_dir()
    assert json.loads((target / "meta.json").read_text())["accepted"]
    assert (target / "trajectories.npz").is_file()
    with writer.begin_episode("scene", "config", "episode") as duplicate:
        duplicate.write_json("meta.json", {})
        with pytest.raises(FileExistsError):
            duplicate.finalize()
        duplicate.abort()


def test_failed_transaction_is_removed(tmp_path):
    writer = DatasetWriter(tmp_path / "dataset")
    with pytest.raises(RuntimeError):
        with writer.begin_episode("scene", "config", "bad") as transaction:
            staging = transaction.staging
            transaction.write_json("meta.json", {})
            raise RuntimeError("reject")
    assert not staging.exists()


def test_unfinalized_success_context_is_cleaned(tmp_path):
    writer = DatasetWriter(tmp_path / "dataset")
    with writer.begin_episode("scene", "config", "unfinished") as transaction:
        staging = transaction.staging
        transaction.write_json("meta.json", {"accepted": False})
    assert not staging.exists()


def test_nonfinite_values_serialize_as_json_null(tmp_path):
    path = tmp_path / "strict.json"
    dump_json(path, {"limits": np.asarray([-np.inf, np.inf])})
    assert json.loads(path.read_text()) == {"limits": [None, None]}

def test_dense_storage_falls_back_without_zarr(tmp_path):
    writer = DatasetWriter(tmp_path / "dataset")
    with writer.begin_episode("scene", "config", "dense") as transaction:
        reference = transaction.write_dense_group("robot_views/before/robot_00", {"rgb": np.zeros((2, 3, 4, 3))})
        transaction.write_json("meta.json", {"dense_ref": reference})
        target = transaction.finalize()
    assert (target / reference).exists()


def test_resume_indexes_and_reject_log(tmp_path):
    writer = DatasetWriter(tmp_path / "dataset")
    dump_json(writer.root / "configurations" / "scene" / "config_000" / "config_meta.json", {"ok": True})
    dump_json(writer.root / "episodes" / "scene" / "config_000" / "episode_000" / "meta.json", {"ok": True})
    writer.record_reject("configuration:scene", "collision", {"attempt": 3})
    writer.record_reject("episode:scene/config_000", "overlap", {"attempt": 4})
    assert writer.completed_configuration_ids("scene") == ("config_000",)
    assert writer.completed_episode_ids("scene", "config_000") == ("episode_000",)
    records = [json.loads(line) for line in (writer.root / "rejects.jsonl").read_text().splitlines()]
    assert [record["reason"] for record in records] == ["collision", "overlap"]
