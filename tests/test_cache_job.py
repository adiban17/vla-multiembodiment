"""Phase 2 checks: the cache_features jobs find their input data without hardcoded folder names."""

import importlib.util
from pathlib import Path

import pytest

from conftest import _write_robot
from vla.data.shards import write_meta

REPO = Path(__file__).resolve().parents[1]


def _kaggle_job():
    spec = importlib.util.spec_from_file_location("kaggle_job", REPO / "scripts" / "kaggle_job.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_jobs_are_registered():
    jobs = _kaggle_job().JOBS
    assert "cache_features" in jobs and "cache_features_smoke" in jobs


def test_find_and_pick_datasets(tmp_path):
    kj = _kaggle_job()
    # Kaggle-style names with commit and timestamp, nested at different depths
    _write_robot(tmp_path / "data-pusht" / "outputs" / "data_pusht-0be9946-20261125-112708", "pusht", [3, 2], 96, True)
    _write_robot(tmp_path / "fetch-smoke" / "data_fetch_smoke-0be9946-20261125-120000", "fetch", [2], 128, True)
    _write_robot(tmp_path / "fetch" / "outputs" / "data_fetch-0be9946-20261125-130000", "fetch", [3, 3, 3], 128, True)
    _write_robot(tmp_path / "mac" / "fetch_mac_render", "fetch", [3, 3, 3, 3], 128, False)
    features = tmp_path / "old-cache" / "fetch"  # a feature cache's meta.json must be ignored
    features.mkdir(parents=True)
    write_meta(features, {"kind": "features", "robot": "fetch"})

    found = kj.find_datasets(tmp_path)
    assert len(found) == 4

    picked = kj.pick_datasets(found, require_training_images=True)
    assert picked["pusht"].name.startswith("data_pusht-")
    assert picked["fetch"].name.startswith("data_fetch-")  # most episodes among training-ok sets

    picked_any = kj.pick_datasets(found, require_training_images=False)
    assert picked_any["fetch"].name == "fetch_mac_render"  # 4 episodes, debug-only allowed off Kaggle


def test_missing_robot_is_a_clear_error(tmp_path):
    kj = _kaggle_job()
    _write_robot(tmp_path / "p", "pusht", [3], 96, True)
    with pytest.raises(RuntimeError, match="fetch"):
        kj.pick_datasets(kj.find_datasets(tmp_path), require_training_images=True)
