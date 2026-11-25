"""Phase 3 checks: configs, learning-rate schedule, 50/50 batches, finding the cache, and a
tiny end-to-end training run on CPU with a synthetic feature cache. Written before the code."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import fake_encode, fake_language
from vla.data.cache_features import build_cache, find_feature_caches, pick_feature_cache
from vla.data.dataset import MixedBatchSampler, split_episodes
from vla.data.shards import write_meta
from vla.train import lr_factor, train
from vla.utils.config import load_config

REPO = Path(__file__).resolve().parents[1]


# --- config -------------------------------------------------------------------


def test_config_groups_and_overrides():
    cfg = load_config()
    assert (cfg.cond, cfg.adapt, cfg.size) == ("film", "frozen", "full")
    assert cfg.model.width == 384 and cfg.model.depth == 8 and cfg.model.heads == 6 and cfg.model.ffn == 1536
    assert cfg.data.batch_size == 256 and cfg.optim.lr == pytest.approx(3e-4)
    # 4000/300, not the spec's 20k/500: measured 2.0 steps/s on a T4 (CLAUDE.md)
    assert cfg.optim.warmup_steps == 300 and cfg.train.steps == 4000 and cfg.train.val_every == 1000
    assert cfg.eval.enabled and cfg.eval.n_episodes == 50 and cfg.eval.seed_start == 100000
    assert cfg.eval.exec_steps == 8 and cfg.eval.paraphrase == 0

    cfg = load_config(["cond=token", "size=tiny", "train.steps=7"])
    assert (cfg.cond, cfg.size) == ("token", "tiny")
    assert cfg.model.width == 64 and cfg.train.steps == 7
    assert cfg.optim.warmup_steps <= 20, "tiny runs must not sit in a 500-step warmup"

    cfg = load_config(["cond=none"])
    assert cfg.cond == "none"


def test_config_errors():
    with pytest.raises(ValueError, match="film"):  # lists the valid choices
        load_config(["cond=flim"])
    with pytest.raises(Exception):  # typo in a key is an error, not a silent new key
        load_config(["train.step=5"])


# --- learning-rate schedule -------------------------------------------------------


def test_warmup_then_cosine_to_zero():
    warm, total = 500, 20000
    assert lr_factor(0, warm, total) == pytest.approx(1 / warm)
    assert lr_factor(warm - 1, warm, total) == pytest.approx(1.0)
    assert lr_factor(warm, warm, total) == pytest.approx(1.0)
    assert lr_factor((warm + total) // 2, warm, total) == pytest.approx(0.5, abs=1e-3)
    assert lr_factor(total, warm, total) == pytest.approx(0.0, abs=1e-9)
    after = [lr_factor(s, warm, total) for s in range(warm, total + 1, 100)]
    assert all(a >= b for a, b in zip(after, after[1:]))
    assert lr_factor(0, 0, 10) == pytest.approx(1.0)  # no warmup


# --- 50/50 batches ------------------------------------------------------------------


def test_mixed_batch_sampler_is_half_and_half_and_deterministic():
    sizes = [10, 7]
    sampler = MixedBatchSampler(sizes, batch_size=6, seed=3)
    it = iter(sampler)
    batches = [next(it) for _ in range(20)]
    for batch in batches:
        assert len(batch) == 6
        assert sum(i < 10 for i in batch) == 3 and sum(i >= 10 for i in batch) == 3
    # within one pass over a robot, every index appears exactly once
    pusht = [i for b in batches for i in b if i < 10][:10]
    fetch = [i for b in batches for i in b if i >= 10][:7]
    assert sorted(pusht) == list(range(10)) and sorted(fetch) == list(range(10, 17))
    it2 = iter(MixedBatchSampler(sizes, batch_size=6, seed=3))
    assert [next(it2) for _ in range(20)] == batches
    it3 = iter(MixedBatchSampler(sizes, batch_size=6, seed=4))
    assert [next(it3) for _ in range(20)] != batches


# --- finding the cache on Kaggle -------------------------------------------------------


def _fake_cache(folder: Path, frames: dict, training_ok: dict):
    folder.mkdir(parents=True)
    (folder / "stats.json").write_text("{}")
    np.save(folder / "language.npy", np.zeros((20, 384), dtype=np.float32))
    for robot, n in frames.items():
        (folder / robot).mkdir()
        write_meta(folder / robot, {"kind": "features", "robot": robot, "n_frames": n,
                                    "images_for_training": training_ok[robot]})
    return folder


def test_find_and_pick_feature_cache(tmp_path):
    ok = {"pusht": True, "fetch": True}
    full = _fake_cache(tmp_path / "vla-cache" / "outputs" / "cache_features-abc1234-20261126-010000",
                       {"pusht": 25000, "fetch": 25000}, ok)
    _fake_cache(tmp_path / "vla-cache-smoke" / "cache_features_smoke-abc1234-20261126-000000",
                {"pusht": 400, "fetch": 150}, ok)
    debug = _fake_cache(tmp_path / "mac" / "cache_debug", {"pusht": 90000, "fetch": 90000},
                        {"pusht": True, "fetch": False})
    # not caches: a Phase 1 dataset folder, and a stats.json without robot folders
    write_meta(tmp_path, {"robot": "pusht"})
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "stats.json").write_text("{}")

    found = find_feature_caches(tmp_path)
    assert len(found) == 3
    assert pick_feature_cache(found, require_training_images=True)["path"] == full
    assert pick_feature_cache(found, require_training_images=False)["path"] == debug

    only_debug = [c for c in found if c["path"] == debug]
    with pytest.raises(RuntimeError, match="debug"):
        pick_feature_cache(only_debug, require_training_images=True)
    with pytest.raises(RuntimeError, match="cache_features"):
        pick_feature_cache([], require_training_images=True)


# --- end to end on CPU ----------------------------------------------------------------


@pytest.fixture
def synthetic_cache(tmp_path, synthetic_data):
    """A real (tiny) feature cache. Its FetchPush images are marked debug-only."""
    run = tmp_path / "inputs" / "my-cache" / "cache_features-0000000-20261126-000000"
    build_cache(run, synthetic_data, fake_encode, fake_language)
    return run


@pytest.fixture
def local_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VLA_OUTPUTS_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("VLA_INPUTS_DIR", str(tmp_path / "inputs"))
    monkeypatch.setenv("WANDB_MODE", "disabled")


def _tiny_cfg(*extra):
    return load_config(["size=tiny", "train.steps=6", "train.val_every=3", "train.log_every=2",
                        "eval.enabled=false", *extra])


def test_tiny_training_run_end_to_end(synthetic_cache, local_env):
    run_dir = train(_tiny_cfg(), allow_dirty=True)
    for name in ("config.yaml", "git.txt", "metrics.json", "ckpt.pt"):
        assert (run_dir / name).is_file(), name
    assert run_dir.name.startswith("film-frozen-s0-")

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["steps"] == 6
    assert [v["step"] for v in metrics["val_curve"]] == [3, 6]
    for key in ("val_loss", "val_loss_pusht", "val_loss_fetch", "train_loss", "steps_per_sec",
                "data_ms_per_step", "compute_ms_per_step", "trainable_params"):
        assert math.isfinite(metrics[key]), key
    # val/loss is the equal-weight average of the two robots
    assert metrics["val_loss"] == pytest.approx((metrics["val_loss_pusht"] + metrics["val_loss_fetch"]) / 2)
    assert metrics["debug_data"] is True
    assert metrics["cache_dir"] == str(synthetic_cache)

    # the run used the cache's own episode split, which is split_episodes(n, 0.1, seed 0)
    stats = json.loads((synthetic_cache / "stats.json").read_text())
    for robot, n in (("pusht", 4), ("fetch", 3)):
        train_eps, val_eps = split_episodes(n, 0.1, 0)
        assert stats["split"][robot]["train"] == train_eps and stats["split"][robot]["val"] == val_eps
        assert metrics["split"][robot] == stats["split"][robot]

    ckpt = torch.load(run_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    assert ckpt["step"] == 6 and "model" in ckpt and ckpt["config"]["cond"] == "film"


def test_same_seed_gives_the_same_run(synthetic_cache, local_env):
    a = json.loads((train(_tiny_cfg(), allow_dirty=True) / "metrics.json").read_text())
    b = json.loads((train(_tiny_cfg(), allow_dirty=True) / "metrics.json").read_text())
    assert a["val_loss"] == pytest.approx(b["val_loss"], rel=1e-5)


def test_debug_only_cache_is_refused_unless_allowed(synthetic_cache, local_env):
    with pytest.raises(RuntimeError, match="debug"):
        train(_tiny_cfg("data.allow_debug_images=false"), allow_dirty=True)


def test_refuses_uncommitted_changes(synthetic_cache, local_env, monkeypatch):
    import vla.train

    monkeypatch.setattr(vla.train, "git_commit", lambda: {"commit": "f" * 40, "dirty": True})
    with pytest.raises(RuntimeError, match="allow-dirty"):
        train(_tiny_cfg())
    assert train(_tiny_cfg(), allow_dirty=True).is_dir()


def test_smoke_train_job_is_registered():
    spec = importlib.util.spec_from_file_location("kaggle_job", REPO / "scripts" / "kaggle_job.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "smoke_train" in module.JOBS
