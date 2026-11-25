"""Phase 2 checks: the unified windowed sample (spec table "One sample, both robots").

Uses tiny synthetic data and fake encoders, so it needs no model downloads. The last test also
runs on the small Mac datasets in runs/debug/ if they exist (shape checks only, never training).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import FETCH_LENGTHS, PUSHT_LENGTHS, fake_encode, fake_language
from vla.data.cache_features import build_cache
from vla.data.dataset import CHUNK, WindowDataset, split_episodes
from vla.data.instructions import instruction_id
from vla.data.shards import read_shards
from vla.models.encoders import preprocess_image

REPO = Path(__file__).resolve().parents[1]
LENGTHS = {"pusht": PUSHT_LENGTHS, "fetch": FETCH_LENGTHS}


@pytest.fixture
def cache(tmp_path, synthetic_data):
    run = tmp_path / "cache"
    build_cache(run, synthetic_data, fake_encode, fake_language)
    return run


def _dataset(cache, robot, **kw):
    kw.setdefault("split", "all")
    kw.setdefault("require_training_images", False)
    return WindowDataset(cache / robot, **kw)


@pytest.mark.parametrize("robot", ["pusht", "fetch"])
def test_shapes_match_the_spec_table(cache, robot):
    for mode in ("cached", "raw"):
        ds = _dataset(cache, robot, mode=mode)
        assert len(ds) == sum(LENGTHS[robot])
        s = ds[7]
        if mode == "cached":
            assert s["vision"].shape == (2, 65, 384) and s["vision"].dtype == torch.float32
            assert "images" not in s
        else:
            assert s["images"].shape == (2, 3, 224, 224) and s["images"].dtype == torch.float32
            assert "vision" not in s
        assert s["lang"].shape == (384,)
        assert isinstance(s["instruction"], str)
        assert s["state"].shape == (8,) and s["state"].dtype == torch.float32
        assert s["actions"].shape == (16, 4) and s["actions"].dtype == torch.float32
        assert s["action_dim_mask"].shape == (4,) and s["action_dim_mask"].dtype == torch.bool
        assert s["action_is_pad"].shape == (16,) and s["action_is_pad"].dtype == torch.bool
        assert s["embodiment_id"].shape == () and s["embodiment_id"].dtype == torch.int64


def test_robot_specific_padding_masks_and_ids(cache):
    p, f = _dataset(cache, "pusht")[3], _dataset(cache, "fetch")[3]
    assert p["embodiment_id"].item() == 0 and f["embodiment_id"].item() == 1
    assert p["action_dim_mask"].tolist() == [True, True, False, False]
    assert f["action_dim_mask"].tolist() == [True, True, True, True]
    assert (p["state"][2:] == 0).all()          # 2 real + 6 zeros
    assert (p["actions"][:, 2:] == 0).all()     # 2 real + 2 padded


@pytest.mark.parametrize("robot", ["pusht", "fetch"])
def test_chunks_never_cross_into_the_next_episode(cache, synthetic_data, robot):
    ds = _dataset(cache, robot)
    raw = read_shards(synthetic_data[robot], with_images=False)
    dims = len(raw["action"][0])
    for i in range(len(ds)):
        s = ds[i]
        t, ep = s["frame"].item(), s["episode"].item()
        end = raw["episode_ends"][ep]
        k = min(CHUNK, end - t)
        assert s["action_is_pad"].tolist() == [False] * k + [True] * (CHUNK - k), (robot, t)
        assert (s["actions"][k:] == 0).all()
        real = ds.normalizer.unnormalize(s["actions"][:k, :dims].numpy(), robot, "action")
        np.testing.assert_allclose(real, raw["action"][t:t + k], atol=1e-3)


def test_last_frame_of_an_episode_has_one_real_step(cache):
    ds = _dataset(cache, "pusht")
    last_of_first_episode = PUSHT_LENGTHS[0] - 1
    s = ds[last_of_first_episode]
    assert s["episode"].item() == 0
    assert s["action_is_pad"].sum().item() == 15 and not s["action_is_pad"][0]


@pytest.mark.parametrize("robot", ["pusht", "fetch"])
def test_frame_t_minus_1_repeats_frame_t_at_episode_start(cache, robot):
    ds = _dataset(cache, robot, mode="cached")
    starts = np.cumsum([0] + LENGTHS[robot][:-1])
    for i in range(len(ds)):
        s = ds[i]
        t = s["frame"].item()
        v = s["vision"][:, 0, 0]  # fake features: value = frame index
        expected_prev = t if t in starts else t - 1
        assert v.tolist() == [float(expected_prev), float(t)], (robot, t)


@pytest.mark.parametrize("robot", ["pusht", "fetch"])
def test_raw_mode_returns_the_same_image_as_the_shard(cache, synthetic_data, robot):
    images = np.load(cache / robot / "images.npy", mmap_mode="r")
    shard_images = read_shards(synthetic_data[robot])["images"]
    assert images.dtype == np.uint8 and images.shape == shard_images.shape
    np.testing.assert_array_equal(images, shard_images)

    ds = _dataset(cache, robot, mode="raw")
    for i in (0, 4, len(ds) - 1):
        s = ds[i]
        t = s["frame"].item()
        torch.testing.assert_close(s["images"][1], preprocess_image(shard_images[t])[0], rtol=0, atol=0)


def test_train_and_val_split_by_episode(cache):
    for robot in ("pusht", "fetch"):
        train, val = _dataset(cache, robot, split="train"), _dataset(cache, robot, split="val")
        tr = {train[i]["episode"].item() for i in range(len(train))}
        va = {val[i]["episode"].item() for i in range(len(val))}
        assert tr and va and not tr & va
        assert tr | va == set(range(len(LENGTHS[robot])))
    assert split_episodes(20, 0.1, seed=0) == split_episodes(20, 0.1, seed=0)
    train, val = split_episodes(20, 0.1, seed=0)
    assert len(val) == 2 and len(train) == 18


def test_debug_images_are_refused_for_training(cache):
    WindowDataset(cache / "pusht", split="train")  # synthetic pusht is marked training-ok
    with pytest.raises(ValueError, match="debug-only"):
        WindowDataset(cache / "fetch", split="train")  # synthetic fetch is marked debug-only


def test_instruction_choice(cache):
    ds = _dataset(cache, "fetch", instructions="train")
    train_ids = {instruction_id("fetch", k) for k in range(8)}
    seen = {ds[i]["instruction_id"].item() for i in range(len(ds))}
    assert seen <= train_ids and len(seen) > 1
    held = _dataset(cache, "fetch", instructions="heldout")
    assert {held[i]["instruction_id"].item() for i in range(10)} <= {instruction_id("fetch", 8), instruction_id("fetch", 9)}
    fixed = _dataset(cache, "fetch", instructions=0)[5]
    assert fixed["instruction_id"].item() == instruction_id("fetch", 0)
    assert (fixed["lang"] == float(instruction_id("fetch", 0))).all()  # fake language vector k = k


def test_max_episodes_caches_only_the_first_episodes(tmp_path, synthetic_data):
    run = tmp_path / "small"
    build_cache(run, synthetic_data, fake_encode, fake_language, max_episodes=2)
    for robot in ("pusht", "fetch"):
        n = sum(LENGTHS[robot][:2])
        assert np.load(run / robot / "vision.npy", mmap_mode="r").shape == (n, 65, 384)
        assert len(_dataset(run, robot)) == n


@pytest.mark.parametrize("name,robot", [("pusht", "pusht"), ("fetch_mac_render", "fetch")])
def test_shapes_on_the_mac_debug_data(tmp_path, name, robot):
    data = REPO / "runs" / "debug" / name
    if not (data / "meta.json").exists():
        pytest.skip(f"{data} not present (gitignored Mac debug data)")
    run = tmp_path / "cache"
    build_cache(run, {robot: data}, fake_encode, fake_language, max_episodes=2)
    ds = WindowDataset(run / robot, split="all", mode="raw", require_training_images=False)
    s = ds[len(ds) - 1]
    assert s["images"].shape == (2, 3, 224, 224) and s["actions"].shape == (16, 4) and s["state"].shape == (8,)
    assert s["action_is_pad"].sum().item() == 15
