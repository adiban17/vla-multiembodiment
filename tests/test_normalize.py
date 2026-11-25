"""Phase 2 checks: per-robot min-max normalization to [-1, 1] and stats.json."""

import numpy as np
import pytest
import torch

from vla.data.normalize import Normalizer, compute_robot_stats, load_stats, save_stats


def _fetch_like(rng, n=200):
    """FetchPush-shaped data with the constant dims seen in Phase 1."""
    state = np.zeros((n, 8), dtype=np.float32)
    state[:, 0:3] = rng.uniform([1.1, 0.5, 0.41], [1.5, 1.0, 0.50], size=(n, 3))  # hand xyz
    state[:, 3] = 0.0                                                         # finger_r: exactly constant
    state[:, 4] = rng.uniform(0.0, 0.0011, size=n)                            # finger_l: 1 mm jitter
    state[:, 5:7] = rng.uniform([1.2, 0.6], [1.5, 0.9], size=(n, 2))          # goal xy
    state[:, 7] = 0.4249                                                      # goal_z: constant
    action = rng.uniform(-1, 1, size=(n, 4)).astype(np.float32)
    action[:, 3] = 0.0                                                        # gripper: constant
    return state, action


def _pusht_like(rng, n=200):
    state = rng.uniform(40, 490, size=(n, 2)).astype(np.float32)
    action = rng.uniform(44, 511, size=(n, 2)).astype(np.float32)
    return state, action


@pytest.fixture
def normalizer():
    rng = np.random.default_rng(0)
    fs, fa = _fetch_like(rng)
    ps, pa = _pusht_like(rng)
    return Normalizer({
        "fetch": compute_robot_stats(fs, fa, action_range=(-1.0, 1.0)),
        "pusht": compute_robot_stats(ps, pa),
    })


@pytest.mark.parametrize("robot,make", [("pusht", _pusht_like), ("fetch", _fetch_like)])
def test_round_trip_and_range(normalizer, robot, make):
    state, action = make(np.random.default_rng(1))
    for key, x in (("state", state), ("action", action)):
        stats = normalizer.stats[robot][key]
        y = normalizer.normalize(x, robot, key)
        assert y.dtype == np.float32 and np.isfinite(y).all()
        span = np.array(stats["max"]) - np.array(stats["min"])
        back = normalizer.unnormalize(y, robot, key)
        real = ~np.array(stats["constant"])
        np.testing.assert_allclose(back[:, real], x[:, real], atol=1e-5 * span[real].max(), rtol=0)


def test_train_data_lands_in_minus_one_to_one(normalizer):
    rng = np.random.default_rng(0)  # the same data the stats were computed from
    fs, fa = _fetch_like(rng)
    ps, pa = _pusht_like(rng)
    for robot, s, a in (("fetch", fs, fa), ("pusht", ps, pa)):
        for key, x in (("state", s), ("action", a)):
            y = normalizer.normalize(x, robot, key)
            assert y.min() >= -1 - 1e-6 and y.max() <= 1 + 1e-6, (robot, key)


def test_constant_dims_map_to_zero_and_back_to_their_value(normalizer):
    state, action = _fetch_like(np.random.default_rng(2))
    const = np.array(normalizer.stats["fetch"]["state"]["constant"])
    # finger_r (exactly 0), finger_l (1 mm jitter) and goal_z are constant; nothing else is
    assert const.tolist() == [False, False, False, True, True, False, False, True]

    y = normalizer.normalize(state, "fetch", "state")
    assert np.isfinite(y).all()
    assert (y[:, const] == 0).all()

    back = normalizer.unnormalize(y, "fetch", "state")
    assert (back[:, 3] == np.float32(0.0)).all()        # exact constant
    assert (back[:, 7] == np.float32(0.4249)).all()     # exact constant
    assert np.abs(back[:, 4] - state[:, 4]).max() <= 0.0011 / 2 + 1e-7  # near-constant -> midpoint


def test_fetch_actions_are_left_as_they_are(normalizer):
    _, action = _fetch_like(np.random.default_rng(3))
    y = normalizer.normalize(action, "fetch", "action")
    np.testing.assert_allclose(y, action, atol=1e-6)
    assert (y[:, 3] == 0).all()
    assert (normalizer.unnormalize(y, "fetch", "action")[:, 3] == 0).all()


def test_torch_inputs_work_like_numpy(normalizer):
    _, action = _pusht_like(np.random.default_rng(4))
    t = torch.from_numpy(action)
    y = normalizer.normalize(t, "pusht", "action")
    assert isinstance(y, torch.Tensor) and y.dtype == torch.float32
    np.testing.assert_allclose(y.numpy(), normalizer.normalize(action, "pusht", "action"), atol=1e-6)
    np.testing.assert_allclose(normalizer.unnormalize(y, "pusht", "action").numpy(), action, atol=1e-3)


def test_stats_json_round_trip(normalizer, tmp_path):
    path = tmp_path / "stats.json"
    save_stats(path, normalizer.stats)
    loaded = Normalizer(load_stats(path))
    state, _ = _fetch_like(np.random.default_rng(5))
    np.testing.assert_array_equal(loaded.normalize(state, "fetch", "state"), normalizer.normalize(state, "fetch", "state"))
    assert Normalizer.load(path).stats == loaded.stats
