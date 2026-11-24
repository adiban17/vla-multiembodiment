"""Phase 1 checks: the two environment wrappers."""

import os

import numpy as np
import pytest

from vla.envs import fetch, pusht
from vla.utils.runtime import is_kaggle

# Rendering FetchPush images needs EGL, which only Kaggle has. Set VLA_TEST_RENDER=1 to try anyway.
needs_render = pytest.mark.skipif(
    not (is_kaggle() or os.environ.get("VLA_TEST_RENDER") == "1"),
    reason="Kaggle-only: FetchPush rendering (EGL)",
)


# --- PushT (images come from pygame, so they work on the Mac) -------------------

def test_pusht_observation_and_step():
    env = pusht.make_env(seed=0)
    obs = env.reset()
    assert obs["image"].shape == (96, 96, 3) and obs["image"].dtype == np.uint8
    assert obs["image"].std() > 0  # a real picture, not a blank frame
    assert obs["state"].shape == (2,) and obs["state"].dtype == np.float32
    assert env.max_steps == 300

    obs, success, done, info = env.step(np.array([256.0, 256.0], dtype=np.float32))
    assert 0.0 <= info["coverage"] <= 1.0
    assert success == (info["coverage"] >= 0.95)
    env.close()


def test_pusht_same_seed_same_start():
    a, b = pusht.make_env(seed=7), pusht.make_env(seed=7)
    np.testing.assert_array_equal(a.reset()["image"], b.reset()["image"])
    a.close(), b.close()


# --- FetchPush ----------------------------------------------------------------

def test_fetch_without_render():
    env = fetch.make_env(seed=0, render=False)
    obs = env.reset()
    assert obs["image"] is None
    assert obs["state"].shape == (8,) and obs["state"].dtype == np.float32
    np.testing.assert_allclose(obs["state"][:3], obs["raw"]["observation"][:3])  # hand xyz
    np.testing.assert_allclose(obs["state"][5:], obs["raw"]["desired_goal"])  # goal xyz
    assert env.max_steps == 50

    obs, success, done, info = env.step(np.zeros(4, dtype=np.float32))
    assert isinstance(success, bool)
    env.close()


def test_fetch_same_seed_same_start():
    a, b = fetch.make_env(seed=3), fetch.make_env(seed=3)
    np.testing.assert_array_equal(a.reset()["state"], b.reset()["state"])
    c = fetch.make_env(seed=4)
    assert not np.array_equal(a.reset()["state"], c.reset()["state"])
    for env in (a, b, c):
        env.close()


@needs_render
def test_fetch_render_128():
    env = fetch.make_env(seed=0, render=True)
    image = env.reset()["image"]
    assert image.shape == (128, 128, 3) and image.dtype == np.uint8
    assert image.std() > 0
    env.close()
