"""Phase 1 checks: FetchPush collection (no rendering) and replaying saved demos."""

import numpy as np
import pytest

from vla.data.collect_fetch import collect
from vla.data.shards import assert_training_images_ok, read_meta, read_shards
from vla.envs import fetch


@pytest.fixture(scope="module")
def collected(tmp_path_factory):
    out = tmp_path_factory.mktemp("fetch")
    meta = collect(out, n_success=3, render=False, n_videos=0)
    return out, meta


def test_collect_writes_successful_full_episodes(collected):
    out, meta = collected
    data = read_shards(out)
    assert meta == read_meta(out)
    assert meta["n_episodes"] == 3 and meta["attempts"] >= 3
    assert 0.0 < meta["expert_success_rate"] <= 1.0
    np.testing.assert_array_equal(data["episode_length"], [50, 50, 50])
    assert data["state"].shape == (150, 8) and data["action"].shape == (150, 4)
    assert all(seed < 100000 for seed in data["episode_seed"])  # 100000+ is reserved for evaluation
    assert "images" not in data and meta["has_images"] is False
    with pytest.raises(ValueError):
        assert_training_images_ok(meta)  # no images -> not training data


def test_replaying_saved_actions_succeeds(collected):
    out, _ = collected
    data = read_shards(out)
    for ep, seed in enumerate(data["episode_seed"]):
        start, end = data["episode_starts"][ep], data["episode_ends"][ep]
        env = fetch.make_env(seed=int(seed), render=False)
        obs = env.reset()
        np.testing.assert_allclose(obs["state"], data["state"][start], atol=1e-6)
        for action in data["action"][start:end]:
            obs, success, done, _ = env.step(action)
        env.close()
        assert success, f"replay of episode {ep} (seed {seed}) failed"
