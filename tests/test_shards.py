"""Phase 1 checks: the .npz shard format round-trips exactly, and the training-image guard."""

import numpy as np
import pytest

from vla.data.shards import ShardWriter, assert_training_images_ok, read_meta, read_shards, summarize, write_meta


def _episode(rng, length, with_images=True, state_dim=8, action_dim=4):
    return dict(
        state=rng.normal(size=(length, state_dim)).astype(np.float32),
        action=rng.uniform(-1, 1, size=(length, action_dim)).astype(np.float32),
        images=rng.integers(0, 256, size=(length, 16, 16, 3), dtype=np.uint8) if with_images else None,
    )


def test_round_trip_across_several_shards(tmp_path):
    rng = np.random.default_rng(0)
    episodes = [_episode(rng, n) for n in (5, 3, 7)]
    seeds = [11, 22, 33]

    with ShardWriter(tmp_path, shard_size=2) as writer:  # 3 episodes, 2 per shard -> 2 shards
        for ep, seed in zip(episodes, seeds):
            writer.add_episode(seed=seed, **ep)
    assert len(list(tmp_path.glob("shard_*.npz"))) == 2

    data = read_shards(tmp_path)
    for key in ["state", "action", "images"]:
        expected = np.concatenate([ep[key] for ep in episodes])
        assert data[key].dtype == expected.dtype
        np.testing.assert_array_equal(data[key], expected)

    np.testing.assert_array_equal(data["episode_seed"], seeds)
    np.testing.assert_array_equal(data["episode_length"], [5, 3, 7])
    np.testing.assert_array_equal(data["episode_starts"], [0, 5, 8])
    np.testing.assert_array_equal(data["episode_ends"], [5, 8, 15])  # exclusive end
    np.testing.assert_array_equal(data["episode_index"], [0] * 5 + [1] * 3 + [2] * 7)
    np.testing.assert_array_equal(data["frame_index"][5:8], [0, 1, 2])


def test_round_trip_without_images(tmp_path):
    rng = np.random.default_rng(1)
    with ShardWriter(tmp_path) as writer:
        writer.add_episode(seed=-1, **_episode(rng, 4, with_images=False, state_dim=2, action_dim=2))
    data = read_shards(tmp_path)
    assert "images" not in data
    assert data["state"].shape == (4, 2)


def test_mismatched_lengths_are_rejected(tmp_path):
    rng = np.random.default_rng(2)
    ep = _episode(rng, 4)
    ep["action"] = ep["action"][:3]
    with ShardWriter(tmp_path) as writer, pytest.raises(ValueError):
        writer.add_episode(seed=0, **ep)


def test_summarize_and_meta(tmp_path):
    rng = np.random.default_rng(3)
    with ShardWriter(tmp_path) as writer:
        writer.add_episode(seed=0, **_episode(rng, 6))
    stats = summarize(read_shards(tmp_path))
    assert stats["n_episodes"] == 1 and stats["n_frames"] == 6
    assert len(stats["action_min"]) == 4 and len(stats["state_max"]) == 8

    write_meta(tmp_path, {"robot": "fetch", **stats})
    assert read_meta(tmp_path)["robot"] == "fetch"


def test_training_guard_rejects_debug_images():
    # Mac-rendered (debug) images must never become training data.
    with pytest.raises(ValueError):
        assert_training_images_ok({"has_images": True, "images_for_training": False, "image_source": "mac debug"})
    with pytest.raises(ValueError):
        assert_training_images_ok({"has_images": False})
    assert_training_images_ok({"has_images": True, "images_for_training": True, "image_source": "egl"})
