"""Shared helpers for the Phase 2 tests: tiny synthetic Phase 1 datasets and fake encoders.

Synthetic data makes leaks easy to see: the image of global frame i is filled with the value i,
and each action encodes its episode and step, so a chunk that crosses into the next episode,
or a wrong frame, shows up as a wrong number.
"""

import numpy as np
import pytest

from vla.data.shards import ShardWriter, write_meta

PUSHT_LENGTHS = [20, 5, 12, 3]
FETCH_LENGTHS = [50, 18, 7]


def _write_robot(out_dir, robot, lengths, image_size, images_for_training, first_frame=0, shard_size=2):
    rng = np.random.default_rng(len(lengths))
    frame = first_frame
    with ShardWriter(out_dir, shard_size=shard_size) as writer:
        for ep, n in enumerate(lengths):
            t = np.arange(n, dtype=np.float32)
            images = np.zeros((n, image_size, image_size, 3), dtype=np.uint8)
            images[:] = (np.arange(frame, frame + n) % 256)[:, None, None, None]
            if robot == "pusht":
                state = np.stack([100 + 10 * ep + t, 300 - t], axis=1)
                action = np.stack([1000.0 * ep + t, 50 + t], axis=1)
            else:
                state = np.zeros((n, 8), dtype=np.float32)
                state[:, 0:3] = rng.uniform([1.1, 0.5, 0.41], [1.5, 1.0, 0.50], size=(n, 3))
                state[:, 4] = rng.uniform(0, 0.001, size=n)
                state[:, 5:7] = [1.2 + 0.1 * ep, 0.8]
                state[:, 7] = 0.4249
                action = np.stack([np.full(n, ep / 10), t / 100, np.full(n, 0.5), np.zeros(n)], axis=1)
            writer.add_episode(state=state, action=action, seed=ep, images=images)
            frame += n
    names = {"pusht": (["agent_x", "agent_y"], ["target_x", "target_y"]),
             "fetch": (["hand_x", "hand_y", "hand_z", "finger_r", "finger_l", "goal_x", "goal_y", "goal_z"],
                       ["dx", "dy", "dz", "gripper"])}[robot]
    write_meta(out_dir, {
        "robot": robot, "state_names": names[0], "action_names": names[1], "has_images": True,
        "image_source": "synthetic test data", "images_for_training": images_for_training,
        "image_shape": [image_size, image_size, 3], "n_episodes": len(lengths), "n_frames": int(sum(lengths)),
    })
    return out_dir


@pytest.fixture
def synthetic_data(tmp_path):
    """{"pusht": dir, "fetch": dir} of Phase 1-format shards. FetchPush is marked debug-only."""
    return {
        "pusht": _write_robot(tmp_path / "data_pusht", "pusht", PUSHT_LENGTHS, 96, True),
        "fetch": _write_robot(tmp_path / "data_fetch", "fetch", FETCH_LENGTHS, 128, False),
    }


def fake_encode(frames):
    """Stand-in for DINOv2: every number of frame i's 65x384 features equals its pixel value."""
    frames = np.asarray(frames)
    value = frames.reshape(len(frames), -1)[:, 0].astype(np.float32)
    return np.broadcast_to(value[:, None, None], (len(frames), 65, 384)).copy()


def fake_language(sentences):
    """Stand-in for MiniLM: sentence k gets the vector filled with k."""
    return np.broadcast_to(np.arange(len(sentences), dtype=np.float32)[:, None], (len(sentences), 384)).copy()
