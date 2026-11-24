"""Phase 1 checks: converting LeRobot-style frames to our arrays (no lerobot needed),
and the PushT side-by-side comparison image."""

import imageio.v3 as iio
import numpy as np

from vla.data.pusht_compare import save_pusht_compare
from vla.data.pusht_loader import frames_to_episodes
from vla.data.shards import ShardWriter


def _fake_frame(ep, t, value):
    return {
        "observation.image": np.full((3, 96, 96), value, dtype=np.float32),  # CHW in [0, 1], like LeRobot
        "observation.state": np.array([ep, t], dtype=np.float32),
        "action": np.array([100 + t, 200 + t], dtype=np.float32),
        "episode_index": np.int64(ep),
        "frame_index": np.int64(t),
    }


def test_frames_to_episodes():
    frames = [_fake_frame(0, t, 1.0) for t in range(3)] + [_fake_frame(1, t, 0.5) for t in range(2)]
    episodes = frames_to_episodes(frames)

    assert [len(ep["action"]) for ep in episodes] == [3, 2]
    assert episodes[0]["images"].shape == (3, 96, 96, 3)  # HWC
    assert episodes[0]["images"].dtype == np.uint8
    assert episodes[0]["images"].max() == 255 and episodes[1]["images"].max() == 128
    assert episodes[1]["state"].dtype == np.float32
    np.testing.assert_array_equal(episodes[0]["action"][:, 0], [100, 101, 102])
    assert episodes[0]["seed"] == -1  # human demos have no env seed


def test_pusht_compare_image(tmp_path):
    shard_dir = tmp_path / "shards"
    with ShardWriter(shard_dir) as writer:
        writer.add_episode(
            state=np.array([[200.0, 300.0]], dtype=np.float32),
            action=np.zeros((1, 2), dtype=np.float32),
            images=np.full((1, 96, 96, 3), 127, dtype=np.uint8),
            seed=-1,
        )
    out_png = tmp_path / "pusht_compare.png"
    save_pusht_compare(shard_dir, out_png, scale=4)

    image = iio.imread(out_png)
    assert image.shape[0] >= 96 * 4 and image.shape[1] >= 2 * 96 * 4  # two upscaled frames side by side
