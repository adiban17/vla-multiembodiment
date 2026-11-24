"""Export the lerobot/pusht human demonstrations to our .npz shard format.

LeRobot stores the images as MP4 video. Decoding video inside the training loop is slow, so we
decode every frame once here and save plain uint8 arrays.

This module runs INSIDE the separate Python 3.12 lerobot env (see vla.data.lerobot_env), never in
the main env: lerobot pins its own torch and numpy. It must only import numpy, lerobot and our
numpy-only helpers.

    python -m vla.data.pusht_loader --out DIR [--max-episodes 5]
"""

import argparse
import time
from pathlib import Path

import numpy as np

from vla.data.shards import ShardWriter, read_shards, summarize, write_meta
from vla.utils.video import save_mp4

REPO_ID = "lerobot/pusht"
IMAGE_KEY = "observation.image"
KEYS = (IMAGE_KEY, "observation.state", "action", "episode_index", "frame_index")
STATE_NAMES = ["agent_x", "agent_y"]
ACTION_NAMES = ["target_x", "target_y"]


def _numpy(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def image_to_uint8_hwc(image) -> np.ndarray:
    """LeRobot gives (3,H,W) float in [0,1]; we store (H,W,3) uint8 in [0,255]."""
    a = _numpy(image)
    if a.ndim == 3 and a.shape[0] == 3:
        a = a.transpose(1, 2, 0)
    if a.dtype != np.uint8:
        a = np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8)
    return a


def iter_episodes(frames):
    """Group consecutive frames into episodes. Yields dicts of images, state, action, seed."""
    current, buffer = None, []

    def finish():
        frame_index = [f["frame_index"] for f in buffer]
        if frame_index != list(range(len(buffer))):
            raise ValueError(f"episode {current}: frames out of order or missing")
        return {
            "images": np.stack([image_to_uint8_hwc(f[IMAGE_KEY]) for f in buffer]),
            "state": np.stack([_numpy(f["observation.state"]) for f in buffer]).astype(np.float32),
            "action": np.stack([_numpy(f["action"]) for f in buffer]).astype(np.float32),
            "seed": -1,  # human demos: no env seed
        }

    for frame in frames:
        frame = dict(frame, episode_index=int(frame["episode_index"]), frame_index=int(frame["frame_index"]))
        if current is not None and frame["episode_index"] != current:
            yield finish()
            buffer = []
        current = frame["episode_index"]
        buffer.append(frame)
    if buffer:
        yield finish()


def frames_to_episodes(frames) -> list:
    return list(iter_episodes(frames))


def iter_frames(dataset, num_workers: int = 4, batch_size: int = 64):
    """Read the dataset in order with several worker processes (video decoding is the slow part)."""
    import torch

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    for batch in loader:
        for i in range(len(batch["action"])):
            yield {key: batch[key][i] for key in KEYS}


def export(out_dir, max_episodes: int | None = None, num_workers: int = 4, n_videos: int = 3) -> dict:
    import lerobot
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out_dir = Path(out_dir)
    t0 = time.time()
    episodes = list(range(max_episodes)) if max_episodes else None
    dataset = LeRobotDataset(REPO_ID, episodes=episodes, video_backend="pyav")
    print(f"{REPO_ID}: {dataset.num_episodes} episodes, {dataset.num_frames} frames, {dataset.fps} fps")

    with ShardWriter(out_dir, shard_size=50) as writer:
        for i, ep in enumerate(iter_episodes(iter_frames(dataset, num_workers))):
            writer.add_episode(state=ep["state"], action=ep["action"], seed=-1, images=ep["images"])
            if i < n_videos:
                save_mp4(out_dir / "videos" / f"pusht_episode{i}.mp4", ep["images"], fps=dataset.fps, scale=4)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1} episodes, {time.time() - t0:.0f}s")

    data = read_shards(out_dir, with_images=False)
    meta = {
        "robot": "pusht",
        "source": REPO_ID,
        "lerobot_version": lerobot.__version__,
        "fps": int(dataset.fps),
        "state_names": STATE_NAMES,
        "action_names": ACTION_NAMES,
        "has_images": True,
        "image_source": f"{REPO_ID} dataset (decoded MP4)",
        "images_for_training": True,
        "image_shape": list(np.load(next(out_dir.glob("shard_*.npz")))["images"].shape[1:]),
        "seconds": round(time.time() - t0, 1),
        **summarize(data),
    }
    write_meta(out_dir, meta)
    print(f"Exported {meta['n_episodes']} episodes, {meta['n_frames']} frames, images {meta['image_shape']}")
    for key in ("state", "action"):
        for name, lo, hi in zip(meta[f"{key}_names"], meta[f"{key}_min"], meta[f"{key}_max"]):
            print(f"  {key:6s} {name:9s} min {lo:9.3f}  max {hi:9.3f}")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-episodes", type=int, default=None, help="export only the first N episodes")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    export(args.out, max_episodes=args.max_episodes, num_workers=args.workers)


if __name__ == "__main__":
    main()
