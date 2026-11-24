"""One on-disk format for both robots: compressed .npz "shards" plus a meta.json.

A dataset folder looks like:
    meta.json            robot, fps, names, stats, where the images came from
    shard_00000.npz      up to `shard_size` episodes each
    shard_00001.npz
    videos/              a few MP4s to look at

Each shard holds, per frame:   images (T,H,W,3 uint8, optional), state, action (float32),
                               episode_index, frame_index (int32)
and, per episode:              episode_seed (int64, -1 if none), episode_length (int32).

This file needs only numpy (+ json), so it also runs inside the separate lerobot env.
"""

import json
from pathlib import Path

import numpy as np


class ShardWriter:
    """Collect episodes and write them as shard_00000.npz, shard_00001.npz, ...

    Use as a context manager so the last partial shard is always written:
        with ShardWriter(out_dir, shard_size=50) as writer:
            writer.add_episode(state=..., action=..., seed=..., images=...)
    """

    def __init__(self, out_dir, shard_size=50):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.n_episodes = 0  # episodes written so far (gives the global episode_index)
        self.paths = []
        self._buffer = []
        self._has_images = None

    def add_episode(self, state, action, seed=-1, images=None):
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        length = len(state)
        if len(action) != length or (images is not None and len(images) != length):
            raise ValueError(f"episode arrays differ in length: state {length}, action {len(action)}, "
                             f"images {None if images is None else len(images)}")
        if self._has_images is None:
            self._has_images = images is not None
        elif self._has_images != (images is not None):
            raise ValueError("every episode in a dataset must either have images or not")
        if images is not None:
            images = np.asarray(images)
            if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
                raise ValueError(f"images must be (T,H,W,3) uint8, got {images.shape} {images.dtype}")

        self._buffer.append(dict(state=state, action=action, seed=int(seed), images=images))
        if len(self._buffer) >= self.shard_size:
            self._flush()

    def _flush(self):
        if not self._buffer:
            return
        first_index = self.n_episodes
        arrays = {
            "state": np.concatenate([ep["state"] for ep in self._buffer]),
            "action": np.concatenate([ep["action"] for ep in self._buffer]),
            "episode_index": np.concatenate(
                [np.full(len(ep["state"]), first_index + i, dtype=np.int32) for i, ep in enumerate(self._buffer)]
            ),
            "frame_index": np.concatenate([np.arange(len(ep["state"]), dtype=np.int32) for ep in self._buffer]),
            "episode_seed": np.array([ep["seed"] for ep in self._buffer], dtype=np.int64),
            "episode_length": np.array([len(ep["state"]) for ep in self._buffer], dtype=np.int32),
        }
        if self._has_images:
            arrays["images"] = np.concatenate([ep["images"] for ep in self._buffer])

        path = self.out_dir / f"shard_{len(self.paths):05d}.npz"
        np.savez_compressed(path, **arrays)
        self.paths.append(path)
        self.n_episodes += len(self._buffer)
        self._buffer = []

    def close(self):
        self._flush()
        return self.paths

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()


def read_shards(data_dir, with_images: bool = True) -> dict:
    """Load every shard in data_dir and join them. Adds episode_starts / episode_ends
    (frame offsets; end is exclusive), so episode i is frames [starts[i], ends[i]).
    with_images=False skips the (large) images array."""
    paths = sorted(Path(data_dir).glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no shard_*.npz files in {data_dir}")
    parts = []
    for path in paths:
        with np.load(path) as shard:
            parts.append({key: shard[key] for key in shard.files if with_images or key != "images"})
    data = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    data["episode_ends"] = np.cumsum(data["episode_length"]).astype(np.int64)
    data["episode_starts"] = data["episode_ends"] - data["episode_length"]
    return data


def summarize(data: dict) -> dict:
    """Counts and per-dimension min/max of state and action (plain lists, so JSON-friendly)."""
    stats = {"n_episodes": int(len(data["episode_length"])), "n_frames": int(len(data["state"]))}
    for key in ("state", "action"):
        stats[f"{key}_min"] = data[key].min(axis=0).round(4).tolist()
        stats[f"{key}_max"] = data[key].max(axis=0).round(4).tolist()
    if "images" in data:
        stats["image_shape"] = list(data["images"].shape[1:])
    return stats


def write_meta(data_dir, meta: dict) -> None:
    (Path(data_dir) / "meta.json").write_text(json.dumps(meta, indent=2))


def read_meta(data_dir) -> dict:
    return json.loads((Path(data_dir) / "meta.json").read_text())


def assert_training_images_ok(meta: dict) -> None:
    """Refuse datasets whose images must not be used for training.

    Training images must come from the same renderer as evaluation: EGL on Kaggle for
    FetchPush, the LeRobot dataset for PushT. Mac-rendered images are debug-only.
    Phase 2 calls this before building the training set.
    """
    if not meta.get("has_images"):
        raise ValueError("dataset has no images, so it cannot be used for training")
    if not meta.get("images_for_training"):
        raise ValueError(
            f"images from {meta.get('image_source', 'unknown source')!r} are debug-only and must not "
            "be used for training (FetchPush training images must be rendered with EGL on Kaggle)"
        )
