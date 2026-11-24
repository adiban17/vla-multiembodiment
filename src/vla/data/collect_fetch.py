"""Collect FetchPush demonstrations with the scripted expert.

Runs the expert on env seeds 0, 1, 2, ... and keeps only successful 50-step episodes, until
`n_success` are saved. Writes .npz shards, meta.json (with the expert's raw success rate) and a
few MP4s (when rendering).

Training data must be rendered with EGL on Kaggle (job "data_fetch"), the same renderer as
evaluation. Anything rendered elsewhere is marked debug-only in meta.json.

Mac debug run (no images):   .venv/bin/python -m vla.data.collect_fetch --n 20 --out runs/debug/fetch
"""

import argparse
import platform
import time
from pathlib import Path

import numpy as np

from vla.data.fetch_expert import FetchPushExpert
from vla.data.shards import ShardWriter, read_shards, summarize, write_meta
from vla.envs import fetch
from vla.utils.runtime import is_kaggle
from vla.utils.video import save_mp4

EVAL_SEED_START = 100_000  # seeds 100000-100049 are reserved for evaluation


def run_episode(seed: int, render: bool, noise_std: float) -> dict:
    """One expert episode. Records (image, state, action) at each step before acting."""
    env = fetch.make_env(seed=seed, render=render)
    expert = FetchPushExpert(rng=np.random.default_rng(seed), noise_std=noise_std)
    obs = env.reset()
    images, states, actions = [], [], []
    done, success = False, False
    while not done:
        action = expert.act(obs["raw"])
        if render:
            images.append(obs["image"])
        states.append(obs["state"])
        actions.append(action)
        obs, success, done, _ = env.step(action)
    env.close()
    return {
        "success": success,
        "state": np.stack(states),
        "action": np.stack(actions),
        "images": np.stack(images) if render else None,
    }


def collect(out_dir, n_success: int, render: bool, start_seed: int = 0, n_videos: int = 3,
            noise_std: float = 0.1, shard_size: int = 50, max_attempts: int | None = None) -> dict:
    out_dir = Path(out_dir)
    if max_attempts is None:
        max_attempts = 3 * n_success + 20
    images_for_training = bool(render and is_kaggle() and fetch.renderer_name() == "egl")
    image_source = f"mujoco-{fetch.renderer_name()} on {'kaggle' if is_kaggle() else platform.system()}" if render else None

    attempts, kept, seed, t0 = 0, 0, start_seed, time.time()
    with ShardWriter(out_dir, shard_size=shard_size) as writer:
        while kept < n_success:
            if seed >= EVAL_SEED_START:
                raise RuntimeError("ran into the evaluation seeds (100000+); lower start_seed or n_success")
            if attempts >= max_attempts:
                raise RuntimeError(f"only {kept} successes in {attempts} attempts; the expert is too weak")
            ep = run_episode(seed, render, noise_std)
            attempts += 1
            if ep["success"]:
                writer.add_episode(state=ep["state"], action=ep["action"], seed=seed, images=ep["images"])
                kept += 1
                if render and kept <= n_videos:
                    save_mp4(out_dir / "videos" / f"fetch_seed{seed}.mp4", ep["images"], fps=fetch.FPS, scale=2)
            seed += 1
            if attempts % 50 == 0:
                print(f"  {attempts} attempts, {kept} kept, {time.time() - t0:.0f}s")

    stats = summarize(read_shards(out_dir, with_images=False))
    meta = {
        "robot": "fetch",
        "env_id": fetch.ENV_ID,
        "fps": fetch.FPS,
        "state_names": fetch.STATE_NAMES,
        "action_names": fetch.ACTION_NAMES,
        "has_images": render,
        "image_source": image_source,
        "images_for_training": images_for_training,
        "camera": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in fetch.CAMERA.items()} if render else None,
        "expert": {"noise_std": noise_std},
        "seeds": [start_seed, seed - 1],
        "attempts": attempts,
        "expert_success_rate": kept / attempts,
        "seconds": round(time.time() - t0, 1),
        **stats,
    }
    if render:
        meta["image_shape"] = [fetch.IMAGE_SIZE, fetch.IMAGE_SIZE, 3]
    write_meta(out_dir, meta)
    print_summary(meta)
    return meta


def print_summary(meta: dict) -> None:
    print(f"Expert raw success rate: {meta['expert_success_rate']:.1%} "
          f"({meta['n_episodes']} kept of {meta['attempts']} attempts, seeds {meta['seeds'][0]}-{meta['seeds'][1]})")
    print(f"Frames: {meta['n_frames']}  images: {meta.get('image_shape') if meta['has_images'] else 'none'}  "
          f"source: {meta['image_source']}  usable for training: {meta['images_for_training']}")
    for key in ("state", "action"):
        names = meta[f"{key}_names"]
        for name, lo, hi in zip(names, meta[f"{key}_min"], meta[f"{key}_max"]):
            print(f"  {key:6s} {name:9s} min {lo:9.4f}  max {hi:9.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=20, help="number of successful episodes to keep")
    parser.add_argument("--out", required=True, help="output folder (Mac: use runs/debug/...)")
    parser.add_argument("--render", action="store_true", help="save 128x128 images (EGL on Kaggle)")
    parser.add_argument("--start-seed", type=int, default=0)
    args = parser.parse_args()
    collect(args.out, args.n, render=args.render, start_seed=args.start_seed)


if __name__ == "__main__":
    main()
