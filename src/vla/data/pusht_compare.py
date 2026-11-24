"""Side-by-side check: a PushT training frame (LeRobot dataset) next to a gym-pusht frame.

Training uses dataset images; evaluation uses images that gym-pusht draws live. If they look
different (colours, scale, line widths, goal outline), a policy can do well on validation and
still fail every rollout. The dataset does not store the block's pose, so the pusher is put at
the same position but the block will usually sit elsewhere; compare the overall look.
"""

from pathlib import Path

import numpy as np

from vla.data.shards import read_shards
from vla.envs import pusht
from vla.utils.video import upscale


def _labelled(image: np.ndarray, text: str, height: int = 28) -> np.ndarray:
    from PIL import Image, ImageDraw

    strip = Image.new("RGB", (image.shape[1], height), "white")
    ImageDraw.Draw(strip).text((6, 7), text, fill="black")
    return np.concatenate([np.asarray(strip), image], axis=0)


def save_pusht_compare(shard_dir, out_png, scale: int = 4, episode: int = 0) -> dict:
    import imageio.v3 as iio

    data = read_shards(shard_dir)
    start = int(data["episode_starts"][episode])
    dataset_image = data["images"][start]
    agent_xy = data["state"][start]

    env = pusht.make_env(seed=0)
    env.reset()
    block = env.env.unwrapped.block
    state = [float(agent_xy[0]), float(agent_xy[1]), float(block.position[0]), float(block.position[1]), float(block.angle)]
    env_image = env.reset(options={"reset_to_state": state})["image"]
    env.close()

    if dataset_image.shape != env_image.shape:
        raise ValueError(f"image sizes differ: dataset {dataset_image.shape} vs env {env_image.shape}")

    left = _labelled(upscale(dataset_image, scale), "LeRobot dataset (training)")
    right = _labelled(upscale(env_image, scale), "gym-pusht env (evaluation)")
    gap = np.full((left.shape[0], 8, 3), 255, dtype=np.uint8)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_png, np.concatenate([left, gap, right], axis=1))

    def colours(image):  # the 3 most common colours (background, goal, block...)
        values, counts = np.unique(image.reshape(-1, 3), axis=0, return_counts=True)
        return [tuple(int(c) for c in values[i]) for i in np.argsort(-counts)[:3]]

    report = {
        "image_shape": list(env_image.shape),
        "dataset_top_colours": colours(dataset_image),
        "env_top_colours": colours(env_image),
        "dataset_mean_rgb": dataset_image.reshape(-1, 3).mean(0).round(1).tolist(),
        "env_mean_rgb": env_image.reshape(-1, 3).mean(0).round(1).tolist(),
    }
    print(f"Saved {out_png}: {report}")
    return report
