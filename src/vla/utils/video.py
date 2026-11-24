"""Save frames as an MP4 you can watch (imageio + its bundled ffmpeg)."""

from pathlib import Path

import numpy as np


def upscale(frames: np.ndarray, factor: int) -> np.ndarray:
    """Make small frames bigger by repeating pixels (no smoothing, so nothing is invented)."""
    if factor == 1:
        return frames
    return frames.repeat(factor, axis=-3).repeat(factor, axis=-2)


def save_mp4(path, frames, fps: int, scale: int = 1) -> Path:
    """frames: (T,H,W,3) uint8. H*scale and W*scale should be multiples of 16 for H.264."""
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, list(upscale(np.asarray(frames), scale)), fps=fps, macro_block_size=16)
    return path
