"""Per-robot min-max normalization of states and actions to [-1, 1], saved in stats.json.

Why: PushT actions are pixels (0..512) and FetchPush actions are already in [-1, 1]. Without
rescaling, PushT numbers would swamp FetchPush in the shared loss. Each robot and each dimension
gets its own min and max, computed from the TRAIN split only. The same stats.json is loaded at
evaluation, so the policy's outputs are turned back into real units the same way.

    y = 2 * (x - min) / (max - min) - 1          normalize
    x = (y + 1) / 2 * (max - min) + min          unnormalize

Constant dimensions (FetchPush finger widths, goal_z, the gripper action) would divide by zero.
A dimension whose train range is below `min_range` counts as constant: normalize gives 0 and
unnormalize gives back its value (the midpoint (min+max)/2, which is exact when min == max).
min_range = 0.01 means 1 cm for FetchPush (finger jitter of ~1 mm is noise, not signal) and
0.01 px for PushT.

FetchPush actions use the fixed range [-1, 1], so normalizing leaves them as they are.
Works on numpy arrays and torch tensors.
"""

import json
from pathlib import Path

import numpy as np

MIN_RANGE = 1e-2


def _dim_stats(x: np.ndarray, fixed_range=None, min_range: float = MIN_RANGE, names=None) -> dict:
    x = np.asarray(x, dtype=np.float32)
    if fixed_range is not None:
        lo = np.full(x.shape[1], fixed_range[0], dtype=np.float32)
        hi = np.full(x.shape[1], fixed_range[1], dtype=np.float32)
    else:
        lo, hi = x.min(axis=0), x.max(axis=0)
    constant = (hi - lo) < min_range
    stats = {
        "min": [float(v) for v in lo],
        "max": [float(v) for v in hi],
        "constant": [bool(c) for c in constant],
        "fixed_range": list(fixed_range) if fixed_range is not None else None,
    }
    if names is not None:
        stats["names"] = list(names)
    return stats


def compute_robot_stats(state, action, action_range=None, min_range: float = MIN_RANGE,
                        state_names=None, action_names=None) -> dict:
    """Stats for one robot from its train-split frames. action_range=(-1, 1) for FetchPush."""
    return {
        "state": _dim_stats(state, None, min_range, state_names),
        "action": _dim_stats(action, action_range, min_range, action_names),
        "min_range": min_range,
    }


def save_stats(path, stats: dict) -> None:
    Path(path).write_text(json.dumps(stats, indent=2))


def load_stats(path) -> dict:
    return json.loads(Path(path).read_text())


class Normalizer:
    """normalize / unnormalize with the stats of each robot ("pusht", "fetch") and key ("state", "action").

    Inputs have the robot's REAL number of dimensions in the last axis (2 for PushT actions,
    not the padded 4): slice padded model outputs before calling unnormalize.
    """

    def __init__(self, stats: dict):
        self.stats = stats
        self._arrays = {}
        for robot, robot_stats in stats.items():
            if not isinstance(robot_stats, dict) or "state" not in robot_stats:
                continue  # other entries, such as the episode split
            for key in ("state", "action"):
                s = robot_stats[key]
                lo = np.asarray(s["min"], dtype=np.float32)
                hi = np.asarray(s["max"], dtype=np.float32)
                const = np.asarray(s["constant"], dtype=bool)
                span = np.where(const, np.float32(1), hi - lo).astype(np.float32)
                center = ((lo + hi) / 2).astype(np.float32)
                self._arrays[robot, key] = (lo, span, center, const)

    @classmethod
    def load(cls, path) -> "Normalizer":
        return cls(load_stats(path))

    def _get(self, x, robot, key):
        lo, span, center, const = self._arrays[robot, key]
        if x.shape[-1] != len(lo):
            raise ValueError(f"{robot} {key} has {len(lo)} dims, got input with {x.shape[-1]}")
        try:
            import torch

            if isinstance(x, torch.Tensor):
                to = lambda a: torch.as_tensor(a, device=x.device)  # noqa: E731
                return (to(lo).to(x.dtype), to(span).to(x.dtype), to(center).to(x.dtype), to(const), torch.where)
        except ImportError:
            pass
        return lo, span, center, const, np.where

    def normalize(self, x, robot: str, key: str):
        if not hasattr(x, "shape"):
            x = np.asarray(x, dtype=np.float32)
        lo, span, _, const, where = self._get(x, robot, key)
        y = 2 * (x - lo) / span - 1
        return where(const, 0 * y, y)

    def unnormalize(self, y, robot: str, key: str):
        if not hasattr(y, "shape"):
            y = np.asarray(y, dtype=np.float32)
        lo, span, center, const, where = self._get(y, robot, key)
        x = (y + 1) / 2 * span + lo
        return where(const, center + 0 * x, x)
