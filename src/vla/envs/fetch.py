"""FetchPush: a simulated 7-joint Fetch arm pushes a cube onto a red target on a table.

Wraps Gymnasium-Robotics "FetchPush-v4" (MuJoCo physics, 25 Hz control, 50 steps per episode).

    obs["image"]  128x128x3 uint8 if render=True, else None
    obs["state"]  (8,) float32: hand xyz (m), 2 finger joints (m), goal xyz (m)
                  The cube position is NOT in the state: the policy must see it in the image.
    obs["raw"]    the full gym observation (the scripted expert reads the cube from here)
    action        (4,) in [-1, 1]: hand dx, dy, dz (x 5 cm), gripper (no effect: gripper locked)
    success       cube within 5 cm of the target (info["is_success"]); judged at the final step

Rendering: on Kaggle we use EGL (GPU rendering without a screen). It must be chosen before
MuJoCo is first imported, so this module sets it at import time. On the Mac, MuJoCo's own
renderer is used, and those images are for debugging only (never for training).
"""

import os

import numpy as np

from vla.utils.runtime import is_kaggle

if is_kaggle():
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ENV_ID = "FetchPush-v4"
FPS = 25
MAX_STEPS = 50
IMAGE_SIZE = 128
STATE_NAMES = ["hand_x", "hand_y", "hand_z", "finger_r", "finger_l", "goal_x", "goal_y", "goal_z"]
ACTION_NAMES = ["dx", "dy", "dz", "gripper"]

# A camera close to the table, so the 5 cm cube is visible at 128x128.
# (The library default shows the whole robot from 2.5 m, where the cube is only a few pixels.)
CAMERA = {"distance": 1.0, "azimuth": 180.0, "elevation": -45.0, "lookat": np.array([1.30, 0.75, 0.42])}


def renderer_name() -> str:
    """Which MuJoCo OpenGL backend renders images: "egl" on Kaggle, "default" otherwise."""
    return os.environ.get("MUJOCO_GL", "default")


class FetchPushEnv:
    fps = FPS
    max_steps = MAX_STEPS

    def __init__(self, seed: int, render: bool = False):
        import gymnasium as gym
        import gymnasium_robotics

        gym.register_envs(gymnasium_robotics)
        self.seed = seed
        self.render = render
        kwargs = {}
        if render:
            kwargs = dict(render_mode="rgb_array", width=IMAGE_SIZE, height=IMAGE_SIZE, default_camera_config=CAMERA)
        self.env = gym.make(ENV_ID, max_episode_steps=MAX_STEPS, **kwargs)

    def _obs(self, raw):
        o = raw["observation"]
        state = np.concatenate([o[0:3], o[9:11], raw["desired_goal"]]).astype(np.float32)
        image = np.asarray(self.env.render(), dtype=np.uint8).copy() if self.render else None
        return {"image": image, "state": state, "raw": raw}

    def reset(self):
        """Start an episode. The same seed always gives the same cube and goal positions."""
        raw, _ = self.env.reset(seed=self.seed)
        return self._obs(raw)

    def step(self, action):
        """Returns obs, success, done, info."""
        raw, _, terminated, truncated, info = self.env.step(np.asarray(action, dtype=np.float32))
        success = bool(info["is_success"])
        return self._obs(raw), success, bool(terminated or truncated), info

    def close(self):
        self.env.close()


def make_env(seed: int, render: bool = False) -> FetchPushEnv:
    return FetchPushEnv(seed, render)
