"""PushT: a round pusher on a 2-D table pushes a T-shaped block onto a T-shaped target.

Wraps gym-pusht ("gym_pusht/PushT-v0"). Physics is pymunk and images are drawn with pygame,
so this runs anywhere, the Mac included, with no GPU or OpenGL.

    obs["image"]  96x96x3 uint8 top-down picture
    obs["state"]  (2,) float32: pusher x, y in pixels (0..512)
    action        (2,): absolute target position for the pusher, pixels (0..512)
    success       block covers >= 95% of the target (episode ends then, or after 300 steps)
"""

import numpy as np

ENV_ID = "gym_pusht/PushT-v0"
FPS = 10
MAX_STEPS = 300
IMAGE_SIZE = 96
SUCCESS_COVERAGE = 0.95
STATE_NAMES = ["agent_x", "agent_y"]
ACTION_NAMES = ["target_x", "target_y"]


class PushTEnv:
    fps = FPS
    max_steps = MAX_STEPS

    def __init__(self, seed: int):
        import gymnasium as gym
        import gym_pusht  # noqa: F401  (importing registers the env with gymnasium)

        self.seed = seed
        self.env = gym.make(
            ENV_ID,
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
            observation_width=IMAGE_SIZE,
            observation_height=IMAGE_SIZE,
            max_episode_steps=MAX_STEPS,
        )

    @staticmethod
    def _obs(raw):
        return {
            "image": np.asarray(raw["pixels"], dtype=np.uint8),
            "state": np.asarray(raw["agent_pos"], dtype=np.float32),
            "raw": raw,
        }

    def reset(self, options=None):
        """Start an episode. The same seed always gives the same starting layout.

        options={"reset_to_state": [agent_x, agent_y, block_x, block_y, block_angle]} places
        everything exactly (used by the dataset-vs-env image comparison)."""
        raw, _ = self.env.reset(seed=self.seed, options=options)
        return self._obs(raw)

    def step(self, action):
        """Returns obs, success, done, info (info["coverage"] is 0..1)."""
        raw, _, terminated, truncated, info = self.env.step(np.asarray(action, dtype=np.float32))
        info = dict(info, coverage=float(info["coverage"]))
        success = info["coverage"] >= SUCCESS_COVERAGE
        return self._obs(raw), success, bool(terminated or truncated or success), info

    def close(self):
        self.env.close()


def make_env(seed: int) -> PushTEnv:
    return PushTEnv(seed)
