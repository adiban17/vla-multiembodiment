"""Phase 1 gate: the scripted FetchPush expert succeeds on at least 80% of 50 seeds (no rendering)."""

import numpy as np

from vla.data.fetch_expert import FetchPushExpert
from vla.envs import fetch


def run_expert_episode(seed):
    env = fetch.make_env(seed=seed, render=False)
    expert = FetchPushExpert(rng=np.random.default_rng(seed))
    obs = env.reset()
    actions, success, done = [], False, False
    while not done:
        action = expert.act(obs["raw"])
        actions.append(action)
        obs, success, done, _ = env.step(action)
    env.close()
    return success, np.array(actions)


def test_expert_actions_are_valid():
    _, actions = run_expert_episode(seed=0)
    assert actions.shape == (50, 4)
    assert actions.min() >= -1.0 and actions.max() <= 1.0
    assert np.all(actions[:, 3] == 0.0)  # gripper command is fixed at 0


def test_expert_success_rate_at_least_80_percent():
    results = [run_expert_episode(seed)[0] for seed in range(50)]
    rate = float(np.mean(results))
    print(f"\nExpert success rate over 50 seeds: {rate:.0%}")
    assert rate >= 0.80, f"expert success {rate:.0%} < 80%"
