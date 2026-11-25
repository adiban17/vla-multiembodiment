"""Phase 4 checks: the rollout evaluation harness. Written before the code.

Gate (docs/spec.md, Phase 4): the scripted expert scores its known rate through the harness, a
random policy scores about 0%, and evaluation sees exactly what training saw.
"""

import json

import numpy as np
import pytest
import torch

from conftest import fake_encode, fake_language
from vla.data.cache_features import build_cache
from vla.data.collect_fetch import run_episode
from vla.data.dataset import WindowDataset
from vla.data.instructions import all_instructions, paraphrases
from vla.data.normalize import Normalizer
from vla.evaluate import (
    ExpertPolicy,
    RandomPolicy,
    VLARolloutPolicy,
    evaluate_policy,
    rollout,
    wilson_interval,
)
from vla.models.policy import build_policy
from vla.utils.config import load_config

EVAL_SEEDS = range(100_000, 100_005)


# --- statistics ---------------------------------------------------------------


def test_wilson_interval():
    lo, hi = wilson_interval(0, 50)
    assert lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(0.0713, abs=1e-3)
    lo, hi = wilson_interval(25, 50)
    assert lo == pytest.approx(0.3664, abs=1e-3) and hi == pytest.approx(0.6336, abs=1e-3)
    lo, hi = wilson_interval(50, 50)
    assert hi == pytest.approx(1.0, abs=1e-12) and lo == pytest.approx(0.9287, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 1.0)  # no data: no information


# --- scripted expert and random policy (no images, CPU) --------------------------


def test_expert_through_harness_matches_the_plain_loop():
    """Same seeds + same expert RNG => the harness must give exactly the plain loop's results."""
    policy = ExpertPolicy()
    assert not policy.needs_images
    results = [rollout("fetch", seed, policy, exec_steps=8) for seed in EVAL_SEEDS]
    for seed, r in zip(EVAL_SEEDS, results):
        plain = run_episode(seed, render=False, noise_std=0.1)
        assert r["success"] == plain["success"], seed
        assert r["steps"] == 50 and r["policy_calls"] == 50  # the expert decides every step
        np.testing.assert_array_equal(np.stack(r["actions"]), plain["action"])
    rate = np.mean([r["success"] for r in results])
    print(f"\nExpert through the harness, seeds {EVAL_SEEDS[0]}-{EVAL_SEEDS[-1]}: {rate:.0%}")

    out = evaluate_policy(policy, robots=("fetch",), n_episodes=5, seed_start=100_000)
    assert out["fetch"]["n"] == 5 and out["fetch"]["successes"] == sum(r["success"] for r in results)
    assert out["fetch"]["wilson_low"] <= out["fetch"]["success_rate"] <= out["fetch"]["wilson_high"]
    assert out["fetch"]["seconds"] > 0 and [e["seed"] for e in out["fetch"]["episodes"]] == list(EVAL_SEEDS)


def test_random_policy_two_pusht_episodes():
    out = evaluate_policy(RandomPolicy(), robots=("pusht",), n_episodes=2, seed_start=100_000)
    p = out["pusht"]
    assert p["n"] == 2 and p["successes"] == 0 and p["success_rate"] == 0.0
    for e in p["episodes"]:
        assert e["steps"] == 300 and not e["success"]
        assert 0.0 <= e["max_coverage"] <= 1.0
    assert 0.0 <= p["mean_max_coverage"] <= 1.0
    assert out["mean_success"] == 0.0 and out["eval_seconds"] > 0


def test_random_policy_actions_are_in_range():
    policy = RandomPolicy()
    for robot, dims, lo, hi in (("pusht", 2, 0.0, 512.0), ("fetch", 4, -1.0, 1.0)):
        policy.reset(robot, seed=1)
        chunk = policy.act(None, None, 0)
        assert chunk.shape == (16, dims) and chunk.min() >= lo and chunk.max() <= hi
    assert np.all(chunk[:, 3] == 0.0)  # FetchPush gripper fixed at 0, like the expert


# --- chunked execution and frame history ---------------------------------------------


class CountingPolicy:
    """Always aims the pusher at one point; records what the harness hands it."""

    needs_images = True

    def reset(self, robot, seed):
        self.calls = []

    def act(self, prev_obs, obs, t):
        self.calls.append({"t": t, "prev_is_obs": prev_obs is obs, "prev_state": prev_obs["state"].copy(),
                           "prev_image": prev_obs["image"].copy(), "image": obs["image"].copy()})
        return np.tile(np.array([[400.0, 100.0]], dtype=np.float32), (16, 1))


def test_chunks_of_8_and_frame_history():
    policy = CountingPolicy()
    r = rollout("pusht", 100_000, policy, exec_steps=8)
    assert r["steps"] == 300 and not r["success"]
    assert [c["t"] for c in policy.calls] == list(range(0, 300, 8))  # 38 calls
    assert r["policy_calls"] == 38
    first = policy.calls[0]
    assert first["prev_is_obs"]  # at t=0 there is no t-1: frame t is repeated, as in training
    states = np.stack(r["states"])  # state before each step's action
    for c in policy.calls[1:]:
        assert not c["prev_is_obs"]
        np.testing.assert_array_equal(c["prev_state"], states[c["t"] - 1])  # the previous ENV step


# --- the VLA adapter: train/eval match and feature reuse ---------------------------------


def _live_language(sentences):
    """Live stand-in for MiniLM that agrees with fake_language over the full sentence list."""
    order = all_instructions()
    return fake_language(order)[[order.index(s) for s in sentences]]


@pytest.fixture
def cache_run(tmp_path, synthetic_data):
    run = tmp_path / "cache"
    build_cache(run, synthetic_data, fake_encode, fake_language)
    return run


def _tiny_model():
    torch.manual_seed(0)
    return build_policy(load_config(["size=tiny"])).eval()


def _vla_policy(cache_run, encode_fn=fake_encode, reuse_features=True):
    normalizer = Normalizer.load(cache_run / "stats.json")
    lang = {robot: _live_language([paraphrases(robot)[0]])[0] for robot in ("pusht", "fetch")}
    return VLARolloutPolicy(_tiny_model(), normalizer, encode_fn, lang, device="cpu",
                            reuse_features=reuse_features)


@pytest.mark.parametrize("robot", ["pusht", "fetch"])
def test_eval_batch_equals_training_sample(cache_run, robot):
    """For the same frame, the eval adapter builds exactly the tensors training used."""
    ds = WindowDataset(cache_run / robot, split="all", mode="cached", instructions=0,
                       require_training_images=False)
    images = np.load(cache_run / robot / "images.npy")
    raw_state = np.load(cache_run / robot / "lowdim.npz")["state"]
    policy = _vla_policy(cache_run)
    for i in (0, 1, 7):  # frame 0 = first step of an episode (frame t repeated)
        sample = ds[i]
        t = int(sample["frame"])
        start = int(ds.episode_starts[int(sample["episode"])])
        obs = {"image": images[t], "state": raw_state[t]}
        prev = obs if t == start else {"image": images[t - 1], "state": raw_state[t - 1]}
        policy.reset(robot, seed=0)
        batch = policy.make_batch(prev, obs, t - start)
        for key in ("vision", "state", "lang", "embodiment_id"):
            torch.testing.assert_close(batch[key][0], sample[key], rtol=0, atol=0, msg=f"{key} differs")


def _informative_encode(frames):
    """A fake encoder whose features depend on the whole image (so the policy reacts to it)."""
    frames = np.asarray(frames, dtype=np.float32)
    mean = frames.reshape(len(frames), -1).mean(axis=1) / 255.0
    corner = frames[:, :8, :8].reshape(len(frames), -1).mean(axis=1) / 255.0
    feats = np.zeros((len(frames), 65, 384), dtype=np.float32)
    feats[:, :, 0::2] = mean[:, None, None]
    feats[:, :, 1::2] = corner[:, None, None]
    return feats


class CountingEncoder:
    def __init__(self):
        self.frames = 0

    def __call__(self, frames):
        self.frames += len(frames)
        return _informative_encode(frames)


@pytest.mark.parametrize("exec_steps, frames_with_reuse, frames_without", [
    (8, 1 + 37 * 2, 38 * 2),   # t=0 encodes its frame once; later calls need t-1 and t, both new
    (1, 300, 300 * 2),          # every step: frame t-1 was frame t of the previous call -> reused
])
def test_feature_reuse_is_identical_and_encodes_each_frame_once(cache_run, exec_steps, frames_with_reuse,
                                                                frames_without):
    results = {}
    for reuse in (True, False):
        encoder = CountingEncoder()
        policy = _vla_policy(cache_run, encode_fn=encoder, reuse_features=reuse)
        r = rollout("pusht", 100_000, policy, exec_steps=exec_steps)
        results[reuse] = (r, encoder.frames)
    (with_reuse, n_with), (without, n_without) = results[True], results[False]
    np.testing.assert_array_equal(np.stack(with_reuse["actions"]), np.stack(without["actions"]))
    assert with_reuse["max_coverage"] == without["max_coverage"]
    assert n_with == frames_with_reuse and n_without == frames_without
    assert n_with <= with_reuse["steps"]  # no frame is encoded twice


def test_vla_actions_are_unnormalized_and_clipped(cache_run):
    policy = _vla_policy(cache_run)
    images = np.load(cache_run / "pusht" / "images.npy")
    state = np.load(cache_run / "pusht" / "lowdim.npz")["state"]
    policy.reset("pusht", seed=0)
    obs = {"image": images[0], "state": state[0]}
    chunk = policy.act(obs, obs, 0)
    assert chunk.shape == (16, 2) and chunk.dtype == np.float32
    assert chunk.min() >= 0.0 and chunk.max() <= 512.0
    policy.reset("fetch", seed=0)
    images = np.load(cache_run / "fetch" / "images.npy")
    state = np.load(cache_run / "fetch" / "lowdim.npz")["state"]
    obs = {"image": images[0], "state": state[0]}
    chunk = policy.act(obs, obs, 0)
    assert chunk.shape == (16, 4) and chunk.min() >= -1.0 and chunk.max() <= 1.0


# --- train.py evaluates the final checkpoint ------------------------------------------


def test_train_evaluates_the_final_model(tmp_path, synthetic_data, monkeypatch):
    import vla.evaluate
    from vla.train import train

    build_cache(tmp_path / "inputs" / "cache", synthetic_data, fake_encode, fake_language)
    monkeypatch.setenv("VLA_OUTPUTS_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("VLA_INPUTS_DIR", str(tmp_path / "inputs"))
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(vla.evaluate, "load_encoders",
                        lambda device: (_informative_encode, _live_language, {"vision_model": "fake"}))

    cfg = load_config(["size=tiny", "train.steps=4", "train.val_every=2", "train.log_every=2",
                       "eval.n_episodes=1", "eval.n_videos=1"])
    run_dir = train(cfg, allow_dirty=True)

    for name in ("stats.json", "language.npy", "ckpt.pt", "metrics.json"):
        assert (run_dir / name).is_file(), name
    m = json.loads((run_dir / "metrics.json").read_text())
    for key in ("pusht_success", "fetch_success", "mean_success", "pusht_wilson_low", "pusht_wilson_high",
                "fetch_wilson_low", "fetch_wilson_high", "pusht_mean_max_coverage", "eval_seconds",
                "eval_minutes", "train_minutes", "total_minutes"):
        assert key in m, key
    assert m["mean_success"] == pytest.approx((m["pusht_success"] + m["fetch_success"]) / 2)
    assert m["eval"]["instruction"]["pusht"] == paraphrases("pusht")[0]
    assert m["eval"]["language_max_abs_diff"] == pytest.approx(0.0)
    assert m["eval"]["pusht"]["n"] == 1 and m["eval"]["fetch"]["n"] == 1
    assert m["run_key"] == "film-frozen-tiny-s0-4steps"
    assert len(list((run_dir / "videos").glob("eval_*.mp4"))) == 2  # one per robot
