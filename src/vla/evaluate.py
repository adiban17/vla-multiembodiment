"""Closed-loop rollout evaluation (docs/spec.md "Evaluation protocol").

Every policy faces the same 50 starting states per robot: env seeds 100000-100049. The loop:

    observe -> policy predicts a chunk of actions (real units) -> execute the first 8 -> repeat

    PushT      up to 300 steps; success if coverage >= 0.95 at ANY step; also the max coverage
    FetchPush  50 steps; success if the cube is within 5 cm of the goal at the FINAL step

A "policy" here is any object with:
    needs_images            does it look at camera images? (FetchPush renders only if needed)
    reset(robot, seed)      called at the start of each episode
    act(prev_obs, obs, t)   -> (k, action_dim) actions in real units. obs is the current
                            observation at env step t, prev_obs the one at step t-1 (at t = 0
                            there is none, so prev_obs IS obs, exactly as in training).

Policies: ExpertPolicy (scripted FetchPush expert, a sanity check), RandomPolicy (should score
about 0%), VLARolloutPolicy (a trained model; its images go through the same
preprocess_image() + encode_frames() path as the feature cache), BaselineRolloutPolicy (a
per-robot baseline; it gets the raw uint8 frames, exactly as in training, and is evaluated on its
own robot only) and AdaptedVLARolloutPolicy (Phase 6: a VLA run whose DINOv2 was fine-tuned or
LoRA-adapted; it gets the raw frames and encodes them with ITS OWN adapted DINOv2, never the frozen
encoder from the cache).

Command line (Mac examples):
    python -m vla.evaluate --policy expert --robots fetch --episodes 5
    python -m vla.evaluate --policy random --robots pusht --episodes 2
    python -m vla.evaluate --policy outputs/<run_id> --paraphrase 8      # held-out instruction
    python -m vla.evaluate --policy outputs/<baseline run_id>            # its own robot only
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from vla.data.dataset import EMBODIMENT_ID, STATE_DIM, _pad_last
from vla.data.fetch_expert import FetchPushExpert
from vla.data.instructions import all_instructions, paraphrases
from vla.data.normalize import Normalizer
from vla.envs import fetch, pusht
from vla.utils.video import save_mp4

ROBOTS = ("pusht", "fetch")
SEED_START = 100_000
ACTION_BOUNDS = {"pusht": (0.0, 512.0), "fetch": (-1.0, 1.0)}  # the envs' action spaces
ACTION_DIMS = {"pusht": 2, "fetch": 4}
VIDEO_SCALE = {"pusht": 4, "fetch": 2}  # 96 -> 384 px, 128 -> 256 px
FPS = {"pusht": pusht.FPS, "fetch": fetch.FPS}


# --- statistics ------------------------------------------------------------------


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n. Unlike "p +- 1.96 * sd", it stays
    inside [0, 1] and is still sensible at 0/50 or 50/50."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


# --- policies --------------------------------------------------------------------


class ExpertPolicy:
    """The scripted FetchPush expert (it reads the cube position from the simulator).
    Same RNG seeding as data collection, so the harness can be checked against the plain loop."""

    needs_images = False

    def __init__(self, noise_std: float = 0.1):
        self.noise_std = noise_std

    def reset(self, robot, seed):
        if robot != "fetch":
            raise ValueError("the scripted expert exists for FetchPush only")
        self.expert = FetchPushExpert(rng=np.random.default_rng(seed), noise_std=self.noise_std)

    def act(self, prev_obs, obs, t):
        return self.expert.act(obs["raw"])[None]  # one action: the expert decides every step


class RandomPolicy:
    """16 uniformly random actions per call (PushT: anywhere on the table; FetchPush: random hand
    moves, gripper 0). A random policy should score about 0%."""

    needs_images = False
    chunk = 16

    def reset(self, robot, seed):
        self.robot = robot
        self.rng = np.random.default_rng(seed)

    def act(self, prev_obs, obs, t):
        lo, hi = ACTION_BOUNDS[self.robot]
        if self.robot == "pusht":
            return self.rng.uniform(lo, hi, size=(self.chunk, 2)).astype(np.float32)
        moves = self.rng.uniform(lo, hi, size=(self.chunk, 3))
        return np.concatenate([moves, np.zeros((self.chunk, 1))], axis=1).astype(np.float32)


def to_env_actions(pred: torch.Tensor, normalizer: Normalizer, robot: str) -> np.ndarray:
    """(16, 4) model output in normalized units -> (16, action_dim) real actions, clipped to the env's range."""
    dims = ACTION_DIMS[robot]
    actions = normalizer.unnormalize(pred.float().cpu().numpy()[:, :dims], robot, "action")
    lo, hi = ACTION_BOUNDS[robot]
    return np.clip(actions, lo, hi).astype(np.float32)


def state_input(normalizer: Normalizer, obs: dict, robot: str) -> np.ndarray:
    """The env's raw state -> the normalized, padded (8,) state that training saw."""
    state = normalizer.normalize(np.asarray(obs["state"], dtype=np.float32), robot, "state")
    return np.ascontiguousarray(_pad_last(state, STATE_DIM))


class VLARolloutPolicy:
    """A trained VLA policy driving an env.

    model          VLAPolicy (frozen-encoder path: it reads DINOv2 features)
    normalizer     Normalizer from the run's stats.json (a copy of the cache's)
    encode_fn      uint8 frames (N, H, W, 3) -> float32 (N, 65, 384); in real use
                   encode_frames(DINOv2, ...), which calls the one preprocess_image()
    lang_vectors   {"pusht": (384,), "fetch": (384,)} MiniLM vectors of the instruction
    reuse_features keep the DINOv2 features of the last call's frames, keyed by env step, so no
                   frame is encoded twice. With 8 actions per call, frame t-1 was not seen by the
                   previous call (it saw t-9 and t-8), so this saves little there: only at t = 0,
                   where frame t is used twice. With 1 action per call it halves the encoding.
                   The results are identical either way (a test checks this).
    """

    needs_images = True

    def __init__(self, model, normalizer: Normalizer, encode_fn, lang_vectors: dict, device: str,
                 use_amp: bool = False, reuse_features: bool = True):
        self.model = model
        self.normalizer = normalizer
        self.encode_fn = encode_fn
        self.lang_vectors = {r: np.asarray(v, dtype=np.float32) for r, v in lang_vectors.items()}
        self.device = device
        self.use_amp = use_amp
        self.reuse_features = reuse_features
        self.robot = None
        self._features = {}  # env step -> (65, 384) features of that step's frame

    def reset(self, robot, seed):
        self.robot = robot
        self._features = {}

    def _encode(self, images) -> np.ndarray:
        feats = np.asarray(self.encode_fn(np.stack(images)), dtype=np.float32)
        # The feature cache stores fp16, so training saw fp16-rounded features: round the same way.
        return feats.astype(np.float16).astype(np.float32)

    def _frame_features(self, steps, images) -> list[np.ndarray]:
        if not self.reuse_features:
            return list(self._encode(images))
        missing = {}
        for step, image in zip(steps, images):
            if step not in self._features and step not in missing:
                missing[step] = image
        if missing:
            new = self._encode(list(missing.values()))
            self._features.update(zip(missing, new))
        feats = [self._features[s] for s in steps]
        self._features = dict(zip(steps, feats))  # keep only what the next call can reuse
        return feats

    def make_batch(self, prev_obs, obs, t) -> dict:
        """The model input for env step t: the same tensors WindowDataset gives in training."""
        robot = self.robot
        steps = [t - 1 if prev_obs is not obs else t, t]
        vision = np.stack(self._frame_features(steps, [prev_obs["image"], obs["image"]]))
        state = state_input(self.normalizer, obs, robot)
        return {
            "vision": torch.from_numpy(vision)[None],
            "state": torch.from_numpy(state)[None],
            "lang": torch.from_numpy(self.lang_vectors[robot])[None],
            "embodiment_id": torch.tensor([EMBODIMENT_ID[robot]], dtype=torch.int64),
        }

    def act(self, prev_obs, obs, t):
        batch = {k: v.to(self.device) for k, v in self.make_batch(prev_obs, obs, t).items()}
        with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=torch.float16,
                                                    enabled=self.use_amp):
            pred = self.model(batch)[0]  # (16, 4), normalized units
        return to_env_actions(pred, self.normalizer, self.robot)


class BaselineRolloutPolicy:
    """A trained per-robot baseline (BC-MLP or ResNet-18) driving an env.

    It gets the raw uint8 frames t-1 and t, the same "pixels" WindowDataset gave in training; the
    model itself runs preprocess_image(). State normalization and action unnormalization are the
    VLA's (the run's own stats.json).
    """

    needs_images = True

    def __init__(self, model, normalizer: Normalizer, device: str, use_amp: bool = False):
        self.model = model
        self.normalizer = normalizer
        self.device = device
        self.use_amp = use_amp
        self.robot = None

    def reset(self, robot, seed):
        self.robot = robot

    def make_batch(self, prev_obs, obs, t) -> dict:
        """The model input for env step t: the same tensors WindowDataset(mode="pixels") gives."""
        pixels = np.stack([np.asarray(prev_obs["image"], dtype=np.uint8), np.asarray(obs["image"], dtype=np.uint8)])
        return {
            "pixels": torch.from_numpy(pixels)[None],
            "state": torch.from_numpy(state_input(self.normalizer, obs, self.robot))[None],
            "embodiment_id": torch.tensor([EMBODIMENT_ID[self.robot]], dtype=torch.int64),
        }

    def act(self, prev_obs, obs, t):
        batch = {k: v.to(self.device) for k, v in self.make_batch(prev_obs, obs, t).items()}
        with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=torch.float16,
                                                    enabled=self.use_amp):
            pred = self.model(batch)[0]
        return to_env_actions(pred, self.normalizer, self.robot)


class AdaptedVLARolloutPolicy(BaselineRolloutPolicy):
    """A trained adaptation run (AdaptedVLAPolicy: full_ft or lora16) driving an env.

    Like the baselines, it hands the model the raw uint8 frames t-1 and t; the model runs
    preprocess_image() and its own adapted DINOv2, exactly as in training. Plus the instruction's
    MiniLM vector (from the run's language.npy: MiniLM is frozen, so these are the vectors training saw).
    """

    def __init__(self, model, normalizer: Normalizer, lang_vectors: dict, device: str, use_amp: bool = False):
        super().__init__(model, normalizer, device, use_amp)
        self.lang_vectors = {r: np.asarray(v, dtype=np.float32) for r, v in lang_vectors.items()}

    def make_batch(self, prev_obs, obs, t) -> dict:
        """The model input for env step t: the same tensors WindowDataset(mode="pixels") gives the VLA."""
        batch = super().make_batch(prev_obs, obs, t)
        batch["lang"] = torch.from_numpy(self.lang_vectors[self.robot])[None]
        return batch


# --- one episode -------------------------------------------------------------------


def make_env(robot: str, seed: int, render: bool):
    if robot == "pusht":
        return pusht.make_env(seed)  # PushT always draws its image (pygame, cheap)
    if robot == "fetch":
        return fetch.make_env(seed, render=render)
    raise ValueError(f"unknown robot {robot!r}")


def rollout(robot: str, seed: int, policy, exec_steps: int = 8, keep_frames: bool = False) -> dict:
    """Run one episode. Each policy call returns a chunk; the first `exec_steps` are executed."""
    env = make_env(robot, seed, render=policy.needs_images or keep_frames)
    policy.reset(robot, seed)
    t0 = time.perf_counter()
    obs = env.reset()
    prev = obs  # no frame t-1 at the start: frame t is repeated
    frames = [obs["image"]] if keep_frames else None
    actions, states = [], []
    t, calls, success, max_coverage, done = 0, 0, False, 0.0, False
    while not done:
        chunk = np.asarray(policy.act(prev, obs, t), dtype=np.float32)
        calls += 1
        for action in chunk[:exec_steps]:
            states.append(obs["state"].copy())
            actions.append(action)
            prev = obs
            obs, step_success, done, info = env.step(action)
            t += 1
            if keep_frames:
                frames.append(obs["image"])
            if robot == "pusht":
                max_coverage = max(max_coverage, info["coverage"])
                success = success or step_success  # at any step
            else:
                success = step_success  # judged at the final step
            if done:
                break
    env.close()
    return {"seed": seed, "success": bool(success), "steps": t, "policy_calls": calls,
            "max_coverage": float(max_coverage) if robot == "pusht" else None,
            "seconds": time.perf_counter() - t0, "actions": actions, "states": states, "frames": frames}


# --- many episodes -------------------------------------------------------------------


def _pick_videos(kept: dict, random_idx: int, n_videos: int) -> list[int]:
    """Episode indices to save: first success, first failure, the random one, then fill up."""
    order = [kept["success"][:1], kept["failure"][:1], [random_idx] if random_idx in kept["frames"] else [],
             kept["success"][1:], kept["failure"][1:]]
    picked = []
    for group in order:
        for idx in group:
            if idx not in picked and len(picked) < n_videos:
                picked.append(idx)
    return picked


def evaluate_robot(policy, robot: str, n_episodes: int = 50, seed_start: int = SEED_START, exec_steps: int = 8,
                   n_videos: int = 0, video_dir=None, video_tag: str = "", log=print) -> dict:
    random_idx = int(np.random.default_rng(0).integers(n_episodes))  # the "random" video, same for every run
    kept = {"success": [], "failure": [], "frames": {}}  # at most 2 + 2 + 1 episodes of frames in memory
    episodes, calls, t0 = [], 0, time.perf_counter()
    for i in range(n_episodes):
        seed = seed_start + i
        want = n_videos > 0 and (len(kept["success"]) < 2 or len(kept["failure"]) < 2 or i == random_idx)
        r = rollout(robot, seed, policy, exec_steps, keep_frames=want)
        calls += r["policy_calls"]
        episodes.append({"seed": seed, "success": r["success"], "steps": r["steps"], "max_coverage": r["max_coverage"]})
        if want:
            group = kept["success" if r["success"] else "failure"]
            if len(group) < 2 or i == random_idx:
                if len(group) < 2:
                    group.append(i)
                kept["frames"][i] = r["frames"]
        if (i + 1) % 10 == 0 or i + 1 == n_episodes:
            done = sum(e["success"] for e in episodes)
            log(f"  {robot}: {i + 1}/{n_episodes} episodes, {done} successes, {time.perf_counter() - t0:.0f}s")

    k = sum(e["success"] for e in episodes)
    lo, hi = wilson_interval(k, n_episodes)
    result = {"n": n_episodes, "successes": k, "success_rate": k / n_episodes, "wilson_low": lo, "wilson_high": hi,
              "seconds": time.perf_counter() - t0, "policy_calls": calls, "episodes": episodes, "videos": []}
    if robot == "pusht":
        result["mean_max_coverage"] = float(np.mean([e["max_coverage"] for e in episodes]))
    if n_videos > 0 and video_dir is not None:
        for idx in _pick_videos(kept, random_idx, n_videos):
            e = episodes[idx]
            label = ("success" if e["success"] else "fail") + ("_random" if idx == random_idx else "")
            path = Path(video_dir) / f"eval{video_tag}_{robot}_seed{e['seed']}_{label}.mp4"
            save_mp4(path, np.stack(kept["frames"][idx]), fps=FPS[robot], scale=VIDEO_SCALE[robot])
            result["videos"].append(str(path))
    return result


def evaluate_policy(policy, robots=ROBOTS, n_episodes: int = 50, seed_start: int = SEED_START, exec_steps: int = 8,
                    n_videos: int = 0, video_dir=None, video_tag: str = "", log=print) -> dict:
    """{robot: per-robot result, "mean_success": average over robots, "eval_seconds"}."""
    t0 = time.perf_counter()
    out = {}
    for robot in robots:
        out[robot] = evaluate_robot(policy, robot, n_episodes, seed_start, exec_steps, n_videos, video_dir,
                                    video_tag, log)
    out["mean_success"] = float(np.mean([out[r]["success_rate"] for r in robots]))
    out["eval_seconds"] = time.perf_counter() - t0
    return out


def flat_metrics(result: dict) -> dict:
    """The headline numbers as flat keys (pusht_success, fetch_wilson_low, mean_success, ...)."""
    flat = {}
    for robot in ROBOTS:
        if robot not in result:
            continue
        r = result[robot]
        flat.update({f"{robot}_success": r["success_rate"], f"{robot}_successes": r["successes"], f"{robot}_n": r["n"],
                     f"{robot}_wilson_low": r["wilson_low"], f"{robot}_wilson_high": r["wilson_high"],
                     f"{robot}_eval_seconds": r["seconds"]})
        if "mean_max_coverage" in r:
            flat[f"{robot}_mean_max_coverage"] = r["mean_max_coverage"]
    flat["mean_success"] = result["mean_success"]
    flat["eval_seconds"] = result["eval_seconds"]
    return flat


def print_summary(result: dict, title: str = "Evaluation") -> None:
    print(f"=== {title} ===")
    for robot in ROBOTS:
        if robot in result:
            r = result[robot]
            extra = f" | mean max coverage {r['mean_max_coverage']:.3f}" if "mean_max_coverage" in r else ""
            print(f"{robot:6s} success {r['successes']:3d}/{r['n']} = {r['success_rate']:6.1%} "
                  f"(95% Wilson {r['wilson_low']:.1%} to {r['wilson_high']:.1%}){extra} | {r['seconds']:.0f}s, "
                  f"{r['policy_calls']} policy calls")
    print(f"mean success {result['mean_success']:.1%} | evaluation took {result['eval_seconds'] / 60:.1f} min")


# --- a trained run ----------------------------------------------------------------------


def load_encoders(device: str):
    """DINOv2 + MiniLM on `device` -> (encode_fn, language_fn, info). Tests replace this."""
    from vla.models.encoders import DinoV2Encoder, MiniLMEncoder, encode_frames

    dino = DinoV2Encoder().to(device)
    minilm = MiniLMEncoder().to(device)
    encode = lambda frames: encode_frames(dino, frames, device, batch_size=64)  # noqa: E731
    return encode, minilm.encode, {**dino.info(), **minilm.info()}


def language_vectors(language_fn, language_npy, paraphrase: int) -> tuple[dict, dict, float]:
    """Live MiniLM vectors of paraphrase k for each robot, the sentences, and the largest
    difference from the cache's language.npy (it should be about 0: same model, same sentence)."""
    cached = np.load(language_npy)
    order = all_instructions()
    sentences = {robot: paraphrases(robot, "all")[paraphrase] for robot in ROBOTS}
    live = np.asarray(language_fn([sentences[r] for r in ROBOTS]), dtype=np.float32)
    diff = max(float(np.abs(live[i] - cached[order.index(sentences[r])]).max()) for i, r in enumerate(ROBOTS))
    return {r: live[i] for i, r in enumerate(ROBOTS)}, sentences, diff


def cached_language_vectors(language_npy, paraphrase: int) -> tuple[dict, dict]:
    """Paraphrase k's MiniLM vector for each robot, read from a language.npy, and the sentences.
    Used for adaptation runs, which never load MiniLM (it is frozen in every regime)."""
    cached = np.load(language_npy).astype(np.float32)
    order = all_instructions()
    sentences = {robot: paraphrases(robot, "all")[paraphrase] for robot in ROBOTS}
    return {r: cached[order.index(sentences[r])] for r in ROBOTS}, sentences


def _cache_vision_revision(cache_dir) -> str | None:
    if cache_dir is None:
        return None
    from vla.data.shards import read_meta

    try:
        return read_meta(Path(cache_dir) / "fetch").get("vision_revision")
    except FileNotFoundError:
        return None


def evaluate_run(model, run_dir, cfg, device: str, wb=None, cache_dir=None, paraphrase: int | None = None,
                 n_episodes: int | None = None, n_videos: int | None = None) -> dict:
    """Evaluate a trained model with the run's own stats.json and language.npy. Logs to `wb` if given.

    A model with its own DINOv2 (an adaptation run) is evaluated with THAT encoder: it gets raw frames
    and the frozen DINOv2/MiniLM are never loaded. Otherwise the frozen encoders are loaded and frames
    go through encode_frames(), as in the feature cache."""
    run_dir = Path(run_dir)
    e = cfg.eval
    paraphrase = int(e.paraphrase if paraphrase is None else paraphrase)
    n_episodes = int(e.n_episodes if n_episodes is None else n_episodes)
    n_videos = int(e.n_videos if n_videos is None else n_videos)
    use_amp = bool(cfg.train.amp) and device == "cuda"

    normalizer = Normalizer.load(run_dir / "stats.json")
    warnings = []
    model.eval()
    if getattr(model, "encoder", None) is not None:  # adaptation run: its own DINOv2, cached MiniLM vectors
        lang, sentences = cached_language_vectors(run_dir / "language.npy", paraphrase)
        diff = None
        info = {**model.encoder.info(), "vision_adaptation": model.encoder.mode(),
                "language": "run's language.npy (MiniLM frozen)"}
        print(f"Adapted DINOv2 ({model.encoder.mode()}): rollouts use this run's own encoder weights")
        policy = AdaptedVLARolloutPolicy(model, normalizer, lang, device, use_amp)
    else:
        encode, language_fn, info = load_encoders(device)
        lang, sentences, diff = language_vectors(language_fn, run_dir / "language.npy", paraphrase)
        if diff > 1e-3:
            warnings.append(f"live MiniLM vectors differ from the cache's language.npy by {diff:.2e}")
        cache_rev, eval_rev = _cache_vision_revision(cache_dir), info.get("vision_revision")
        if cache_rev and eval_rev and cache_rev != eval_rev:
            warnings.append(f"DINOv2 weights differ: cache {cache_rev}, evaluation {eval_rev}")
        for w in warnings:
            print(f"WARNING: {w}. Evaluation may not see what training saw.")
        policy = VLARolloutPolicy(model, normalizer, encode, lang, device, use_amp)
    tag = "" if paraphrase == 0 else f"_p{paraphrase}"
    print(f"Evaluating {n_episodes} episodes per robot (seeds {e.seed_start}-{e.seed_start + n_episodes - 1}), "
          f"executing {e.exec_steps} of 16 actions per call, instruction paraphrase {paraphrase}")
    result = evaluate_policy(policy, ROBOTS, n_episodes, int(e.seed_start), int(e.exec_steps), n_videos,
                             run_dir / "videos", tag)
    result.update({"paraphrase": paraphrase, "heldout_instruction": paraphrase >= 8, "instruction": sentences,
                   "language_max_abs_diff": diff, "encoders": info, "warnings": warnings,
                   "exec_steps": int(e.exec_steps), "seed_start": int(e.seed_start)})
    print_summary(result, f"Evaluation of {run_dir.name}")
    log_eval(wb, result, tag)
    return result


def log_eval(wb, result: dict, tag: str = "") -> None:
    """Headline numbers and rollout videos to W&B (does nothing without a run)."""
    if wb is None:
        return
    import wandb

    flat = flat_metrics(result)
    log = {f"eval{tag}/{k}": v for k, v in flat.items()}
    for robot in ROBOTS:
        for path in result.get(robot, {}).get("videos", []):
            log[f"video{tag}/{Path(path).stem}"] = wandb.Video(path, format="mp4")
    wb.log(log)
    wb.summary.update({f"eval{tag}/{k}": v for k, v in flat.items()})
    # which start seeds succeeded, one character per episode ("0110..."), so scripts/aggregate.py can run a
    # paired bootstrap from W&B alone (the totals above do not say WHICH starts succeeded)
    wb.summary.update({f"eval{tag}/{robot}_episode_success": episode_string(result[robot]["episodes"])
                       for robot in ROBOTS if robot in result})


def episode_string(episodes: list[dict]) -> str:
    """'1' for a success and '0' for a failure, in start-seed order."""
    return "".join("1" if e["success"] else "0" for e in sorted(episodes, key=lambda e: e["seed"]))


def evaluate_baseline_run(model, run_dir, cfg, device: str, wb=None, n_episodes: int | None = None,
                          n_videos: int | None = None) -> dict:
    """Evaluate a trained baseline on its own robot (cfg.robot), same seeds and chunking as the VLA.
    "mean_success" is then that robot's success rate."""
    run_dir = Path(run_dir)
    e = cfg.eval
    n_episodes = int(e.n_episodes if n_episodes is None else n_episodes)
    n_videos = int(e.n_videos if n_videos is None else n_videos)
    use_amp = bool(cfg.train.amp) and device == "cuda"
    model.eval()
    policy = BaselineRolloutPolicy(model, Normalizer.load(run_dir / "stats.json"), device, use_amp)
    print(f"Evaluating baseline {cfg.baseline} on {cfg.robot}: {n_episodes} episodes "
          f"(seeds {e.seed_start}-{e.seed_start + n_episodes - 1}), executing {e.exec_steps} of 16 actions per call")
    result = evaluate_policy(policy, (cfg.robot,), n_episodes, int(e.seed_start), int(e.exec_steps), n_videos,
                             run_dir / "videos")
    result.update({"baseline": cfg.baseline, "robot": cfg.robot, "exec_steps": int(e.exec_steps),
                   "seed_start": int(e.seed_start)})
    print_summary(result, f"Evaluation of {run_dir.name}")
    log_eval(wb, result)
    return result


def load_run(run_dir, device: str):
    """(model, cfg) from a run folder's ckpt.pt."""
    from omegaconf import OmegaConf

    from vla.models.policy import build_model, is_baseline

    ckpt = torch.load(Path(run_dir) / "ckpt.pt", map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    if is_baseline(cfg):
        cfg.bc.pretrained = False  # the checkpoint holds the trained weights; no ImageNet download
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval(), cfg


def main(argv=None) -> None:
    from vla.utils.runtime import get_device

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, help="expert, random, or a run folder with ckpt.pt")
    parser.add_argument("--robots", nargs="+", choices=ROBOTS, help="default: both (expert: fetch)")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--exec-steps", type=int, default=8)
    parser.add_argument("--paraphrase", type=int, default=0, help="0-7 training sentences, 8-9 held out")
    parser.add_argument("--videos", type=int, default=0, help="videos per robot (run folders only)")
    args = parser.parse_args(argv)

    if args.policy in ("expert", "random"):
        policy = ExpertPolicy() if args.policy == "expert" else RandomPolicy()
        robots = args.robots or (("fetch",) if args.policy == "expert" else ROBOTS)
        result = evaluate_policy(policy, robots, args.episodes, SEED_START, args.exec_steps)
        print_summary(result, f"{args.policy} policy")
        return

    from vla.models.policy import is_baseline

    device = get_device()
    model, cfg = load_run(args.policy, device)
    cfg.eval.exec_steps = args.exec_steps
    if is_baseline(cfg):
        result = evaluate_baseline_run(model, args.policy, cfg, device, n_episodes=args.episodes, n_videos=args.videos)
        out = Path(args.policy) / "eval.json"
        out.write_text(json.dumps({**flat_metrics(result), "eval": result}, indent=2))
        print(f"Saved {out}")
        return
    result = evaluate_run(model, args.policy, cfg, device, cache_dir=cfg.run.cache_dir, paraphrase=args.paraphrase,
                          n_episodes=args.episodes, n_videos=args.videos)
    out = Path(args.policy) / f"eval_p{args.paraphrase}.json"
    out.write_text(json.dumps({**flat_metrics(result), "eval": result}, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
