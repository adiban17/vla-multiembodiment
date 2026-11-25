"""The one entry point for all work run on Kaggle (and locally on CPU).

Kaggle notebook cell 2 runs:   python scripts/kaggle_job.py --job <name>
A job with a step count:        python scripts/kaggle_job.py --job batch_adaptation --steps 3000
List the jobs:                  python scripts/kaggle_job.py --list

To add work, write a function and decorate it with @register("name"). Each job returns
its run folder. Every job must save its resolved config and git commit (save_run_info).
"""

import argparse
import inspect
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import run_queue` works however we are loaded

from vla.utils.config import load_config
from vla.utils.runtime import get_device, is_kaggle, load_dotenv_if_local, outputs_dir
from vla.utils.tracking import check_wandb_key, git_commit, save_run_info

WANDB_PROJECT = "vla-multiembodiment"

JOBS = {}  # job name -> function


def register(name):
    """Decorator that adds a function to JOBS under `name`."""

    def add(fn):
        JOBS[name] = fn
        return fn

    return add


def new_run_dir(job: str, commit: str) -> Path:
    """<outputs>/<job>-<commit7>-<timestamp>/, for example hello-3f9a2c1-20261125-213000."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return outputs_dir() / f"{job}-{commit[:7]}-{stamp}"


# --- jobs ---------------------------------------------------------------------


@register("hello")
def hello() -> Path:
    """Phase 0 gate: print GPUs and git commit, log a tiny test run to W&B."""
    import torch
    import wandb

    device = get_device()
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    git = git_commit()

    print(f"Python {platform.python_version()} | torch {torch.__version__} | CUDA {torch.version.cuda}")
    print(f"On Kaggle: {is_kaggle()} | device: {device}")
    print("GPUs:", ", ".join(f"[{i}] {name}" for i, name in enumerate(gpus)) or "no CUDA GPU (CPU)")
    print(f"Git commit: {git['commit']} (uncommitted changes: {git['dirty']})")

    run_dir = new_run_dir("hello", git["commit"])
    config = {
        "job": "hello",
        "seed": 0,
        "steps": 50,
        "lr": 0.1,
        "device": device,
        "gpus": gpus,
        "on_kaggle": is_kaggle(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),  # TorchVersion object -> plain string for YAML
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
    }
    save_run_info(run_dir, config, git)

    run = wandb.init(
        project=WANDB_PROJECT, job_type="hello", name=run_dir.name, config=config, dir=str(run_dir)
    )

    # Tiny test run: learn y = 3x + 1 with a single linear layer.
    torch.manual_seed(config["seed"])
    x = torch.randn(256, 1, device=device)
    y = 3 * x + 1
    model = torch.nn.Linear(1, 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
    for step in range(config["steps"]):
        loss = torch.nn.functional.mse_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        wandb.log({"loss": loss.item()}, step=step)

    metrics = {
        "final_loss": loss.item(),
        "weight": model.weight.item(),
        "bias": model.bias.item(),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    run.summary.update(metrics)
    run.finish()

    print(f"Tiny run: loss {metrics['final_loss']:.5f}, learned y = {metrics['weight']:.3f}x + {metrics['bias']:.3f}")
    print(f"Saved run folder: {run_dir}")
    return run_dir


def log_dataset(job: str, run_dir: Path, config: dict, meta: dict, images: list | None = None) -> None:
    """Log a data job's stats, videos and images to W&B, and save metrics.json."""
    import wandb

    metrics = {k: v for k, v in meta.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    run = wandb.init(project=WANDB_PROJECT, job_type="data", name=run_dir.name,
                     config={**config, "meta": meta}, dir=str(run_dir))
    media = {f"video/{p.stem}": wandb.Video(str(p), format="mp4") for p in sorted((run_dir / "videos").glob("*.mp4"))}
    media.update({f"image/{Path(p).stem}": wandb.Image(str(p)) for p in images or []})
    if media:
        run.log(media)
    run.summary.update(metrics)
    run.finish()
    print(f"Saved run folder: {run_dir}")


def _start_data_run(job: str, extra: dict) -> tuple[Path, dict]:
    git = git_commit()
    run_dir = new_run_dir(job, git["commit"])
    config = {"job": job, "on_kaggle": is_kaggle(), "python": platform.python_version(),
              "git_commit": git["commit"], "git_dirty": git["dirty"], **extra}
    save_run_info(run_dir, config, git)
    print(f"{job}: writing to {run_dir} (commit {git['commit'][:7]})")
    return run_dir, config


@register("data_pusht")
def data_pusht() -> Path:
    """Phase 1: export lerobot/pusht (CPU is enough) via a separate Python 3.12 lerobot env."""
    from vla.data.lerobot_env import run_pusht_export
    from vla.data.pusht_compare import save_pusht_compare
    from vla.data.shards import read_meta

    run_dir, config = _start_data_run("data_pusht", {"requirements": "requirements-lerobot.txt"})
    venv = Path("/tmp/lerobot-venv") if is_kaggle() else Path("runs/debug/lerobot-venv").resolve()
    run_pusht_export(run_dir, venv_dir=venv)
    compare_png = run_dir / "pusht_compare.png"
    save_pusht_compare(run_dir, compare_png)
    log_dataset("data_pusht", run_dir, config, read_meta(run_dir), images=[compare_png])
    return run_dir


def _data_fetch(job: str, n_success: int) -> Path:
    from vla.data.collect_fetch import collect
    from vla.envs.fetch import renderer_name

    if not is_kaggle():
        print("WARNING: not on Kaggle, so these images are debug-only and will be refused for training.")
    run_dir, config = _start_data_run(job, {"n_success": n_success, "renderer": renderer_name()})
    meta = collect(run_dir, n_success=n_success, render=True)
    if is_kaggle() and not meta["images_for_training"]:
        raise RuntimeError(f"images were rendered with {meta['image_source']!r}, not EGL; check MUJOCO_GL")
    log_dataset(job, run_dir, config, meta)
    return run_dir


@register("data_fetch_smoke")
def data_fetch_smoke() -> Path:
    """Phase 1: 5 FetchPush demos with EGL images (checks rendering and the pipeline)."""
    return _data_fetch("data_fetch_smoke", n_success=5)


@register("data_fetch")
def data_fetch() -> Path:
    """Phase 1: 500 successful FetchPush demos with EGL images (the training set)."""
    return _data_fetch("data_fetch", n_success=500)


# --- Phase 2: feature cache ---------------------------------------------------


def find_datasets(root: Path) -> list[tuple[Path, dict]]:
    """Every Phase 1 dataset folder under root: has meta.json AND shard_*.npz, and is not a
    feature cache. Kaggle folder names contain a commit and timestamp, so we search instead
    of hardcoding them."""
    from vla.data.shards import read_meta

    found = []
    for meta_path in sorted(Path(root).rglob("meta.json")):
        folder = meta_path.parent
        if not any(folder.glob("shard_*.npz")):
            continue
        meta = read_meta(folder)
        if meta.get("kind") == "features" or "robot" not in meta:
            continue
        found.append((folder, meta))
    return found


def pick_datasets(found: list[tuple[Path, dict]], require_training_images: bool) -> dict:
    """One dataset per robot: the one with the most episodes (so data_fetch wins over
    data_fetch_smoke). With require_training_images, debug-only datasets are skipped."""
    picked = {}
    for robot in ("pusht", "fetch"):
        options = [(d, m) for d, m in found if m["robot"] == robot]
        if require_training_images:
            options = [(d, m) for d, m in options if m.get("images_for_training")]
        if not options:
            raise RuntimeError(
                f"no {robot} dataset{' with training-ok images' if require_training_images else ''} found. "
                f"Attach the {'data_pusht' if robot == 'pusht' else 'data_fetch'} notebook output with Add Input.")
        picked[robot] = max(options, key=lambda dm: dm[1].get("n_episodes", 0))[0]
    return picked


def _cache_features(job: str, max_episodes: int | None) -> Path:
    import torch
    import wandb

    from vla.data.cache_features import build_cache, print_size_report, print_stats, verify_cache
    from vla.models.encoders import DinoV2Encoder, MiniLMEncoder, encode_frames
    from vla.utils.runtime import inputs_dir

    root = inputs_dir()
    print(f"Searching {root} for datasets (folders with meta.json + shard_*.npz):")
    found = find_datasets(root)
    for folder, meta in found:
        print(f"  {meta['robot']:6s} {meta.get('n_episodes', '?'):>4} episodes  "
              f"training images: {bool(meta.get('images_for_training'))!s:5s}  {folder}")
    picked = pick_datasets(found, require_training_images=is_kaggle())
    for robot, folder in picked.items():
        print(f"Using {robot}: {folder}")
    if not is_kaggle():
        print("WARNING: not on Kaggle, so debug-only datasets are allowed; such caches stay debug-only.")

    device = get_device()
    dino = DinoV2Encoder().to(device)
    minilm = MiniLMEncoder().to(device)
    batch_size = 256 if device == "cuda" else 32
    encode = lambda frames: encode_frames(dino, frames, device, batch_size)  # noqa: E731
    encoder_info = {**dino.info(), **minilm.info()}

    run_dir, config = _start_data_run(job, {
        "max_episodes": max_episodes, "device": device, "torch": str(torch.__version__),
        "inputs": {robot: str(folder) for robot, folder in picked.items()}, **encoder_info,
    })
    t0 = time.time()
    summary = build_cache(run_dir, picked, encode, minilm.encode, max_episodes, encoder_info)
    seconds = time.time() - t0
    print("Normalization stats (train split):")
    print_stats(summary["stats"])
    print("Size report:")
    print_size_report(summary["size_report"], list(picked))

    print("Cached vs live features:")
    checks = {}
    for robot, folder in picked.items():
        checks[robot] = verify_cache(run_dir / robot, folder, encode)
        c = checks[robot]
        print(f"  {robot:6s} {c['n_checked']} frames | max |diff| {c['max_abs_diff']:.2e} "
              f"(largest feature {c['max_abs_feature']:.1f}) | images equal: {c['images_equal']} | ok: {c['ok']}")

    metrics = {"seconds": round(seconds, 1), "frames_per_second": round(summary["size_report"]["total_frames"] / seconds, 1),
               **{k: v for k, v in summary["size_report"].items() if not isinstance(v, dict)}}
    for robot in picked:
        metrics[f"{robot}_frames"] = summary["robots"][robot]["n_frames"]
        metrics[f"{robot}_images_for_training"] = summary["robots"][robot]["images_for_training"]
        metrics.update({f"{robot}_check_{k}": v for k, v in checks[robot].items()})
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    run = wandb.init(project=WANDB_PROJECT, job_type="cache", name=run_dir.name, config=config, dir=str(run_dir))
    run.summary.update(metrics)
    run.finish()

    if not all(c["ok"] for c in checks.values()):
        raise RuntimeError(f"cached features do not match live features: {checks}")
    print(f"Done in {seconds:.0f}s. Saved run folder: {run_dir}")
    return run_dir


@register("cache_features_smoke")
def cache_features_smoke() -> Path:
    """Phase 2: feature cache of the first 3 episodes per robot (quick interactive test)."""
    return _cache_features("cache_features_smoke", max_episodes=3)


@register("cache_features")
def cache_features() -> Path:
    """Phase 2: DINOv2 + MiniLM feature cache, raw images, stats.json for all episodes."""
    return _cache_features("cache_features", max_episodes=None)


# --- Phase 3: training --------------------------------------------------------


@register("smoke_train")
def smoke_train() -> Path:
    """Phase 3: FiLM for 1,000 steps on one GPU; prints steps/sec, peak GPU memory, data vs compute time."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # one GPU; must be set before CUDA starts
    import torch

    from vla.train import train
    from vla.utils.config import load_config

    print(f"GPUs visible: {torch.cuda.device_count()} "
          f"({', '.join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())) or 'none, CPU'})")
    cfg = load_config(["cond=film", "adapt=frozen", "train.steps=1000", "eval.enabled=false"])
    run_dir = train(cfg, run_prefix="smoke-")

    m = json.loads((run_dir / "metrics.json").read_text())
    wait = 100 * m["data_wait_fraction"]
    print("\n=== smoke_train summary ===")
    print(f"GPU:              {m['gpu'] or 'none (CPU)'}")
    print(f"Batch size:       {m['batch_size']} (half PushT, half FetchPush)")
    print(f"Trainable params: {m['trainable_params']:,}")
    default_steps = int(load_config().train.steps)
    print(f"Steps per second: {m['steps_per_sec']:.2f}  ->  {default_steps:,} steps (the default) take about "
          f"{default_steps / m['steps_per_sec'] / 60:.0f} min (training only, no evaluation)")
    print(f"Per step:         data wait {m['data_ms_per_step']:.1f} ms, compute {m['compute_ms_per_step']:.1f} ms")
    print(f"GPU waits for data {wait:.0f}% of the time"
          + ("  -> data loading is the bottleneck" if wait > 20 else "  -> fine, the GPU is kept busy"))
    if m["peak_gpu_mem_gb"] is not None:
        print(f"Peak GPU memory:  {m['peak_gpu_mem_gb']:.2f} GB allocated, {m['peak_gpu_mem_reserved_gb']:.2f} GB reserved")
    print(f"Loss:             train {m['train_loss']:.4f} | val {m['val_loss']:.4f} "
          f"(pusht {m['val_loss_pusht']:.4f}, fetch {m['val_loss_fetch']:.4f})")
    return run_dir


# --- Phase 4: evaluation and the first experiment batch -----------------------

BATCH_WORKERS_PER_RUN = 2  # 2 runs share Kaggle's 4 CPU cores (and the simulators in evaluation)
CONDITIONING_VARIANTS = ("token", "film", "sep_heads")
SEEDS = (0, 1, 2)


def conditioning_configs() -> list[list[str]]:
    """The 9 conditioning runs, seed-major: seed 0 of every variant runs first, so a batch cut
    short still has a complete comparison at seed 0."""
    return [[f"cond={c}", f"seed={s}"] for s in SEEDS for c in CONDITIONING_VARIANTS]


def eval_sanity_film_overrides() -> list[str]:
    """film seed 0 exactly as batch_conditioning's queue builds it (same config, same run key),
    so the batch later skips it instead of training it twice. A test checks they match."""
    from run_queue import plan_jobs

    (job,) = plan_jobs([["cond=film", "seed=0"]], workers_per_run=BATCH_WORKERS_PER_RUN)
    return job["overrides"]


@register("eval_sanity")
def eval_sanity() -> Path:
    """Phase 4 gate: expert and random policy through the harness, then one full FiLM seed-0 run with evaluation."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # one GPU; must be set before CUDA starts
    import wandb

    from vla.data.collect_fetch import run_episode
    from vla.evaluate import ROBOTS, SEED_START, ExpertPolicy, RandomPolicy, evaluate_policy, flat_metrics, print_summary
    from vla.train import run_key, train
    from vla.utils.config import load_config

    n = 50
    run_dir, config = _start_data_run("eval_sanity", {"n_episodes": n, "seed_start": SEED_START, "exec_steps": 8})

    print("\n(a) Scripted FetchPush expert through the harness (EGL videos of 3 episodes)")
    expert = evaluate_policy(ExpertPolicy(), ("fetch",), n, SEED_START, 8, n_videos=3,
                             video_dir=run_dir / "videos", video_tag="_expert")
    print_summary(expert, "expert policy")
    plain = [run_episode(SEED_START + i, render=False, noise_std=0.1)["success"] for i in range(n)]
    harness = [e["success"] for e in expert["fetch"]["episodes"]]
    expert_matches = harness == plain
    print(f"Plain expert loop on the same seeds: {sum(plain)}/{n} = {sum(plain) / n:.0%} | "
          f"harness gives identical per-episode results: {expert_matches}")

    print("\n(b) Random policy, both robots (should be about 0%)")
    random_result = evaluate_policy(RandomPolicy(), ROBOTS, n, SEED_START, 8)
    print_summary(random_result, "random policy")

    metrics = {"expert_fetch_success": expert["fetch"]["success_rate"], "expert_plain_loop_success": sum(plain) / n,
               "expert_matches_plain_loop": expert_matches,
               **{f"random_{k}": v for k, v in flat_metrics(random_result).items()},
               "expert": expert, "random": random_result}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    run = wandb.init(project=WANDB_PROJECT, job_type="eval_sanity", name=run_dir.name, config=config, dir=str(run_dir))
    run.log({f"video/{Path(p).stem}": wandb.Video(p, format="mp4") for p in expert["fetch"]["videos"]})
    run.summary.update({k: v for k, v in metrics.items() if not isinstance(v, dict)})
    run.finish()

    print("\n(c) One full FiLM seed-0 run with evaluation (the same run batch_conditioning would do first)")
    cfg = load_config(eval_sanity_film_overrides())
    print(f"Run key {run_key(cfg)} | {cfg.train.steps} steps | data-loader workers {cfg.data.num_workers}")
    film_dir = train(cfg)
    m = json.loads((film_dir / "metrics.json").read_text())

    remaining = len(conditioning_configs()) - 1  # batch_conditioning skips this run
    print("\n=== eval_sanity summary ===")
    print(f"Expert (FetchPush):      {expert['fetch']['successes']}/{n} = {expert['fetch']['success_rate']:.0%} through "
          f"the harness; plain loop {sum(plain) / n:.0%}; identical: {expert_matches}")
    print(f"Random policy:           PushT {random_result['pusht']['success_rate']:.0%} "
          f"(mean max coverage {random_result['pusht']['mean_max_coverage']:.3f}), "
          f"FetchPush {random_result['fetch']['success_rate']:.0%}")
    print(f"FiLM s0 ({m['steps']} steps):   PushT {m['pusht_success']:.0%} "
          f"[{m['pusht_wilson_low']:.0%}, {m['pusht_wilson_high']:.0%}], max coverage {m['pusht_mean_max_coverage']:.3f} | "
          f"FetchPush {m['fetch_success']:.0%} [{m['fetch_wilson_low']:.0%}, {m['fetch_wilson_high']:.0%}] | "
          f"mean {m['mean_success']:.0%}")
    print(f"Val loss:                {m['val_loss']:.4f} (pusht {m['val_loss_pusht']:.4f}, fetch {m['val_loss_fetch']:.4f})")
    print(f"Time per run (1 GPU):    train {m['train_minutes']:.1f} min ({m['steps_per_sec']:.2f} steps/s, "
          f"GPU waits for data {100 * m['data_wait_fraction']:.0f}%) + evaluation {m['eval_minutes']:.1f} min "
          f"(PushT {m['pusht_eval_seconds'] / 60:.1f}, FetchPush {m['fetch_eval_seconds'] / 60:.1f}) "
          f"= {m['total_minutes']:.1f} min")
    print(f"batch_conditioning:      {remaining} runs left, 2 at a time -> about "
          f"{-(-remaining // 2) * m['total_minutes'] / 60:.1f} h if two parallel runs do not slow each other down")
    if not expert_matches:
        raise RuntimeError("the harness and the plain expert loop disagree: the harness has a bug")
    return run_dir


@register("batch_conditioning")
def batch_conditioning() -> Path:
    """Phase 4: token, film, sep_heads x seeds 0, 1, 2 (4,000 steps + evaluation), 2 at a time; skips finished runs."""
    from run_queue import run_queue

    rows = run_queue(conditioning_configs(), workers_per_run=BATCH_WORKERS_PER_RUN)
    failed = [r["key"] or r["label"] for r in rows if r["status"] in ("failed", "timeout")]
    if failed:
        raise RuntimeError(f"{len(failed)} runs failed: {failed}. Run a new version to retry only those.")
    return outputs_dir()


# --- Phase 5: baselines ------------------------------------------------------

BASELINES = ("bc_mlp", "resnet18")
BASELINE_ROBOTS = ("pusht", "fetch")
SMOKE_BASELINE_STEPS = 200


def baseline_configs() -> list[list[str]]:
    """The 12 baseline runs (2 models x 2 robots x 3 seeds), seed-major like conditioning_configs."""
    return [[f"baseline={b}", f"robot={r}", f"seed={s}"] for s in SEEDS for b in BASELINES for r in BASELINE_ROBOTS]


def _prefetch_resnet_weights() -> None:
    """Download the ImageNet ResNet-18 weights once, before two runs start at the same time
    (otherwise both would write the same file in the torch cache at once)."""
    from torchvision.models import ResNet18_Weights, resnet18

    resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    print("ResNet-18 ImageNet weights are in the torch cache.")


@register("smoke_baselines")
def smoke_baselines() -> Path:
    """Phase 5: 200 steps of each baseline on each robot + 2 rollouts; prints steps/s and peak GPU memory."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # one GPU; must be set before CUDA starts
    import torch

    from vla.train import run_key, train
    from vla.utils.config import load_config

    print(f"GPUs visible: {torch.cuda.device_count()} "
          f"({', '.join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())) or 'none, CPU'})")
    _prefetch_resnet_weights()
    rows = []
    for name in BASELINES:
        for robot in BASELINE_ROBOTS:
            # 2 data-loader workers, as each run gets in batch_baselines (2 runs share 4 CPU cores)
            cfg = load_config([f"baseline={name}", f"robot={robot}", f"train.steps={SMOKE_BASELINE_STEPS}",
                               "train.val_every=100", "eval.n_episodes=2", "eval.n_videos=0",
                               f"data.num_workers={BATCH_WORKERS_PER_RUN}"])
            print(f"\n=== {run_key(cfg)} ===")
            run_dir = train(cfg, run_prefix="smoke-")
            m = json.loads((run_dir / "metrics.json").read_text())
            rows.append({"name": name, "robot": robot, "run_dir": str(run_dir), **m})

    full = load_config([f"baseline={BASELINES[0]}", f"robot={BASELINE_ROBOTS[0]}"])
    steps, episodes = int(full.train.steps), int(full.eval.n_episodes)
    print("\n=== smoke_baselines summary (1 T4, batch 128, 2 data-loader workers) ===")
    print(f"{'model':9s} {'robot':6s} {'params':>11s} {'steps/s':>8s} {'data wait':>9s} {'compute':>9s} "
          f"{'peak GPU':>14s} {'val L1':>7s} {'success':>8s} {'eval/ep':>8s} {'est. full run':>14s}")
    total = 0.0
    for r in rows:
        success = r["eval"][r["robot"]]  # the per-robot result: successes, n, ...
        eval_per_episode = r["eval_minutes"] / success["n"]
        # eval_minutes of the smoke run includes start-up, so this estimate errs on the long side
        est = steps / r["steps_per_sec"] / 60 + episodes * eval_per_episode
        total += est
        mem = (f"{r['peak_gpu_mem_gb']:.2f}/{r['peak_gpu_mem_reserved_gb']:.2f} GB"
               if r["peak_gpu_mem_gb"] is not None else "-")
        print(f"{r['name']:9s} {r['robot']:6s} {r['trainable_params']:>11,d} {r['steps_per_sec']:>8.2f} "
              f"{100 * r['data_wait_fraction']:>8.0f}% {r['compute_ms_per_step']:>7.1f}ms {mem:>14s} "
              f"{r['val_loss']:>7.4f} {success['successes']:>4d}/{success['n']:<3d} {60 * eval_per_episode:>7.1f}s "
              f"{est:>10.1f} min")
    print(f"Peak GPU memory is allocated/reserved. Full run = {steps:,} steps + {episodes} episodes.")
    print(f"batch_baselines: {len(baseline_configs())} runs = 3 seeds of the 4 rows above, 2 at a time -> about "
          f"{3 * total / 2 / 60:.1f} h if two parallel runs do not slow each other down")
    slow = [r for r in rows if r["data_wait_fraction"] > 0.2]
    for r in slow:
        print(f"NOTE: {r['name']} on {r['robot']} waits for data {100 * r['data_wait_fraction']:.0f}% of the time "
              f"(data loading is its bottleneck).")
    return outputs_dir()


@register("batch_baselines")
def batch_baselines() -> Path:
    """Phase 5: bc_mlp, resnet18 x pusht, fetch x seeds 0, 1, 2 (4,000 steps + evaluation), 2 at a time; skips finished runs."""
    from run_queue import run_queue

    _prefetch_resnet_weights()
    rows = run_queue(baseline_configs(), workers_per_run=BATCH_WORKERS_PER_RUN)
    failed = [r["key"] or r["label"] for r in rows if r["status"] in ("failed", "timeout")]
    if failed:
        raise RuntimeError(f"{len(failed)} runs failed: {failed}. Run a new version to retry only those.")
    return outputs_dir()


# --- Phase 6: adaptation regimes ------------------------------------------------

ADAPT_COND = "sep_heads"  # chosen from batch_conditioning by the pre-set rule (best mean success)
ADAPT_MODES = ("full_ft", "lora16", "frozen_matched")
SMOKE_ADAPT_STEPS = 200
RUN_BUDGET_MIN = 40  # train + validation + evaluation per run, leaving a margin under the 45-minute target


def adaptation_configs(steps: int) -> list[list[str]]:
    """The 9 adaptation runs (3 modes x 3 seeds) on sep_heads, seed-major, all with the same step count.
    frozen_matched has the adaptation runs' batch size (64), so only "unfreezing" differs."""
    warmup = int(load_config().optim.warmup_steps)
    if int(steps) <= warmup:
        raise ValueError(f"--steps {steps} must be larger than the {warmup} warmup steps")
    return [[f"cond={ADAPT_COND}", f"adapt={a}", f"seed={s}", f"train.steps={int(steps)}"]
            for s in SEEDS for a in ADAPT_MODES]


def max_steps_within(budget_s: float, sec_per_step: float, val_seconds_each: float, val_every: int,
                     fixed_s: float, granularity: int = 100) -> int:
    """Largest step count (a multiple of `granularity`) whose run fits in budget_s seconds:
    steps x sec_per_step + (number of validations) x val_seconds_each + fixed_s (start-up and evaluation).
    Training validates every val_every steps and at the last step: ceil(steps / val_every) times. 0 if none fits."""
    best, n = 0, granularity
    while n <= 1_000_000:
        if n * sec_per_step + math.ceil(n / val_every) * val_seconds_each + fixed_s > budget_s:
            break
        best, n = n, n + granularity
    return best


def _prefetch_dino() -> None:
    """Download DINOv2-small once, before two runs start at the same time."""
    from transformers import AutoModel

    from vla.models.encoders import DINO_NAME

    AutoModel.from_pretrained(DINO_NAME)
    print(f"{DINO_NAME} is in the Hugging Face cache.")


@register("smoke_adapt")
def smoke_adapt() -> Path:
    """Phase 6: 200 steps of full_ft and lora16 (batch 64, 2 workers); prints s/step, data wait, peak memory, max steps for 40 min."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # one GPU; must be set before CUDA starts
    import accelerate
    import peft
    import torch
    import transformers

    from vla.evaluate import ROBOTS
    from vla.train import run_key, train

    print(f"torch {torch.__version__} | transformers {transformers.__version__} | peft {peft.__version__} | "
          f"accelerate {accelerate.__version__}")
    print(f"GPUs visible: {torch.cuda.device_count()} "
          f"({', '.join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())) or 'none, CPU'})")
    _prefetch_dino()
    summary_dir, config = _start_data_run("smoke_adapt", {"steps": SMOKE_ADAPT_STEPS, "modes": ["full_ft", "lora16"],
                                                          "budget_min": RUN_BUDGET_MIN})
    full = load_config([f"cond={ADAPT_COND}", "adapt=full_ft"])
    episodes, val_every = int(full.eval.n_episodes), int(full.train.val_every)
    rows = []
    for mode in ("full_ft", "lora16"):
        # 2 data-loader workers, as each run gets in batch_adaptation (2 runs share 4 CPU cores);
        # one validation (at step 200) and 2 rollouts per robot, to time both
        cfg = load_config([f"cond={ADAPT_COND}", f"adapt={mode}", f"train.steps={SMOKE_ADAPT_STEPS}",
                           f"train.val_every={SMOKE_ADAPT_STEPS}", "eval.n_episodes=2", "eval.n_videos=0",
                           f"data.num_workers={BATCH_WORKERS_PER_RUN}"])
        print(f"\n=== {run_key(cfg)} ===")
        t0 = time.perf_counter()
        run_dir = train(cfg, run_prefix="smoke-")
        job_seconds = time.perf_counter() - t0  # the whole run: start-up, training, validation, evaluation
        m = json.loads((run_dir / "metrics.json").read_text())
        sec_per_step = 1 / m["steps_per_sec"]
        timed_steps = SMOKE_ADAPT_STEPS - min(10, SMOKE_ADAPT_STEPS // 10)  # train() skips the first 10 in timing
        val_each = m["val_seconds"] / max(1, m["n_validations"])
        eval_per_episode = {r: m["eval"][r]["seconds"] / m["eval"][r]["n"] for r in ROBOTS}
        eval_full = episodes * sum(eval_per_episode.values())
        # everything that is not a timed step, validation or evaluation: start-up, model loading, warm-up steps
        startup = job_seconds - timed_steps * sec_per_step - m["val_seconds"] - m["eval_minutes"] * 60
        fixed = startup + eval_full
        rows.append({"mode": mode, "run_dir": str(run_dir), "sec_per_step": sec_per_step, "val_each": val_each,
                     "eval_per_episode": eval_per_episode, "eval_full_s": eval_full, "startup_s": startup,
                     "max_steps": max_steps_within(RUN_BUDGET_MIN * 60, sec_per_step, val_each, val_every, fixed),
                     **{k: m[k] for k in ("encoder_trainable_params", "head_trainable_params", "trainable_params",
                                          "data_ms_per_step", "compute_ms_per_step", "data_wait_fraction",
                                          "peak_gpu_mem_gb_at_200", "peak_gpu_mem_reserved_gb", "val_loss")}})

    b, workers = int(cfg.data.batch_size), int(cfg.data.num_workers)
    print(f"\n=== smoke_adapt summary (1 {'GPU' if torch.cuda.is_available() else 'CPU'}, sep_heads, batch {b} = "
          f"{b // 2} PushT + {b - b // 2} FetchPush, {workers} data-loader workers, "
          f"fp16 autocast: {bool(cfg.train.amp) and torch.cuda.is_available()}) ===")
    print(f"{'mode':8s} {'DINOv2 trainable':>17s} {'rest':>11s} {'s/step':>7s} {'data wait':>15s} {'compute':>9s} "
          f"{'peak GPU @200':>14s} {'reserved':>9s} {'val':>6s} {'eval/ep PushT':>13s} {'Fetch':>6s} {'max steps':>9s}")
    for r in rows:
        mem = f"{r['peak_gpu_mem_gb_at_200']:.2f} GB" if r["peak_gpu_mem_gb_at_200"] is not None else "-"
        reserved = f"{r['peak_gpu_mem_reserved_gb']:.2f} GB" if r["peak_gpu_mem_reserved_gb"] is not None else "-"
        print(f"{r['mode']:8s} {r['encoder_trainable_params']:>17,d} {r['head_trainable_params']:>11,d} "
              f"{r['sec_per_step']:>7.3f} {r['data_ms_per_step']:>7.1f} ms {100 * r['data_wait_fraction']:>3.0f}% "
              f"{r['compute_ms_per_step']:>6.1f} ms {mem:>14s} {reserved:>9s} {r['val_each']:>5.0f}s "
              f"{r['eval_per_episode']['pusht']:>12.1f}s {r['eval_per_episode']['fetch']:>5.1f}s {r['max_steps']:>9,d}")
    for r in rows:
        print(f"{r['mode']}: {r['max_steps']:,} steps x {r['sec_per_step']:.3f} s + validation "
              f"({r['val_each']:.0f} s every {val_every:,} steps) + evaluation {r['eval_full_s'] / 60:.1f} min "
              f"({episodes} + {episodes} episodes) + start-up {r['startup_s'] / 60:.1f} min <= {RUN_BUDGET_MIN} min")
    mems = {r["mode"]: r["peak_gpu_mem_gb_at_200"] for r in rows}
    if None not in mems.values():
        print(f"LoRA peak memory vs full fine-tuning: {mems['lora16']:.2f} vs {mems['full_ft']:.2f} GB "
              f"({100 * (1 - mems['lora16'] / mems['full_ft']):.0f}% lower)")
    steps = min(r["max_steps"] for r in rows)
    print("PushT evaluation time per episode is from an untrained model (episodes run the full 300 steps), "
          "so the estimate errs on the long side. Two runs at once share the 4 CPU cores; if data wait is high, "
          "leave a margin.")
    print(f"Use the smaller value for all 9 runs (the same budget for every mode):")
    print(f"    python scripts/kaggle_job.py --job batch_adaptation --steps {steps}")
    print(f"batch_adaptation: 9 runs, 2 at a time -> at most about {math.ceil(9 / 2) * RUN_BUDGET_MIN / 60:.1f} h "
          f"(frozen_matched runs are faster)")
    (summary_dir / "metrics.json").write_text(json.dumps({"recommended_steps": steps, "rows": rows}, indent=2))
    return summary_dir


@register("batch_adaptation")
def batch_adaptation(steps: int | None = None) -> Path:
    """Phase 6: full_ft, lora16, frozen_matched x seeds 0, 1, 2 on sep_heads (--steps N + evaluation), 2 at a time; skips finished runs."""
    if steps is None:
        raise ValueError("batch_adaptation needs --steps N (take N from the smoke_adapt summary)")
    from run_queue import run_queue

    configs = adaptation_configs(steps)
    _prefetch_dino()
    rows = run_queue(configs, workers_per_run=BATCH_WORKERS_PER_RUN)
    failed = [r["key"] or r["label"] for r in rows if r["status"] in ("failed", "timeout")]
    if failed:
        raise RuntimeError(f"{len(failed)} runs failed: {failed}. Run a new version to retry only those.")
    return outputs_dir()


# --- command line -----------------------------------------------------------


def run_job(name: str, steps: int | None = None):
    """Run JOBS[name]. --steps goes only to jobs that take a step count (batch_adaptation)."""
    fn = JOBS[name]
    takes_steps = "steps" in inspect.signature(fn).parameters
    if steps is not None and not takes_steps:
        raise ValueError(f"job {name!r} does not take --steps")
    return fn(steps=steps) if takes_steps else fn()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", choices=sorted(JOBS), help="name of the job to run")
    parser.add_argument("--steps", type=int, help="training steps per run (batch_adaptation; from smoke_adapt)")
    parser.add_argument("--list", action="store_true", help="list the available jobs")
    args = parser.parse_args()

    if args.list or not args.job:
        for name, fn in sorted(JOBS.items()):
            print(f"{name:22s} {(fn.__doc__ or '').strip().splitlines()[0]}")
        return

    load_dotenv_if_local()
    check_wandb_key()
    run_job(args.job, args.steps)


if __name__ == "__main__":
    main()
