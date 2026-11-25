"""The one entry point for all work run on Kaggle (and locally on CPU).

Kaggle notebook cell 2 runs:   python scripts/kaggle_job.py --job <name>
List the jobs:                  python scripts/kaggle_job.py --list

To add work, write a function and decorate it with @register("name"). Each job returns
its run folder. Every job must save its resolved config and git commit (save_run_info).
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import run_queue` works however we are loaded

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


# --- command line -----------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", choices=sorted(JOBS), help="name of the job to run")
    parser.add_argument("--list", action="store_true", help="list the available jobs")
    args = parser.parse_args()

    if args.list or not args.job:
        for name, fn in sorted(JOBS.items()):
            print(f"{name:22s} {(fn.__doc__ or '').strip().splitlines()[0]}")
        return

    load_dotenv_if_local()
    check_wandb_key()
    JOBS[args.job]()


if __name__ == "__main__":
    main()
