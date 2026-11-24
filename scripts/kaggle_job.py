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

from vla.utils.runtime import get_device, is_kaggle, load_dotenv_if_local, outputs_dir
from vla.utils.tracking import git_commit, save_run_info

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


# --- command line -----------------------------------------------------------


def check_wandb_key() -> None:
    """Fail early with a clear message instead of W&B asking for a key mid-run."""
    if os.environ.get("WANDB_MODE") in ("offline", "disabled") or os.environ.get("WANDB_API_KEY"):
        return
    if is_kaggle():
        sys.exit("WANDB_API_KEY is not set. Add it in Kaggle (Add-ons > Secrets) and re-run cell 1.")
    print("No WANDB_API_KEY in .env, so W&B logs offline (files only).")
    os.environ["WANDB_MODE"] = "offline"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", choices=sorted(JOBS), help="name of the job to run")
    parser.add_argument("--list", action="store_true", help="list the available jobs")
    args = parser.parse_args()

    if args.list or not args.job:
        for name, fn in sorted(JOBS.items()):
            print(f"{name:12s} {(fn.__doc__ or '').strip().splitlines()[0]}")
        return

    load_dotenv_if_local()
    check_wandb_key()
    JOBS[args.job]()


if __name__ == "__main__":
    main()
