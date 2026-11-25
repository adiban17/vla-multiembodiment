"""Train the VLA policy by behavior cloning (spec "Training recipe").

    python -m vla.train cond=film seed=1                      # full run
    python -m vla.train size=tiny train.steps=20 --allow-dirty   # Mac CPU check
    python -m vla.train baseline=resnet18 robot=fetch seed=1  # a per-robot baseline (Phase 5)

What one run does:
1. Refuses to start with uncommitted code (every number must trace to a commit), unless
   --allow-dirty, which also saves the diff into the run folder.
2. Seeds everything, finds the feature cache (searching /kaggle/input on Kaggle), and refuses
   caches whose images are debug-only.
3. Uses the cache's own episode split and normalization (stats.json); nothing is recomputed.
   Data is read from read-only memory-mapped files; nothing is copied.
4. Every batch is 50% PushT, 50% FetchPush. The loss weights the two robots equally.
   A baseline (baseline=bc_mlp|resnet18, robot=pusht|fetch) trains on its one robot instead,
   reading raw uint8 frames (data.mode=pixels); everything else below is the same.
   An adaptation run (adapt=full_ft|lora16, Phase 6) also reads raw uint8 frames and runs its own
   trainable DINOv2 inside every step; adapt=frozen_matched is the frozen control with the same
   batch size (64) and step count.
5. AdamW, 500 warmup steps then cosine decay to 0, fp16 autocast + GradScaler on CUDA,
   gradient clipping at 1.0. DINOv2 gets its own learning rate (encoder.lr). Logs to W&B; validation
   loss every 1,000 steps; peak GPU memory after 200 steps; seconds per step and data-wait time.
6. Writes a run folder: config.yaml, git.txt, metrics.json, ckpt.pt, plus copies of the cache's
   stats.json and language.npy (so the run folder alone is enough to evaluate it).
7. Evaluates the final model with closed-loop rollouts (vla.evaluate; 50 episodes per robot) and
   adds the success rates to metrics.json and W&B. ckpt.pt is saved first, so a crash during
   evaluation never loses the training.
"""

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from vla.data.cache_features import find_feature_caches, pick_feature_cache, print_feature_caches
from vla.data.dataset import MixedBatchSampler, WindowDataset, collate_mixed_pixels, split_episodes
from vla.data.normalize import Normalizer
from vla.models.policy import ROBOT_NAMES, build_model, count_params, encoder_mode, encoder_params, is_baseline
from vla.utils.config import load_config, to_dict
from vla.utils.runtime import get_device, inputs_dir, is_kaggle, load_dotenv_if_local, outputs_dir
from vla.utils.seed import seed_everything, worker_init_fn
from vla.utils.tracking import check_wandb_key, dirty_files, git_commit, save_run_info

SPLIT_SEED = 0  # the cache's episode split (vla.data.cache_features.write_stats)
PEAK_MEMORY_STEP = 200  # spec: read torch.cuda.max_memory_allocated() after 200 steps


# --- optimizer and schedule ---------------------------------------------------------


def make_optimizer(model: nn.Module, optim_cfg, encoder_lr: float | None = None) -> torch.optim.AdamW:
    """AdamW. Weight decay only on Linear and Conv2d weight matrices (ResNet-18, DINOv2's patch
    embedding); biases, LayerNorms, BatchNorms, embeddings and the learned query/position vectors are
    not decayed (standard practice).

    The model's own trainable DINOv2 (Phase 6: all of it, or only its LoRA weights) gets encoder_lr;
    everything else gets optim_cfg.lr. Head groups come first, so a model without an encoder gets
    exactly the 2 groups it always had. The schedule (make_scheduler) scales every group alike."""
    linear_weights = {id(m.weight) for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))}
    encoder = getattr(model, "encoder", None)
    encoder_ids = {id(p) for p in encoder.parameters()} if encoder is not None else set()
    decay, no_decay, enc_decay, enc_no_decay = [], [], [], []
    for p in model.parameters():
        if p.requires_grad:
            in_encoder, decayed = id(p) in encoder_ids, id(p) in linear_weights
            if in_encoder:
                (enc_decay if decayed else enc_no_decay).append(p)
            else:
                (decay if decayed else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": optim_cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    if enc_decay or enc_no_decay:
        if encoder_lr is None:
            raise ValueError("the model trains DINOv2: set encoder.lr (configs/adapt/)")
        groups += [g for g in ({"params": enc_decay, "weight_decay": optim_cfg.weight_decay, "lr": float(encoder_lr)},
                               {"params": enc_no_decay, "weight_decay": 0.0, "lr": float(encoder_lr)}) if g["params"]]
    return torch.optim.AdamW(groups, lr=optim_cfg.lr)


def lr_factor(step: int, warmup: int, total: int) -> float:
    """Multiplier on the peak learning rate: linear warmup over `warmup` steps, then cosine to 0
    at step `total`."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1 + math.cos(math.pi * progress))


def make_scheduler(optimizer, warmup: int, total: int):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, warmup, total))


# --- data -------------------------------------------------------------------------


def resolve_cache(cfg) -> dict:
    """Find the feature cache: data.cache_dir if set, else search inputs_dir().
    Debug-only caches are refused on Kaggle always, and on the Mac unless data.allow_debug_images."""
    root = Path(cfg.data.cache_dir) if cfg.data.cache_dir else inputs_dir()
    allow_debug = bool(cfg.data.allow_debug_images) and not is_kaggle()
    print(f"Searching {root} for feature caches (stats.json + pusht/ + fetch/ with kind 'features'):")
    found = find_feature_caches(root)
    print_feature_caches(found)
    cache = pick_feature_cache(found, require_training_images=not allow_debug)
    print(f"Using cache: {cache['path']}")
    if not cache["training_ok"]:
        print("WARNING: this cache has debug-only images. This is a debug run, never a result.")
    return cache


def check_split(cache_dir: Path, stats: dict) -> dict:
    """The cache's train/val split must be the deterministic split_episodes(n, val_frac, seed 0)."""
    split = {}
    for robot in ROBOT_NAMES:
        s = stats["split"][robot]
        n = len(np.load(cache_dir / robot / "lowdim.npz")["episode_length"])
        train_eps, val_eps = split_episodes(n, s["val_frac"], SPLIT_SEED)
        if s["seed"] != SPLIT_SEED or s["train"] != train_eps or s["val"] != val_eps:
            raise RuntimeError(f"{robot}: the split in stats.json is not split_episodes({n}, {s['val_frac']}, "
                               f"seed {SPLIT_SEED}); rebuild the cache")
        split[robot] = s
    return split


def train_robots(cfg) -> tuple[str, ...]:
    """Robots a run trains on: both for the VLA, cfg.robot for a baseline."""
    return (cfg.robot,) if is_baseline(cfg) else ROBOT_NAMES


def build_datasets(cache_dir: Path, cfg, require_training_images: bool):
    """Train and val WindowDatasets for the run's robots, sharing one Normalizer from stats.json."""
    normalizer = Normalizer.load(cache_dir / "stats.json")
    split = check_split(cache_dir, normalizer.stats)
    robots = train_robots(cfg)
    kw = dict(mode=cfg.data.mode, normalizer=normalizer, require_training_images=require_training_images,
              seed=cfg.seed)
    train_sets = [WindowDataset(cache_dir / r, split="train", instructions="train", **kw) for r in robots]
    val_sets = [WindowDataset(cache_dir / r, split="val", instructions="train", **kw) for r in robots]
    return train_sets, val_sets, {r: split[r] for r in robots}


def uses_mixed_pixels(cfg) -> bool:
    """The VLA reading raw frames (adaptation runs): its batches mix 96x96 and 128x128 frames."""
    return not is_baseline(cfg) and cfg.data.mode == "pixels"


def _loader_kwargs(cfg, device: str, seed: int) -> dict:
    workers = int(cfg.data.num_workers)
    kw = dict(num_workers=workers, pin_memory=device == "cuda", worker_init_fn=worker_init_fn,
              generator=torch.Generator().manual_seed(seed))
    if workers > 0:
        kw["prefetch_factor"] = int(cfg.data.prefetch_factor)
    if uses_mixed_pixels(cfg):
        kw["collate_fn"] = collate_mixed_pixels
    return kw


def make_train_loader(train_sets, cfg, device: str) -> DataLoader:
    """Endless 50/50 batches (robot order = embodiment id: PushT, FetchPush). With one robot
    (baselines) every batch is that robot's, shuffled and reshuffled after each pass."""
    sampler = MixedBatchSampler([len(d) for d in train_sets], int(cfg.data.batch_size), seed=int(cfg.seed))
    return DataLoader(ConcatDataset(train_sets), batch_sampler=sampler,
                      persistent_workers=int(cfg.data.num_workers) > 0, **_loader_kwargs(cfg, device, cfg.seed))


def _to(value, device: str):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, list) and value and all(isinstance(v, torch.Tensor) for v in value):
        return [v.to(device, non_blocking=True) for v in value]  # pixels of mixed sizes
    return value


def to_device(batch: dict, device: str) -> dict:
    return {k: _to(v, device) for k, v in batch.items()}


@torch.no_grad()
def validate(model, val_sets, cfg, device: str, use_amp: bool) -> dict:
    """Masked L1 over ALL val windows, per robot; val loss = average of the robot means
    (the same equal-weight rule as the training loss). Same windows and instructions every time.
    Only the robots in val_sets are reported (a baseline has one)."""
    model.eval()
    sums = torch.zeros(len(ROBOT_NAMES), device=device)
    counts = torch.zeros(len(ROBOT_NAMES), device=device)
    for ds in val_sets:
        ds.reset_rng()
        loader = DataLoader(ds, batch_size=int(cfg.data.val_batch_size), shuffle=False,
                            **_loader_kwargs(cfg, device, cfg.seed))
        for batch in loader:
            batch = to_device(batch, device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                out = model.loss(batch)
            sums += out["sums"]
            counts += out["counts"]
    model.train()
    means = (sums / counts.clamp(min=1)).tolist()
    robots = [ds.robot for ds in val_sets]
    present = [m for name, m, c in zip(ROBOT_NAMES, means, counts.tolist()) if c > 0 and name in robots]
    return {"loss": sum(present) / len(present),
            **{f"loss_{name}": m for name, m in zip(ROBOT_NAMES, means) if name in robots}}


# --- run folder -------------------------------------------------------------------


def model_name(cfg) -> str:
    """What the run's W&B group and folder name start with: <cond>-<adapt> for the VLA
    (film-frozen), <baseline>-<robot> for a baseline (resnet18-fetch)."""
    return f"{cfg.baseline}-{cfg.robot}" if is_baseline(cfg) else f"{cfg.cond}-{cfg.adapt}"


def run_key(cfg) -> str:
    """What makes two runs "the same experiment", without the commit, e.g. film-frozen-full-s0-4000steps
    or resnet18-fetch-full-s1-4000steps. The run queue skips a config whose key already has a
    finished, evaluated run in W&B."""
    return f"{model_name(cfg)}-{cfg.size}-s{cfg.seed}-{int(cfg.train.steps)}steps"


def make_run_dir(cfg, commit: str, prefix: str = "") -> Path:
    """<outputs>/<prefix><cond>-<adapt>-s<seed>-<commit7>, e.g. film-frozen-s1-3f9a2c1
    (baselines: bc_mlp-pusht-s0-3f9a2c1). A timestamp (and a counter) is added if that folder exists."""
    name = f"{prefix}{model_name(cfg)}-s{cfg.seed}-{commit[:7]}"
    candidates = [name, f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"]
    candidates += [f"{candidates[1]}-{i}" for i in range(2, 100)]
    for candidate in candidates:
        path = outputs_dir() / candidate
        try:
            path.mkdir(parents=True)
            return path
        except FileExistsError:
            continue
    raise RuntimeError(f"could not create a run folder for {name}")


# --- the training loop -------------------------------------------------------------


def train(cfg, allow_dirty: bool = False, run_prefix: str = "") -> Path:
    """Run one training job described by cfg (from load_config). Returns the run folder."""
    baseline = is_baseline(cfg)
    if baseline and cfg.data.mode != "pixels":
        raise ValueError(f"baselines read raw frames: data.mode must be 'pixels', got {cfg.data.mode!r}")
    regime = "none" if baseline else encoder_mode(cfg)
    if not baseline:
        wanted = "cached" if regime == "frozen" else "pixels"
        if cfg.data.mode != wanted:
            raise ValueError(f"encoder.mode {regime} reads data.mode={wanted!r}, got {cfg.data.mode!r}")
    git = git_commit()
    if git["dirty"] and not allow_dirty:
        raise RuntimeError("the code has uncommitted changes:\n" + dirty_files() +
                           "\nCommit them first (every run must trace to a commit), or pass --allow-dirty to debug.")

    seed_everything(int(cfg.seed))
    device = get_device()
    use_amp = bool(cfg.train.amp) and device == "cuda"
    steps, log_every, val_every = int(cfg.train.steps), int(cfg.train.log_every), int(cfg.train.val_every)

    cache = resolve_cache(cfg)
    cache_dir = cache["path"]
    robots = train_robots(cfg)
    train_sets, val_sets, split = build_datasets(cache_dir, cfg, require_training_images=cache["training_ok"])
    loader = make_train_loader(train_sets, cfg, device)

    model = build_model(cfg).to(device)
    n_trainable, n_total = count_params(model), count_params(model, trainable_only=False)
    n_encoder = encoder_params(model)  # the model's own DINOv2 (0 when it reads cached features)
    n_head = n_trainable - n_encoder
    encoder_lr = cfg.encoder.lr if n_encoder else None
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else None

    run_dir = make_run_dir(cfg, git["commit"], run_prefix)
    for name in ("stats.json", "language.npy"):  # the run folder is self-contained for evaluation
        shutil.copy2(cache_dir / name, run_dir / name)
    run_info = {
        "run_id": run_dir.name, "key": run_key(cfg), "model": cfg.baseline if baseline else "vla",
        "robots": list(robots), "git_commit": git["commit"], "git_dirty": git["dirty"],
        "cache_dir": str(cache_dir), "debug_data": not cache["training_ok"], "device": device, "gpu": gpu,
        "amp": use_amp, "torch": str(torch.__version__), "on_kaggle": is_kaggle(),
        "trainable_params": n_trainable, "total_params": n_total, "encoder_mode": regime,
        "encoder_trainable_params": n_encoder, "head_trainable_params": n_head, "encoder_lr": encoder_lr,
        "train_windows": {r: len(d) for r, d in zip(robots, train_sets)},
        "val_windows": {r: len(d) for r, d in zip(robots, val_sets)},
    }
    config = {**to_dict(cfg), "run": run_info}
    save_run_info(run_dir, config, git)

    import wandb

    wb = wandb.init(project=cfg.wandb.project, name=run_dir.name, group=model_name(cfg),
                    job_type="train", config=config, dir=str(run_dir))
    print(f"Run {run_dir.name} | device {device}{f' ({gpu})' if gpu else ''} | fp16 autocast: {use_amp}")
    what = f"Baseline {cfg.baseline} on {cfg.robot}" if baseline else f"Variant {cfg.cond}, adapt {cfg.adapt}"
    mix = f"all {cfg.robot}" if baseline else "half PushT, half FetchPush"
    print(f"{what} | trainable parameters {n_trainable:,} | batch {cfg.data.batch_size} ({mix}) | {steps} steps")
    if not baseline:
        lr_text = f"lr {encoder_lr:g}" if encoder_lr is not None else "frozen, cached features"
        print(f"Trainable: DINOv2 {n_encoder:,} ({regime}, {lr_text}) + rest of the model {n_head:,} "
              f"(lr {cfg.optim.lr:g})")
    print(f"Train windows {run_info['train_windows']} | val windows {run_info['val_windows']}")

    optimizer = make_optimizer(model, cfg.optim, encoder_lr=encoder_lr)
    scheduler = make_scheduler(optimizer, int(cfg.optim.warmup_steps), steps)
    scaler = torch.amp.GradScaler(device, enabled=use_amp)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    skip = min(10, steps // 10)  # the first steps include worker start-up and CUDA warm-up: not timed
    timed = {"steps": 0, "data": 0.0, "compute": 0.0}
    window = {"steps": 0, "data": 0.0, "compute": 0.0, "loss": 0.0, **dict.fromkeys(robots, 0.0)}
    last_train, val_curve, bad_steps = {}, [], 0
    mem_step, peak_mem_at_step, val_seconds = min(PEAK_MEMORY_STEP, steps), None, 0.0
    wall_start = time.perf_counter()

    model.train()
    batches = iter(loader)
    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        batch = next(batches)                      # waiting for the data loader
        t1 = time.perf_counter()
        batch = to_device(batch, device)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            out = model.loss(batch)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(out["loss"]).backward()
        scaler.unscale_(optimizer)                 # so clipping sees the true gradient size
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.optim.grad_clip))
        scaler.step(optimizer)                     # skipped automatically if fp16 gradients overflowed
        scaler.update()
        scheduler.step()
        if device == "cuda":
            torch.cuda.synchronize()               # so the timer measures the GPU work, not just its launch
        t2 = time.perf_counter()

        if step == mem_step and device == "cuda":  # before any validation, which would add its own peak
            peak_mem_at_step = torch.cuda.max_memory_allocated() / 1e9
            print(f"step {step:6d} | peak GPU memory so far {peak_mem_at_step:.2f} GB allocated "
                  f"({torch.cuda.max_memory_reserved() / 1e9:.2f} GB reserved)")
            wb.log({"perf/peak_gpu_mem_gb_at_200": peak_mem_at_step}, step=step)

        loss = out["loss"].item()
        bad_steps = bad_steps + 1 if not math.isfinite(loss) else 0
        if bad_steps >= 10:
            raise RuntimeError(f"loss was not finite for 10 steps in a row (step {step}); stopping")
        if step > skip:
            timed["steps"] += 1
            timed["data"] += t1 - t0
            timed["compute"] += t2 - t1
        for key, value in (("steps", 1), ("data", t1 - t0), ("compute", t2 - t1), ("loss", loss),
                           *((r, out[f"loss_{r}"].item()) for r in robots)):
            window[key] += value

        if step % log_every == 0 or step == steps:
            n = window["steps"]
            seconds = window["data"] + window["compute"]
            last_train = {"loss": window["loss"] / n, **{f"loss_{r}": window[r] / n for r in robots}}
            log = {
                **{f"train/{k}": v for k, v in last_train.items()}, "train/lr": scheduler.get_last_lr()[0],
                "train/grad_norm": float(grad_norm), "train/amp_scale": scaler.get_scale() if use_amp else 1.0,
                "perf/steps_per_sec": n / seconds, "perf/data_ms": 1000 * window["data"] / n,
                "perf/compute_ms": 1000 * window["compute"] / n, "perf/data_wait_frac": window["data"] / seconds,
            }
            if device == "cuda":
                log["perf/gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            wb.log(log, step=step)
            per_robot = ", ".join(f"{r} {last_train[f'loss_{r}']:.4f}" for r in robots)
            print(f"step {step:6d} | loss {log['train/loss']:.4f} ({per_robot}) | lr {log['train/lr']:.2e} | "
                  f"{log['perf/steps_per_sec']:.2f} steps/s | data {log['perf/data_ms']:.1f} ms, "
                  f"compute {log['perf/compute_ms']:.1f} ms")
            window = dict.fromkeys(window, 0.0)
            window["steps"] = 0

        if step % val_every == 0 or step == steps:
            t_val = time.perf_counter()
            val = validate(model, val_sets, cfg, device, use_amp)
            val_seconds += time.perf_counter() - t_val
            val_curve.append({"step": step, **val})
            wb.log({f"val/{k}": v for k, v in val.items()}, step=step)
            per_robot = ", ".join(f"{r} {val[f'loss_{r}']:.4f}" for r in robots)
            print(f"step {step:6d} | VAL loss {val['loss']:.4f} ({per_robot})")

    del batches  # stops the data-loader workers
    wall = time.perf_counter() - wall_start
    timed_seconds = timed["data"] + timed["compute"]
    final_val = val_curve[-1]
    metrics = {
        "run_id": run_dir.name, "run_key": run_info["key"], "model": run_info["model"],
        "robot": cfg.robot, "cond": cfg.cond, "adapt": cfg.adapt, "seed": int(cfg.seed), "steps": steps,
        "train_loss": last_train["loss"], **{f"train_loss_{r}": last_train[f"loss_{r}"] for r in robots},
        "val_loss": final_val["loss"], **{f"val_loss_{r}": final_val[f"loss_{r}"] for r in robots},
        "val_curve": val_curve,
        "steps_per_sec": timed["steps"] / timed_seconds if timed["steps"] else float("nan"),
        "data_ms_per_step": 1000 * timed["data"] / max(1, timed["steps"]),
        "compute_ms_per_step": 1000 * timed["compute"] / max(1, timed["steps"]),
        "data_wait_fraction": timed["data"] / timed_seconds if timed_seconds else float("nan"),
        "wall_minutes": wall / 60, "train_minutes": wall / 60,
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None,
        "peak_gpu_mem_reserved_gb": torch.cuda.max_memory_reserved() / 1e9 if device == "cuda" else None,
        "peak_gpu_mem_gb_at_200": peak_mem_at_step, "peak_mem_step": mem_step,
        "val_seconds": val_seconds, "n_validations": len(val_curve),
        "trainable_params": n_trainable, "total_params": n_total, "encoder_mode": regime,
        "encoder_trainable_params": n_encoder, "head_trainable_params": n_head, "encoder_lr": encoder_lr,
        "batch_size": int(cfg.data.batch_size), "device": device, "gpu": gpu, "train_windows": run_info["train_windows"],
        "debug_data": run_info["debug_data"], "cache_dir": str(cache_dir), "split": split,
        "git_commit": git["commit"],
    }
    if cfg.train.save_checkpoint:
        torch.save({"model": model.state_dict(), "config": config, "step": steps}, run_dir / "ckpt.pt")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))  # training results are safe now
    print(f"Training done in {wall / 60:.1f} min.")

    if cfg.eval.enabled:
        from vla.evaluate import evaluate_baseline_run, evaluate_run, flat_metrics

        t0 = time.perf_counter()
        if baseline:  # its own robot only; no DINOv2 or MiniLM
            result = evaluate_baseline_run(model, run_dir, cfg, device, wb=wb)
        else:
            result = evaluate_run(model, run_dir, cfg, device, wb=wb, cache_dir=cache_dir)
        eval_minutes = (time.perf_counter() - t0) / 60  # includes loading DINOv2 and MiniLM
        metrics.update(flat_metrics(result))
        metrics.update({"eval_minutes": eval_minutes, "eval": result})
    metrics["total_minutes"] = (time.perf_counter() - wall_start) / 60
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    wb.summary.update({k: v for k, v in metrics.items() if not isinstance(v, (dict, list))})
    wb.finish()
    print(f"Done in {metrics['total_minutes']:.1f} min. Saved run folder: {run_dir}")
    return run_dir


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("overrides", nargs="*", help="cond=film, size=tiny, seed=1, train.steps=1000, ...")
    parser.add_argument("--allow-dirty", action="store_true", help="run with uncommitted changes (debug only)")
    parser.add_argument("--result-file", help="write {\"run_dir\": ...} here when done (used by scripts/run_queue.py)")
    args = parser.parse_args(argv)
    load_dotenv_if_local()
    check_wandb_key()
    run_dir = train(load_config(args.overrides), allow_dirty=args.allow_dirty)
    if args.result_file:
        Path(args.result_file).write_text(json.dumps({"run_dir": str(run_dir)}))


if __name__ == "__main__":
    main()
