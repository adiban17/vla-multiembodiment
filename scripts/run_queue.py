"""Run a list of training configs, two at a time (one per GPU), and print a results table.

    python scripts/run_queue.py "cond=token seed=0" "cond=film seed=0" "cond=sep_heads seed=0"

What it does:
- Each config runs as its own process, `python -m vla.train <config>`, on its own GPU
  (CUDA_VISIBLE_DEVICES=0 or 1). When a run finishes, that GPU takes the next config.
- **Skips runs already done.** A new Kaggle version starts with an empty output folder, so "done"
  is looked up in W&B: a finished, evaluated, non-debug run with the same run key
  (vla.train.run_key, e.g. film-frozen-full-s0-4000steps; the commit is not part of it).
  --rerun turns this off. If W&B cannot be reached, it warns and runs everything.
- **Keeps going if a run crashes** (or exceeds --timeout-min): that row is marked failed and
  the next config starts. Each run's output is streamed with a [gpu0 film-s1] prefix and also
  saved to <queue folder>/logs/<key>.log.
- **Data-loader workers.** Kaggle has 4 CPU cores, shared by both runs and the simulators used
  in evaluation, so with 2 runs at once each run gets 2 data-loader workers instead of 4
  (--workers-per-run; it becomes the override data.num_workers=N and is saved in the run's config).
- Prints a final table, plus mean +- std over seeds and a pooled 95% Wilson interval per
  variant, and saves queue_results.json. Exit code 1 if any run failed.
"""

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from vla.evaluate import wilson_interval
from vla.train import run_key
from vla.utils.config import load_config
from vla.utils.runtime import REPO_ROOT, outputs_dir
from vla.utils.tracking import git_commit, save_run_info

WANDB_PROJECT = "vla-multiembodiment"
TABLE_KEYS = ("steps_per_sec", "pusht_success", "fetch_success", "mean_success", "val_loss", "total_minutes",
              "pusht_successes", "pusht_n", "fetch_successes", "fetch_n")


def default_workers_per_run(slots: int):
    """2 data-loader workers per run when 2+ runs share Kaggle's 4 CPU cores; else the config's own value."""
    return 2 if slots >= 2 else None


def plan_jobs(configs, workers_per_run=None, extra=()) -> list[dict]:
    """Turn config override lists into jobs with their run key. A config that does not load
    (a typo such as cond=flim) gets key None and an error, instead of stopping the whole queue."""
    jobs = []
    for i, config in enumerate(configs):
        overrides = [*config, *extra]
        if workers_per_run is not None:
            overrides.append(f"data.num_workers={workers_per_run}")
        try:
            key, error = run_key(load_config(overrides)), None
        except Exception as e:  # noqa: BLE001 - reported in the table
            key, error = None, f"{type(e).__name__}: {e}"
        jobs.append({"index": i, "key": key, "label": key or " ".join(config), "overrides": overrides,
                     "error": error})
    return jobs


def wandb_finished_lookup(project: str = WANDB_PROJECT):
    """key -> info about a finished, evaluated, non-debug W&B run with that key, or None."""
    if os.environ.get("WANDB_MODE") in ("disabled", "offline"):
        print(f"W&B is {os.environ['WANDB_MODE']}: cannot check for finished runs, so every config runs.")
        return lambda key: None
    try:
        import wandb

        runs = list(wandb.Api(timeout=60).runs(project, filters={"state": "finished"}))
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not list W&B runs ({type(e).__name__}: {e}); every config runs.")
        return lambda key: None
    index = {}
    for r in runs:
        info = (r.config or {}).get("run") or {}
        key, summary = info.get("key"), r.summary
        if not key or info.get("debug_data") or summary.get("eval/mean_success") is None:
            continue
        found = {"run": r.name, "commit": str(info.get("git_commit", ""))[:7], "url": r.url,
                 "steps_per_sec": summary.get("steps_per_sec"), "val_loss": summary.get("val_loss"),
                 "total_minutes": summary.get("total_minutes")}
        for k in ("pusht_success", "fetch_success", "mean_success", "pusht_successes", "pusht_n",
                  "fetch_successes", "fetch_n"):
            found[k] = summary.get(f"eval/{k}")
        index.setdefault(key, found)  # newest first
    print(f"W&B: {len(index)} finished, evaluated run keys in project {project}.")
    return index.get


def _train_command(allow_dirty: bool):
    def command_for(job, result_file, slot):
        cmd = [sys.executable, "-m", "vla.train", *job["overrides"], "--result-file", str(result_file)]
        return cmd + (["--allow-dirty"] if allow_dirty else [])

    return command_for


def _short(key: str) -> str:
    return key.replace("-frozen", "").replace("-full", "")


def _run_one(job, slot, n_gpus, command_for, queue_dir: Path, timeout_min: float, log) -> dict:
    key = job["key"]
    result_file = queue_dir / "results" / f"{key}.json"
    log_path = queue_dir / "logs" / f"{key}.log"
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if n_gpus:
        env["CUDA_VISIBLE_DEVICES"] = str(slot)
    prefix = f"[{'gpu' if n_gpus else 'slot'}{slot} {_short(key)}]"
    row = {"key": key, "overrides": job["overrides"], "slot": slot, "status": "failed", "note": ""}

    log(f"{prefix} START {' '.join(job['overrides'])}")
    start = time.time()
    proc = subprocess.Popen(command_for(job, result_file, slot), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env, cwd=REPO_ROOT, bufsize=1)
    timed_out = threading.Event()

    def kill():
        timed_out.set()
        proc.kill()

    timer = threading.Timer(timeout_min * 60, kill)
    timer.start()
    with open(log_path, "w") as f:
        for line in proc.stdout:
            f.write(line)
            log(f"{prefix} {line.rstrip()}")
    code = proc.wait()
    timer.cancel()
    row["minutes"] = (time.time() - start) / 60

    if timed_out.is_set():
        row.update(status="timeout", note=f"killed after {timeout_min:g} min")
    elif code != 0:
        row["note"] = f"exit code {code}; see logs/{log_path.name}"
    elif not result_file.is_file():
        row["note"] = "finished without a result file"
    else:
        run_dir = Path(json.loads(result_file.read_text())["run_dir"])
        metrics = json.loads((run_dir / "metrics.json").read_text())
        row.update(status="finished", run_dir=str(run_dir), **{k: metrics.get(k) for k in TABLE_KEYS})
    log(f"{prefix} {row['status'].upper()} after {row['minutes']:.1f} min {row['note']}".rstrip())
    return row


def _fmt(value, spec):
    return "-" if value is None else format(value, spec)


def print_table(rows: list[dict], log=print) -> None:
    log("\n=== Queue results ===")
    log(f"{'run key':38s} {'status':9s} {'min':>6s} {'steps/s':>7s} {'PushT':>6s} {'Fetch':>6s} {'mean':>6s} "
        f"{'val L1':>7s}  note")
    for r in rows:
        log(f"{r['key'] or r['label']:38s} {r['status']:9s} {_fmt(r.get('minutes') or r.get('total_minutes'), '6.1f')} "
            f"{_fmt(r.get('steps_per_sec'), '7.2f')} {_fmt(r.get('pusht_success'), '6.1%')} "
            f"{_fmt(r.get('fetch_success'), '6.1%')} {_fmt(r.get('mean_success'), '6.1%')} "
            f"{_fmt(r.get('val_loss'), '7.4f')}  {r.get('note', '')}")

    groups = {}
    for r in rows:
        if r["status"] in ("finished", "skipped") and r.get("mean_success") is not None:
            groups.setdefault(re.sub(r"-s\d+-", "-", r["key"]), []).append(r)
    if not groups:
        return
    log("\n=== Per variant: mean +- std over seeds | pooled 95% Wilson interval ===")
    for name, rs in groups.items():
        parts = [f"{name:34s} seeds {len(rs)}"]
        for metric in ("pusht", "fetch", "mean"):
            values = [r[f"{metric}_success"] for r in rs if r.get(f"{metric}_success") is not None]
            parts.append(f"{metric} {np.mean(values):.1%} +- {np.std(values, ddof=1) if len(values) > 1 else 0:.1%}")
        for robot in ("pusht", "fetch"):
            k = [r.get(f"{robot}_successes") for r in rs]
            n = [r.get(f"{robot}_n") for r in rs]
            if None not in k and None not in n:
                lo, hi = wilson_interval(int(sum(k)), int(sum(n)))
                parts.append(f"{robot} pooled {int(sum(k))}/{int(sum(n))} [{lo:.1%}, {hi:.1%}]")
        log(" | ".join(parts))


def run_queue(configs, slots: int | None = None, workers_per_run="auto", extra=(), allow_dirty: bool = False,
              rerun: bool = False, is_finished=None, command_for=None, timeout_min: float = 120,
              queue_dir=None, log=print) -> list[dict]:
    """Run every config (a list of override strings) and return one result row per config."""
    import torch

    n_gpus = torch.cuda.device_count()
    slots = slots or max(1, n_gpus)
    if n_gpus and slots > n_gpus:
        raise ValueError(f"{slots} slots but only {n_gpus} GPUs")
    if workers_per_run == "auto":
        workers_per_run = default_workers_per_run(slots)
    git = git_commit()
    if queue_dir is None:
        queue_dir = outputs_dir() / f"queue-{git['commit'][:7]}-{time.strftime('%Y%m%d-%H%M%S')}"
    queue_dir = Path(queue_dir)
    for sub in ("logs", "results"):
        (queue_dir / sub).mkdir(parents=True, exist_ok=True)

    jobs = plan_jobs(configs, workers_per_run, extra)
    save_run_info(queue_dir, {"job": "run_queue", "configs": [list(c) for c in configs], "extra": list(extra),
                              "slots": slots, "gpus": n_gpus, "workers_per_run": workers_per_run,
                              "rerun": rerun, "timeout_min": timeout_min, "git_commit": git["commit"]}, git)
    log(f"Queue: {len(jobs)} configs | {slots} at a time on {f'{n_gpus} GPUs' if n_gpus else 'CPU'} | "
        f"data-loader workers per run: {workers_per_run if workers_per_run is not None else 'config default'} | "
        f"folder {queue_dir}")

    if is_finished is None and not rerun:
        is_finished = wandb_finished_lookup()
    rows = [None] * len(jobs)
    pending = queue.Queue()
    for job in jobs:
        if job["error"]:
            rows[job["index"]] = {"key": None, "label": job["label"], "overrides": job["overrides"],
                                  "status": "failed", "note": f"bad config: {job['error']}"}
            log(f"SKIP bad config {job['label']!r}: {job['error']}")
            continue
        done = None if rerun else is_finished(job["key"])
        if done:
            rows[job["index"]] = {"key": job["key"], "overrides": job["overrides"], "status": "skipped",
                                  "note": f"already finished in W&B: {done.get('run')} ({done.get('commit')})",
                                  **{k: done.get(k) for k in TABLE_KEYS}}
            log(f"SKIP {job['key']}: already finished in W&B as {done.get('run')} (commit {done.get('commit')})")
            continue
        pending.put(job)

    command_for = command_for or _train_command(allow_dirty)
    lock = threading.Lock()

    def slot_worker(slot):
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            try:
                row = _run_one(job, slot, n_gpus, command_for, queue_dir, timeout_min, log)
            except Exception as e:  # noqa: BLE001 - the queue keeps going whatever happens
                row = {"key": job["key"], "overrides": job["overrides"], "slot": slot, "status": "failed",
                       "note": f"queue error: {type(e).__name__}: {e}"}
            with lock:
                rows[job["index"]] = row
                (queue_dir / "queue_results.json").write_text(json.dumps(
                    {"workers_per_run": workers_per_run, "slots": slots, "rows": [r for r in rows if r]}, indent=2))

    threads = [threading.Thread(target=slot_worker, args=(s,), daemon=True) for s in range(slots)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    (queue_dir / "queue_results.json").write_text(json.dumps(
        {"workers_per_run": workers_per_run, "slots": slots, "rows": rows}, indent=2))
    print_table(rows, log)
    log(f"Saved {queue_dir / 'queue_results.json'}")
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+", help='one quoted string per run, e.g. "cond=film seed=1"')
    parser.add_argument("--slots", type=int, help="runs at a time (default: number of GPUs, at least 1)")
    parser.add_argument("--workers-per-run", type=int, help="data-loader workers per run (default: 2 if 2+ slots)")
    parser.add_argument("--extra", nargs="*", default=[], help="overrides added to every config")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--rerun", action="store_true", help="do not skip runs already finished in W&B")
    parser.add_argument("--timeout-min", type=float, default=120)
    args = parser.parse_args(argv)

    from vla.utils.runtime import load_dotenv_if_local
    from vla.utils.tracking import check_wandb_key

    load_dotenv_if_local()
    check_wandb_key()
    rows = run_queue([c.split() for c in args.configs], args.slots,
                     "auto" if args.workers_per_run is None else args.workers_per_run, args.extra,
                     args.allow_dirty, args.rerun, timeout_min=args.timeout_min)
    return 1 if any(r["status"] in ("failed", "timeout") for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
