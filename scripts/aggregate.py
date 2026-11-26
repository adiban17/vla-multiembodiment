"""Phase 7: collect every finished run from W&B into results/results.csv and the README tables.

    .venv/bin/python scripts/aggregate.py                          # W&B key from .env
    .venv/bin/python scripts/aggregate.py --episodes-dir outputs/kaggle   # + exact per-seed results

What it does, in order:
1. Lists every run in the W&B project. A run counts if it finished, is a training run, is not a smoke
   test (name starts with "smoke-"), did not train on Mac debug data, and was evaluated.
2. Lines them up against the 30 runs the experiment plan expects (9 conditioning, 12 baseline,
   9 adaptation). Runs that are missing or still running are marked, so you can rerun this script later.
3. Finds which of the 50 start seeds each run solved. W&B keeps only totals (e.g. 32/50) for runs made
   before Phase 7, so this needs the run's metrics.json from the Kaggle outputs (--episodes-dir). Without
   it, the paired bootstrap falls back to the 10-episode blocks printed in each run's console log
   (labeled "approx."), and FetchPush on the 45 unsolved starts is given as a range.
4. Writes results/results.csv (one row per run, with W&B run IDs and git commits), results/tables.md,
   the README block between the results markers, results/bc_mlp_fetch.md, 3 GIFs and the plots.

Every number in a table can be traced: each table lists the W&B run IDs it was computed from.
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import kaggle_job, plots, ...` works

from kaggle_job import (ADAPT_COND, ADAPT_MODES, BASELINE_ROBOTS, BASELINES, CONDITIONING_VARIANTS, SEEDS,
                        WANDB_PROJECT)
from vla.evaluate import ROBOTS, SEED_START, wilson_interval
from vla.utils.runtime import REPO_ROOT, load_dotenv_if_local, outputs_dir

ADAPT_STEPS = 2500  # batch_adaptation was launched with --steps 2500
N_STARTS = 50
KNOWN_SOLVED = (100004, 100010, 100021, 100036, 100049)  # scripts/fetch_start_check.py, Phase 5
FETCH_FLOOR = len(KNOWN_SOLVED) / N_STARTS  # 10%: a policy that leaves the cube alone wins these
N_BOOT = 10_000
RESULTS_DIR = REPO_ROOT / "results"
README = REPO_ROOT / "README.md"
README_START = "<!-- results:start -->"
README_END = "<!-- results:end -->"
ADAPT_ORDER = ("frozen_matched", "lora16", "full_ft")  # table order: control, LoRA, full fine-tuning
ROBOT_NAMES = {"pusht": "PushT", "fetch": "FetchPush"}

CSV_FIELDS = [
    "run_key", "group", "label", "status", "wandb_id", "run_name", "url", "git_commit", "model", "cond", "adapt",
    "baseline", "robot", "seed", "steps",
    "pusht_successes", "pusht_n", "pusht_success", "pusht_wilson_low", "pusht_wilson_high", "pusht_mean_max_coverage",
    "fetch_successes", "fetch_n", "fetch_success", "fetch_wilson_low", "fetch_wilson_high",
    "fetch_unsolved_successes", "fetch_unsolved_low", "fetch_unsolved_high", "fetch_unsolved_n",
    "mean_success", "train_loss", "val_loss", "peak_gpu_mem_gb_at_200", "total_minutes", "train_minutes",
    "eval_minutes", "steps_per_sec", "trainable_params", "encoder_trainable_params", "gpu",
    "episodes_source", "pusht_episodes", "fetch_episodes", "pusht_blocks", "fetch_blocks",
]


# --- the experiment plan --------------------------------------------------------------------


def expected_runs(adapt_steps: int = ADAPT_STEPS) -> list[dict]:
    """The 30 runs of the plan, each with the run key the training code itself gives it."""
    from vla.train import run_key
    from vla.utils.config import load_config

    specs = []
    for cond in CONDITIONING_VARIANTS:
        specs += [("conditioning", cond, [f"cond={cond}", f"seed={s}"]) for s in SEEDS]
    for baseline in BASELINES:
        for robot in BASELINE_ROBOTS:
            specs += [("baseline", f"{baseline}-{robot}", [f"baseline={baseline}", f"robot={robot}", f"seed={s}"])
                      for s in SEEDS]
    for mode in ADAPT_ORDER:
        assert mode in ADAPT_MODES
        specs += [("adaptation", mode, [f"cond={ADAPT_COND}", f"adapt={mode}", f"seed={s}",
                                        f"train.steps={int(adapt_steps)}"]) for s in SEEDS]
    out = []
    for group, label, overrides in specs:
        cfg = load_config(overrides)
        out.append({"group": group, "label": label, "overrides": overrides, "key": run_key(cfg),
                    "cond": cfg.cond, "adapt": cfg.adapt, "baseline": cfg.baseline, "robot": cfg.robot,
                    "seed": int(cfg.seed), "steps": int(cfg.train.steps)})
    return out


# --- which W&B runs count ------------------------------------------------------------------


def _info(run) -> dict:
    return (run.config or {}).get("run") or {}


def is_eligible(run) -> bool:
    """Finished, a training run, not a smoke test, not Mac debug data, and evaluated."""
    info = _info(run)
    return (run.state == "finished" and run.job_type == "train" and not str(run.name).startswith("smoke-")
            and bool(info.get("key")) and not info.get("debug_data")
            and run.summary.get("eval/mean_success") is not None)


def _is_candidate(run) -> bool:
    """A real (non-smoke, non-debug) training run, whatever its state."""
    info = _info(run)
    return (run.job_type == "train" and not str(run.name).startswith("smoke-") and bool(info.get("key"))
            and not info.get("debug_data"))


def _get(summary, *keys):
    for k in keys:
        v = summary.get(k)
        if v is not None:
            return v
    return None


def make_row(run, exp: dict | None, status: str, run_key: str | None = None) -> dict:
    """One results.csv row. exp is the plan entry (None for a run outside the plan)."""
    row = dict.fromkeys(CSV_FIELDS)
    cfg = {} if run is None else (run.config or {})
    meta = exp or {"group": "other", "label": None, "cond": cfg.get("cond"), "adapt": cfg.get("adapt"),
                   "baseline": cfg.get("baseline"), "robot": cfg.get("robot"), "seed": cfg.get("seed"),
                   "steps": (cfg.get("train") or {}).get("steps")}
    row.update({"run_key": run_key or (exp or {}).get("key"), "group": meta["group"], "label": meta["label"],
                "status": status, **{k: meta[k] for k in ("cond", "adapt", "baseline", "robot", "seed", "steps")}})
    if row["label"] is None and run is not None:
        row["label"] = str(row["run_key"]).rsplit("-s", 1)[0]
    if run is None:
        return row
    s, info = run.summary, _info(run)
    row.update({"wandb_id": run.id, "run_name": run.name, "url": run.url,
                "git_commit": str(info.get("git_commit") or s.get("git_commit") or "")[:7],
                "model": info.get("model") or s.get("model") or "vla", "gpu": s.get("gpu") or info.get("gpu")})
    for robot in ROBOTS:
        for k in ("successes", "n", "success", "wilson_low", "wilson_high"):
            row[f"{robot}_{k}"] = s.get(f"eval/{robot}_{k}")
    row["pusht_mean_max_coverage"] = s.get("eval/pusht_mean_max_coverage")
    row["mean_success"] = s.get("eval/mean_success")
    for k in ("train_loss", "val_loss", "peak_gpu_mem_gb_at_200", "total_minutes", "train_minutes", "eval_minutes",
              "steps_per_sec"):
        row[k] = s.get(k)
    row["trainable_params"] = _get(s, "trainable_params") or info.get("trainable_params")
    enc = s.get("encoder_trainable_params", info.get("encoder_trainable_params"))
    if enc is None and info.get("encoder_mode", "frozen") in ("frozen", "none"):
        enc = 0  # runs from before Phase 6 read cached features: DINOv2 has no trainable weights
    row["encoder_trainable_params"] = enc
    row["_run"] = run
    return row


def _created(run) -> str:
    return str(getattr(run, "created_at", "") or "")


def match_runs(runs, expected: list[dict]) -> list[dict]:
    """One row per planned run (finished / running / failed / missing), plus duplicates and runs outside the plan."""
    runs = sorted(runs, key=_created, reverse=True)  # newest first
    by_key: dict[str, list] = {}
    for run in runs:
        if _is_candidate(run):
            by_key.setdefault(_info(run)["key"], []).append(run)
    rows, planned = [], set()
    for exp in expected:
        planned.add(exp["key"])
        found = by_key.get(exp["key"], [])
        good = [r for r in found if is_eligible(r)]
        if good:
            rows.append(make_row(good[0], exp, "finished"))
            rows += [make_row(r, exp, "duplicate") for r in good[1:]]
        elif found:
            r = found[0]
            status = r.state if r.state != "finished" else "not evaluated"
            rows.append(make_row(r, exp, status))
        else:
            rows.append(make_row(None, exp, "missing"))
    for key, found in by_key.items():
        if key in planned:
            continue
        good = [r for r in found if is_eligible(r)]
        for i, r in enumerate(good):
            rows.append(make_row(r, None, "finished" if i == 0 else "duplicate", run_key=key))
    return rows


# --- per-seed outcomes ---------------------------------------------------------------------


def load_local_metrics(episodes_dir) -> dict:
    """run_id -> metrics.json contents, for every metrics.json (with an "eval" block) under episodes_dir."""
    out = {}
    if episodes_dir is None or not Path(episodes_dir).is_dir():
        return out
    for path in sorted(Path(episodes_dir).rglob("metrics.json")):
        try:
            m = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(m, dict) and isinstance(m.get("eval"), dict):
            out.setdefault(m.get("run_id") or path.parent.name, m)
    return out


def _episode_string(episodes) -> str:
    return "".join("1" if e["success"] else "0" for e in sorted(episodes, key=lambda e: e["seed"]))


def episodes_for(row: dict, local: dict, summary) -> dict:
    """{robot: "0110...", "source": "metrics.json" | "wandb" | "none"}. A source is used only if its totals
    agree with the W&B summary, so a metrics.json from a different run can never slip in."""
    robots = [r for r in ROBOTS if row.get(f"{r}_successes") is not None]

    def agrees(strings):
        return all(len(strings[r]) == row[f"{r}_n"] and strings[r].count("1") == row[f"{r}_successes"]
                   for r in robots)

    m = local.get(row.get("run_name"))
    if m and robots:
        try:
            strings = {r: _episode_string(m["eval"][r]["episodes"]) for r in robots}
            if agrees(strings):
                return {**strings, "source": "metrics.json"}
            print(f"WARNING: {row['run_name']}: metrics.json totals differ from W&B; not used")
        except (KeyError, TypeError):
            print(f"WARNING: {row['run_name']}: metrics.json has no per-episode results; not used")
    summary = summary or {}
    strings = {r: summary.get(f"eval/{r}_episode_success") for r in robots}
    if robots and all(isinstance(v, str) for v in strings.values()) and agrees(strings):
        return {**strings, "source": "wandb"}
    return {"source": "none"}


_BLOCK_RE = re.compile(r"^\s*(pusht|fetch): (\d+)/(\d+) episodes, (\d+) successes", re.MULTILINE)


def parse_eval_blocks(log_text: str) -> dict:
    """Successes per block of episodes, from the evaluation progress lines in a run's console log
    ("  pusht: 20/50 episodes, 1 successes, 42s" every 10 episodes). Only the last complete evaluation
    per robot is kept, and only if all blocks have the same size."""
    seqs: dict[str, list] = {}
    done: dict[str, list] = {}
    for robot, i, n, k in _BLOCK_RE.findall(log_text.replace("\r", "\n")):
        i, n, k = int(i), int(n), int(k)
        seq = seqs.setdefault(robot, [])
        if seq and i <= seq[-1][0]:
            seq.clear()
        seq.append((i, k))
        if i == n:
            done[robot] = list(seq)
    out = {}
    for robot, seq in done.items():
        ends = [0] + [i for i, _ in seq]
        counts = [0] + [k for _, k in seq]
        sizes = {b - a for a, b in zip(ends, ends[1:])}
        if len(sizes) == 1:
            out[robot] = [b - a for a, b in zip(counts, counts[1:])]
    return out


def unsolved_successes(episodes: str, solved, seed_start: int = SEED_START) -> tuple[int, int]:
    """(successes, episodes) over the starts that are NOT already solved before the robot moves."""
    keep = [c for i, c in enumerate(episodes) if seed_start + i not in solved]
    return keep.count("1"), len(keep)


def unsolved_range(k: int, n_total: int = N_STARTS, n_solved: int = len(KNOWN_SOLVED)) -> tuple[int, int]:
    """Without per-seed results: at least k - n_solved and at most k of the wins are on unsolved starts."""
    return max(0, k - n_solved), min(k, n_total - n_solved)


def solved_starts(run_check: bool = True) -> set:
    """The FetchPush eval seeds already solved at reset, recomputed with scripts/fetch_start_check.py."""
    if not run_check:
        return set(KNOWN_SOLVED)
    from fetch_start_check import start_distances

    found = {r["seed"] for r in start_distances(range(SEED_START, SEED_START + N_STARTS)) if r["solved_at_start"]}
    if found != set(KNOWN_SOLVED):
        print(f"WARNING: start check found {sorted(found)}, not {list(KNOWN_SOLVED)} (MuJoCo can differ between "
              "machines); using the Kaggle-matched list")
        return set(KNOWN_SOLVED)
    print(f"FetchPush starts already solved (fetch_start_check.py): {sorted(found)}")
    return found


# --- statistics ----------------------------------------------------------------------------


def mean_std(values) -> tuple:
    """Mean and sample standard deviation (ddof=1) over seeds; std is 0 for a single seed."""
    v = [float(x) for x in values if x is not None]
    if not v:
        return None, None
    if len(v) == 1:
        return v[0], 0.0
    return float(np.mean(v)), float(np.std(v, ddof=1))


def pooled_wilson(successes, ns) -> tuple:
    """(k, n, low, high): all seeds' rollouts pooled, with a 95% Wilson interval."""
    k, n = int(sum(successes)), int(sum(ns))
    if n == 0:
        return 0, 0, None, None
    lo, hi = wilson_interval(k, n)
    return k, n, lo, hi


def paired_bootstrap(a, b, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Gap A - B in percentage points, with a 95% interval from resampling the shared start units.

    a, b: (seeds, units) success fractions; a unit is one start seed (0 or 1) or a block of starts.
    Both variants face the same starts, so each resample draws the SAME units for both (paired).
    A variant's rate in a resample = the mean over its seeds and the drawn units."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = a.mean(axis=0) - b.mean(axis=0)  # per unit
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    gaps = 100 * diff[idx].mean(axis=1)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return {"gap_pp": float(100 * (a.mean() - b.mean())), "lo_pp": float(lo), "hi_pp": float(hi),
            "p_le_0": float(np.mean(gaps <= 0)), "n_units": int(diff.size), "n_boot": n_boot}


def _units(rows, robot, kind):
    """(seeds, units) matrix for one configuration, or None if any seed lacks the data."""
    mats = []
    for r in rows:
        if robot == "mean":
            p, f = _units([r], "pusht", kind), _units([r], "fetch", kind)
            if p is None or f is None or p.shape != f.shape:
                return None
            mats.append((p[0] + f[0]) / 2)
            continue
        if kind == "episodes":
            s = r.get(f"{robot}_episodes")
            if not s:
                return None
            mats.append(np.array([c == "1" for c in s], dtype=float))
        else:
            s, n = r.get(f"{robot}_blocks"), r.get(f"{robot}_n")
            if not s or not n:
                return None
            counts = np.array([int(x) for x in s.split()], dtype=float)
            mats.append(counts / (n / len(counts)))
    if not mats or len({m.shape for m in mats}) != 1:
        return None
    return np.stack(mats)


def compare(rows_a, rows_b, robot) -> dict | None:
    """Paired bootstrap between two configurations: exact per start seed if every run has per-seed results,
    otherwise approximate over the 10-episode blocks of the console log, otherwise None."""
    if not rows_a or not rows_b:
        return None
    for kind, method in (("episodes", "exact"), ("blocks", "approx")):
        a, b = _units(rows_a, robot, kind), _units(rows_b, robot, kind)
        if a is not None and b is not None and a.shape[1] == b.shape[1]:
            return {**paired_bootstrap(a, b), "method": method}
    return None


# --- README, BC-MLP check ------------------------------------------------------------------


def replace_readme_block(text: str, block: str) -> str:
    """Put `block` between the results markers, replacing what was there. The first time, the block goes
    before the first "## " section. The rest of the README is left as it is."""
    full = f"{README_START}\n{block.strip()}\n{README_END}"
    if README_START in text and README_END in text:
        pattern = re.compile(re.escape(README_START) + r".*?" + re.escape(README_END), re.DOTALL)
        return pattern.sub(lambda _: full, text, count=1)
    i = text.find("\n## ")
    if i < 0:
        return text.rstrip("\n") + "\n\n" + full + "\n"
    return text[:i].rstrip("\n") + "\n\n" + full + "\n" + text[i:]


def _window_mean(curve, lo, hi):
    v = [loss for step, loss in curve if lo <= step < hi]
    return float(np.mean(v)) if v else None


def curve_change(curve, start_step: int = 400, window: int = 200) -> float:
    """Relative change of the training loss from just after warmup (steps 400-599) to the last 200 steps."""
    last = max(step for step, _ in curve)
    early = _window_mean(curve, start_step, start_step + window)
    late = _window_mean(curve, last - window + 1, last + 1)
    return late / early - 1


def loss_verdict(bc_curves, contrast_curves, flat_tol: float = 0.05, learn_drop: float = 0.30) -> dict:
    """'flat' if every BC-MLP curve moves less than flat_tol after warmup while the contrast model (ResNet-18
    on the same data) drops more than learn_drop: the data is learnable, BC-MLP just did not learn it."""
    bc = [curve_change(c) for c in bc_curves]
    contrast = [curve_change(c) for c in contrast_curves]
    bc_change, contrast_change = float(np.mean(bc)), float(np.mean(contrast))
    if all(abs(x) < flat_tol for x in bc):
        verdict = "flat" if contrast_change < -learn_drop else "flat (but the contrast model did not learn either)"
    elif bc_change <= -flat_tol:
        verdict = "learned"
    else:
        verdict = "unclear"
    return {"verdict": verdict, "bc_change": bc_change, "contrast_change": contrast_change,
            "bc_changes": bc, "contrast_changes": contrast}


def loss_at(curve, step):
    """The logged loss at the logged step nearest to `step`."""
    return min(curve, key=lambda p: abs(p[0] - step))[1]


# --- formatting ----------------------------------------------------------------------------


def _pct(x, digits=1):
    return "–" if x is None else f"{100 * x:.{digits}f}%"


def _ms(m, s):
    if m is None:
        return "–"
    return f"{100 * m:.1f}% (1 seed)" if s is None else f"{100 * m:.1f} ± {100 * s:.1f}%"


def _wil(k, n, lo, hi):
    return "–" if not n else f"{k}/{n} = {100 * k / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def _gap(c):
    if c is None:
        return "needs per-seed data"
    mark = "" if c["lo_pp"] <= 0 <= c["hi_pp"] else " *"
    return f"{c['gap_pp']:+.1f} [{c['lo_pp']:+.1f}, {c['hi_pp']:+.1f}]{mark}"


def _ids(rows):
    return ", ".join(f"`{r['wandb_id']}`" for r in rows) or "none yet"


# --- configuration summaries ---------------------------------------------------------------


def config_rows(rows, group, label) -> list[dict]:
    return sorted([r for r in rows if r["group"] == group and r["label"] == label and r["status"] == "finished"],
                  key=lambda r: r["seed"])


def summarize(cfg_rows) -> dict:
    """Mean ± std over seeds and the pooled Wilson interval, per robot present."""
    out = {"n_seeds": len(cfg_rows), "ids": _ids(cfg_rows)}
    for robot in ROBOTS:
        rr = [r for r in cfg_rows if r.get(f"{robot}_n")]
        if not rr:
            continue
        out[robot] = {"mean_std": mean_std([r[f"{robot}_success"] for r in rr]),
                      "pooled": pooled_wilson([r[f"{robot}_successes"] for r in rr], [r[f"{robot}_n"] for r in rr])}
    out["mean_success"] = mean_std([r["mean_success"] for r in cfg_rows])
    out["coverage"] = mean_std([r["pusht_mean_max_coverage"] for r in cfg_rows])
    fr = [r for r in cfg_rows if r.get("fetch_n")]
    if fr:
        exact = all(r["fetch_unsolved_successes"] is not None for r in fr)
        n = sum(r["fetch_unsolved_n"] for r in fr)
        lo, hi = sum(r["fetch_unsolved_low"] for r in fr), sum(r["fetch_unsolved_high"] for r in fr)
        out["fetch_unsolved"] = {"exact": exact, "n": n, "low": lo, "high": hi,
                                 "wilson": wilson_interval(lo, n) if exact else None}
    for k in ("peak_gpu_mem_gb_at_200", "total_minutes", "trainable_params", "encoder_trainable_params"):
        out[k] = mean_std([r[k] for r in cfg_rows])
    if len(cfg_rows) == 1:  # a std over one seed means nothing: show none rather than "± 0.0"
        for robot in ROBOTS:
            if robot in out:
                out[robot]["mean_std"] = (out[robot]["mean_std"][0], None)
        out["mean_success"] = (out["mean_success"][0], None)
    return out


def _seeds(s):
    return f"{s['n_seeds']}/{len(SEEDS)}"


def _unsolved(u):
    if u is None:
        return "–"
    if u["exact"]:
        lo, hi = u["wilson"]
        return f"{u['low']}/{u['n']} = {100 * u['low'] / u['n']:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"
    return f"{u['low']}–{u['high']} of {u['n']} (range; needs per-seed data)"


# --- tables --------------------------------------------------------------------------------


def _gap_rows(pairs, robots, method_note):
    lines = ["| Pair (A − B) | " + " | ".join(f"{ROBOT_NAMES.get(r, 'Mean')} gap, pp [95% CI]" for r in robots)
             + " | Method | Verdict |",
             "|---" * (len(robots) + 3) + "|"]
    for name, cmp in pairs:
        cells = [_gap(cmp.get(r)) for r in robots]
        methods = {c["method"] for c in cmp.values() if c}
        method = ("exact (50 starts)" if methods == {"exact"} else
                  "approx. (5 blocks of 10 starts)" if methods else "–")
        real = [ROBOT_NAMES.get(r, "mean") for r in robots
                if cmp.get(r) and not (cmp[r]["lo_pp"] <= 0 <= cmp[r]["hi_pp"])]
        if not methods:
            verdict = "–"
        elif not real:
            verdict = "not distinguishable"
        elif methods == {"exact"}:
            verdict = "real on " + ", ".join(real)
        else:  # 5 blocks are too few for a trustworthy interval: never call it real
            verdict = "excludes 0 on " + ", ".join(real) + " (approx. only: provisional)"
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {method} | {verdict} |")
    lines.append("")
    lines.append(method_note)
    return lines


def gap_note(all_exact: bool) -> str:
    """The note under table 1. The "approx." explanation is only included while some run lacks per-seed data."""
    note = ("Gaps are A − B in percentage points. The interval is a paired bootstrap (10,000 resamples) over the "
            "50 start seeds that every run shares, pooling the 3 training seeds; `*` = the interval excludes 0. "
            "A gap is called real only when the exact (per-seed) interval excludes 0. ")
    if all_exact:
        note += "Every run has per-seed results, so every interval here is exact. "
    else:
        note += ("\"approx.\" = per-seed results were not available, so the bootstrap resamples the 5 blocks of 10 "
                 "starts printed in each run's console log. With only 5 units that interval is unreliable (too narrow "
                 "and lumpy), so an approx. `*` is provisional, not a claim. ")
    return note + "The bootstrap treats the training seeds as fixed: it measures start-state noise, not seed-to-seed noise."


def table_conditioning(rows) -> tuple[list[str], dict]:
    sums = {v: summarize(config_rows(rows, "conditioning", v)) for v in CONDITIONING_VARIANTS}
    lines = ["### 1. Conditioning variants (frozen encoders, 4,000 steps)", "",
             "| Variant | Seeds | PushT success (mean ± std) | PushT pooled [95% Wilson] | PushT max coverage | "
             "FetchPush success (mean ± std) | FetchPush pooled [95% Wilson] | Mean success |",
             "|---|---|---|---|---|---|---|---|"]
    for v, s in sums.items():
        p, f = s.get("pusht", {}), s.get("fetch", {})
        cov = s["coverage"]
        lines.append(f"| {v} | {_seeds(s)} | {_ms(*p.get('mean_std', (None, None)))} | "
                     f"{_wil(*p['pooled']) if p else '–'} | "
                     f"{'–' if cov[0] is None else f'{cov[0]:.3f} ± {cov[1]:.3f}'} | "
                     f"{_ms(*f.get('mean_std', (None, None)))} | {_wil(*f['pooled']) if f else '–'} | "
                     f"{_ms(*s['mean_success'])} |")
    lines += ["", "Runs (W&B IDs, seeds 0, 1, 2): " + " · ".join(f"{v}: {s['ids']}" for v, s in sums.items()), ""]
    pairs = []
    vs = list(CONDITIONING_VARIANTS)
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            a, b = config_rows(rows, "conditioning", vs[i]), config_rows(rows, "conditioning", vs[j])
            pairs.append((f"{vs[i]} − {vs[j]}", {r: compare(a, b, r) for r in ("pusht", "fetch", "mean")}))
    all_exact = all(r.get("episodes_source") not in (None, "none") for r in rows if r["status"] == "finished")
    lines += ["Gaps between variants:", ""] + _gap_rows(pairs, ("pusht", "fetch", "mean"), gap_note(all_exact)) + [""]
    return lines, {"summaries": sums, "pairs": pairs}


def _fetch_configs(rows):
    """(name, rows) for every configuration that has FetchPush results."""
    out = [(v, config_rows(rows, "conditioning", v)) for v in CONDITIONING_VARIANTS]
    out += [(b, config_rows(rows, "baseline", f"{b}-fetch")) for b in BASELINES]
    out += [(f"{ADAPT_COND} {m} ({ADAPT_STEPS} steps)", config_rows(rows, "adaptation", m)) for m in ADAPT_ORDER]
    return out


def table_fetch(rows) -> list[str]:
    lines = ["### 2. FetchPush success two ways", "",
             f"{len(KNOWN_SOLVED)} of the 50 FetchPush start seeds ({', '.join(map(str, KNOWN_SOLVED))}) begin with the "
             "cube already within 5 cm of the target (`scripts/fetch_start_check.py`), so a policy that does "
             f"nothing scores **{FETCH_FLOOR:.0%}** (the floor; the random policy's 5 wins are exactly these seeds). "
             "The second column counts only the 45 starts per run that are not already solved (a do-nothing "
             "policy gets 0% there).", "",
             "| Configuration | Seeds | Overall, pooled [95% Wilson] | vs 10% floor (pp) | "
             "Unsolved starts only, pooled [95% Wilson] |", "|---|---|---|---|---|"]
    ids = []
    for name, rr in _fetch_configs(rows):
        s = summarize(rr)
        f = s.get("fetch")
        if not f:
            lines.append(f"| {name} | {_seeds(s)} | not finished | – | – |")
            continue
        k, n, lo, hi = f["pooled"]
        lines.append(f"| {name} | {_seeds(s)} | {_wil(k, n, lo, hi)} | {100 * (k / n - FETCH_FLOOR):+.1f} | "
                     f"{_unsolved(s.get('fetch_unsolved'))} |")
        ids.append(f"{name}: {s['ids']}")
    lines += ["", "Runs: " + " · ".join(ids), ""]
    return lines


def table_baselines(rows) -> tuple[list[str], list]:
    lines = ["### 3. Baselines vs the VLA (4,000 steps each)", "",
             "Baselines are single-robot specialists (one model per robot); the VLA variants are one model for "
             "both robots.", "",
             "| Model | PushT success (mean ± std) | PushT pooled [95% Wilson] | FetchPush success (mean ± std) | "
             "FetchPush pooled [95% Wilson] | FetchPush, unsolved starts |", "|---|---|---|---|---|---|"]
    ids = []
    entries = []
    for b in BASELINES:
        p, f = summarize(config_rows(rows, "baseline", f"{b}-pusht")), summarize(config_rows(rows, "baseline", f"{b}-fetch"))
        entries.append((b, p.get("pusht"), f.get("fetch"), f.get("fetch_unsolved")))
        ids.append(f"{b}-pusht: {p['ids']} · {b}-fetch: {f['ids']}")
    for v in CONDITIONING_VARIANTS:
        s = summarize(config_rows(rows, "conditioning", v))
        entries.append((f"VLA {v}", s.get("pusht"), s.get("fetch"), s.get("fetch_unsolved")))
        ids.append(f"VLA {v}: {s['ids']}")
    for name, p, f, u in entries:
        lines.append(f"| {name} | {_ms(*p['mean_std']) if p else '–'} | {_wil(*p['pooled']) if p else '–'} | "
                     f"{_ms(*f['mean_std']) if f else '–'} | {_wil(*f['pooled']) if f else '–'} | {_unsolved(u)} |")
    lines += ["", "Runs: " + " · ".join(ids), ""]
    best = best_variant(rows)
    pairs = []
    if best:
        vla = config_rows(rows, "conditioning", best)
        for b in BASELINES:
            pairs.append((f"VLA {best} − {b}", {
                "pusht": compare(vla, config_rows(rows, "baseline", f"{b}-pusht"), "pusht"),
                "fetch": compare(vla, config_rows(rows, "baseline", f"{b}-fetch"), "fetch")}))
        lines += [f"Gaps between the best VLA variant ({best}) and each baseline:", ""]
        lines += _gap_rows(pairs, ("pusht", "fetch"), "Same bootstrap as in table 1.") + [""]
    return lines, pairs


def _ratio(a, b):
    if a is None or b is None:
        return "–"
    if b == 0:
        return "n/a (full_ft = 0)"
    return f"{100 * a / b:.0f}%"


def table_adaptation(rows) -> tuple[list[str], dict]:
    sums = {m: summarize(config_rows(rows, "adaptation", m)) for m in ADAPT_ORDER}
    full = sums["full_ft"]
    lines = [f"### 4. Encoder adaptation ({ADAPT_COND}, {ADAPT_STEPS:,} steps, batch 64)", "",
             "`frozen_matched` = frozen DINOv2 with the same batch size and steps as the adaptation runs (the fair "
             "control). Minutes are wall time per run including evaluation, with two runs sharing the machine "
             "(one per T4). Peak GPU memory is allocated memory at training step 200.", "",
             "| Mode | Seeds | PushT success | FetchPush success | Mean success | % of full_ft (mean) | "
             "Peak GPU memory @200 | Minutes per run | Trainable params (DINOv2 + rest) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m, s in sums.items():
        if not s["n_seeds"]:
            lines.append(f"| {m} | 0/{len(SEEDS)} | not finished | | | | | | |")
            continue
        p, f = s.get("pusht"), s.get("fetch")
        mem, mins = s["peak_gpu_mem_gb_at_200"], s["total_minutes"]
        tp, enc = s["trainable_params"][0], s["encoder_trainable_params"][0]
        params = "–" if tp is None else f"{int(enc or 0):,} + {int(tp - (enc or 0)):,}"
        lines.append(f"| {m} | {_seeds(s)} | {_ms(*p['mean_std']) if p else '–'} | {_ms(*f['mean_std']) if f else '–'} | "
                     f"{_ms(*s['mean_success'])} | {_ratio(s['mean_success'][0], full['mean_success'][0])} | "
                     f"{'–' if mem[0] is None else f'{mem[0]:.2f} GB'} | {'–' if mins[0] is None else f'{mins[0]:.1f}'} | "
                     f"{params} |")
    lines += ["", "Runs: " + " · ".join(f"{m}: {s['ids']}" for m, s in sums.items()), ""]
    lora = sums["lora16"]
    if lora["n_seeds"] and full["n_seeds"]:
        per = ", ".join(f"{ROBOT_NAMES[r]} {_ratio(lora[r]['mean_std'][0], full[r]['mean_std'][0])}"
                        for r in ROBOTS if r in lora and r in full)
        lm, fm = lora["peak_gpu_mem_gb_at_200"][0], full["peak_gpu_mem_gb_at_200"][0]
        mem = f"{100 * (1 - lm / fm):.0f}% lower" if lm and fm else "–"
        lines.append(f"LoRA reaches {_ratio(lora['mean_success'][0], full['mean_success'][0])} of full fine-tuning's "
                     f"mean success ({per}) with {mem} peak GPU memory at step 200 "
                     f"({lora['n_seeds']} and {full['n_seeds']} of {len(SEEDS)} seeds).")
    else:
        lines.append(f"Not finished yet: lora16 {_seeds(lora)} seeds, full_ft {_seeds(full)} seeds. Rerun this script "
                     "when the adaptation batch is done.")
    lines.append("")
    pairs = []
    for a, b in (("lora16", "full_ft"), ("lora16", "frozen_matched"), ("full_ft", "frozen_matched")):
        ra, rb = config_rows(rows, "adaptation", a), config_rows(rows, "adaptation", b)
        if ra and rb:
            pairs.append((f"{a} − {b}", {r: compare(ra, rb, r) for r in ("pusht", "fetch", "mean")}))
    if pairs:
        lines += ["Gaps between adaptation modes:", ""]
        lines += _gap_rows(pairs, ("pusht", "fetch", "mean"), "Same bootstrap as in table 1.") + [""]
    vs_resnet = []
    for m in ("full_ft", "lora16"):
        ra = config_rows(rows, "adaptation", m)
        rp, rf = config_rows(rows, "baseline", "resnet18-pusht"), config_rows(rows, "baseline", "resnet18-fetch")
        if ra and rp and rf:
            vs_resnet.append((f"{m} − resnet18", {"pusht": compare(ra, rp, "pusht"), "fetch": compare(ra, rf, "fetch")}))
    if vs_resnet:
        lines += ["Adapted VLA vs the ResNet-18 specialist (ResNet-18: 4,000 steps, batch 128, one robot per model; "
                  "adapted VLA: 2,500 steps, batch 64, one model for both robots):", ""]
        lines += _gap_rows(vs_resnet, ("pusht", "fetch"), "Same bootstrap as in table 1.") + [""]
    pairs = pairs + vs_resnet
    return lines, {"summaries": sums, "pairs": pairs}


def best_variant(rows):
    """The conditioning variant with the highest mean success (mean over its finished seeds)."""
    scored = [(summarize(config_rows(rows, "conditioning", v))["mean_success"][0], v) for v in CONDITIONING_VARIANTS]
    scored = [(m, v) for m, v in scored if m is not None]
    return max(scored)[1] if scored else None


# --- W&B downloads (console logs, loss curves, videos) -------------------------------------


def cache_dir() -> Path:
    path = outputs_dir() / "aggregate_cache"  # gitignored
    path.mkdir(parents=True, exist_ok=True)
    return path


def download(run, name: str) -> Path | None:
    """A file of a W&B run, cached under outputs/aggregate_cache/<run id>/."""
    root = cache_dir() / run.id
    path = root / name
    if path.is_file():
        return path
    try:
        run.file(name).download(root=str(root), replace=True)
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not download {name} of {run.name}: {type(e).__name__}: {e}")
        return None
    return path if path.is_file() else None


def add_episode_data(rows, local, solved, use_logs: bool = True) -> None:
    """Per-seed strings (or console-log blocks) and the FetchPush unsolved-start counts, for finished rows."""
    for row in rows:
        if row["status"] != "finished":
            continue
        run = row.get("_run")
        eps = episodes_for(row, local, run.summary if run is not None else {})
        row["episodes_source"] = eps["source"]
        for robot in ROBOTS:
            row[f"{robot}_episodes"] = eps.get(robot)
        if eps["source"] == "none" and use_logs and run is not None and row["group"] in ("conditioning", "adaptation",
                                                                                        "baseline"):
            path = download(run, "output.log")
            if path is not None:
                blocks = parse_eval_blocks(path.read_text(errors="replace"))
                for robot, counts in blocks.items():
                    if row.get(f"{robot}_successes") is not None and sum(counts) == row[f"{robot}_successes"]:
                        row[f"{robot}_blocks"] = " ".join(map(str, counts))
        if row.get("fetch_successes") is not None:
            n_unsolved = row["fetch_n"] - len(solved)
            row["fetch_unsolved_n"] = n_unsolved
            if row["fetch_episodes"]:
                k, n = unsolved_successes(row["fetch_episodes"], solved)
                row["fetch_unsolved_successes"] = row["fetch_unsolved_low"] = row["fetch_unsolved_high"] = k
                row["fetch_unsolved_n"] = n
            else:
                row["fetch_unsolved_low"], row["fetch_unsolved_high"] = unsolved_range(
                    row["fetch_successes"], row["fetch_n"], len(solved))


def loss_curve(run, key: str = "train/loss") -> list[tuple[int, float]]:
    return [(int(h["_step"]), float(h[key])) for h in run.scan_history(keys=["_step", key]) if h.get(key) is not None]


_VIDEO_RE = re.compile(r"eval_(pusht|fetch)_seed(\d+)_(success|fail)(_random)?_")


def videos_of(run) -> list[dict]:
    out = []
    for f in run.files():
        m = _VIDEO_RE.search(f.name)
        if f.name.endswith(".mp4") and m:
            out.append({"name": f.name, "robot": m.group(1), "seed": int(m.group(2)),
                        "success": m.group(3) == "success", "random": bool(m.group(4))})
    return out


def bc_mlp_check(rows, solved) -> tuple[str, list]:
    """Did BC-MLP on FetchPush learn and level off, or stay flat at the mean action?"""
    bc = config_rows(rows, "baseline", "bc_mlp-fetch")
    rn = config_rows(rows, "baseline", "resnet18-fetch")
    if not bc or not rn:
        return "BC-MLP / ResNet-18 FetchPush runs not finished.\n", []
    bc_curves = {r["wandb_id"]: loss_curve(r["_run"]) for r in bc}
    rn_curves = {r["wandb_id"]: loss_curve(r["_run"]) for r in rn}
    val = {r["wandb_id"]: loss_curve(r["_run"], "val/loss") for r in bc + rn}
    v = loss_verdict(list(bc_curves.values()), list(rn_curves.values()))
    steps = (50, 200, 1000, 2000, 4000)
    lines = ["# BC-MLP on FetchPush: did it learn?", "",
             "Generated by `scripts/aggregate.py` from the W&B training-loss history (masked L1, normalized "
             "actions, logged every 50 steps; warmup is the first 300 steps).", "",
             "| Run | Seed | " + " | ".join(f"train loss @{s}" for s in steps)
             + " | change after warmup (400–599 → last 200) | val loss @1000…4000 |",
             "|---|---|" + "---|" * (len(steps) + 2)]
    for r in bc + rn:
        c = (bc_curves | rn_curves)[r["wandb_id"]]
        vals = " / ".join(f"{x:.4f}" for _, x in val[r["wandb_id"]])
        lines.append(f"| {r['label']} `{r['wandb_id']}` | {r['seed']} | "
                     + " | ".join(f"{loss_at(c, s):.4f}" for s in steps) + f" | {100 * curve_change(c):+.1f}% | {vals} |")
    lines += ["", f"**Verdict: {v['verdict']}.** After warmup the BC-MLP loss changes by "
              f"{', '.join(f'{100 * x:+.1f}%' for x in v['bc_changes'])} (seeds 0, 1, 2), while ResNet-18 on the same "
              f"data changes by {', '.join(f'{100 * x:+.1f}%' for x in v['contrast_changes'])}. "
              "The rule: flat if every BC-MLP seed moves less than 5% while ResNet-18 drops more than 30%.", ""]
    succ = [f"seed {r['seed']}: {r['fetch_successes']}/{r['fetch_n']}" for r in bc]
    lines.append(f"FetchPush success: {', '.join(succ)}; the floor is {FETCH_FLOOR:.0%} "
                 f"({len(KNOWN_SOLVED)} starts already solved).")
    if all(r["fetch_episodes"] for r in bc):
        for r in bc:
            wins = {SEED_START + i for i, c in enumerate(r["fetch_episodes"]) if c == "1"}
            lines.append(f"- seed {r['seed']}: wins on {sorted(wins)}; exactly the already-solved starts: "
                         f"{'yes' if wins == set(solved) else 'no'}")
    else:
        vids = []
        for r in bc:
            wins = [x["seed"] for x in videos_of(r["_run"]) if x["robot"] == "fetch" and x["success"]]
            vids.append(f"seed {r['seed']}: {wins or 'none'}")
        lines.append("Per-seed results are not in W&B for these runs; the success videos they saved are on start "
                     f"seeds {'; '.join(vids)} (already-solved starts: {sorted(solved)}).")
    lines += ["", "A model that ignores its input and always outputs the average action has this pattern: the loss "
              "falls once (to the error of the mean action) during warmup and then stays put, and it wins only "
              "where no movement is needed.", ""]
    curves = []
    for group, cs in (("bc_mlp", bc_curves), ("resnet18", rn_curves)):
        for rid, c in cs.items():
            curves += [{"model": group, "wandb_id": rid, "step": s, "train_loss": x} for s, x in c]
    return "\n".join(lines), curves


def make_gifs(rows, solved, out_dir: Path) -> list[dict]:
    """3 GIFs from the best seed of the best conditioning variant: a PushT success, a FetchPush success on a
    start that is NOT already solved, and a PushT failure. Falls back to the variant's next-best seed."""
    best = best_variant(rows)
    if best is None:
        return []
    cands = sorted(config_rows(rows, "conditioning", best), key=lambda r: -(r["mean_success"] or 0))
    wanted = [("pusht_success", lambda v: v["robot"] == "pusht" and v["success"]),
              ("fetch_success", lambda v: v["robot"] == "fetch" and v["success"] and v["seed"] not in solved),
              ("pusht_failure", lambda v: v["robot"] == "pusht" and not v["success"] and not v["random"])]
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    for name, ok in wanted:
        for rank, row in enumerate(cands):
            vids = sorted([v for v in videos_of(row["_run"]) if ok(v)], key=lambda v: (v["random"], v["seed"]))
            if not vids:
                continue
            v = vids[0]
            mp4 = download(row["_run"], v["name"])
            if mp4 is None:
                continue
            gif = out_dir / f"{name}.gif"
            mp4_to_gif(mp4, gif)
            sources.append({"gif": f"results/gifs/{gif.name}", "variant": best, "wandb_id": row["wandb_id"],
                            "run_name": row["run_name"], "seed": row["seed"], "best_seed": rank == 0,
                            "start_seed": v["seed"], "robot": v["robot"], "success": v["success"], "video": v["name"],
                            "run_pusht": f"{row['pusht_successes']}/{row['pusht_n']}",
                            "run_fetch": f"{row['fetch_successes']}/{row['fetch_n']}",
                            "run_mean_success": row["mean_success"],
                            "variant_mean_success": summarize(cands)["mean_success"][0]})
            break
        else:
            print(f"WARNING: no video found for {name}")
    (out_dir / "sources.json").write_text(json.dumps(sources, indent=2))
    return sources


def mp4_to_gif(mp4: Path, gif: Path, every: int = 2, max_width: int = 256) -> None:
    """Every 2nd frame, at most 256 px wide, played at the original speed."""
    import imageio.v2 as imageio
    from PIL import Image

    reader = imageio.get_reader(str(mp4), format="ffmpeg")
    fps = float(reader.get_meta_data().get("fps") or 10)
    frames = []
    for i, frame in enumerate(reader):
        if i % every:
            continue
        img = Image.fromarray(frame)
        if img.width > max_width:
            img = img.resize((max_width, round(img.height * max_width / img.width)), Image.LANCZOS)
        frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=128))
    reader.close()
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=round(1000 * every / fps), loop=0,
                   optimize=True)


def gif_markdown(sources) -> list[str]:
    if not sources:
        return []
    titles = {"pusht_success": "PushT success", "fetch_success": "FetchPush success", "pusht_failure": "PushT failure"}
    s0 = sources[0]
    lines = ["### Rollouts", "",
             f"These GIFs come from the **best seed of the best variant**: `{s0['run_name']}` (W&B `{s0['wandb_id']}`), "
             f"seed {s0['seed']} of {s0['variant']}. They are an illustration, not a typical run: this run scored PushT "
             f"{s0['run_pusht']} and FetchPush {s0['run_fetch']} (mean {_pct(s0.get('run_mean_success'))}), while "
             f"{s0['variant']} averages {_pct(s0.get('variant_mean_success'))} over its seeds. The FetchPush success is "
             "on a start that is not already solved.", ""]
    for s in sources:
        name = Path(s["gif"]).stem
        where = ("best seed of the best variant" if s["best_seed"] else
                 f"from `{s['run_name']}` (W&B `{s['wandb_id']}`), seed {s['seed']} of the best variant, because the "
                 "best seed saved no such video")
        lines += [f"**{titles.get(name, name)}**, start seed {s['start_seed']} ({where}; illustration, not a typical "
                  "run)", "", f"![{titles.get(name, name)}]({s['gif']})", ""]
    return lines


# --- main ----------------------------------------------------------------------------------


def write_csv(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in CSV_FIELDS})


def status_lines(rows) -> list[str]:
    planned = [r for r in rows if r["group"] != "other" and r["status"] != "duplicate"]
    lines = []
    for group in ("conditioning", "baseline", "adaptation"):
        g = [r for r in planned if r["group"] == group]
        counts = {}
        for r in g:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        lines.append(f"- {group}: " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())) + f" (of {len(g)})")
    return lines


def main(argv=None) -> None:
    global ADAPT_STEPS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes-dir", default=str(REPO_ROOT / "outputs" / "kaggle"),
                        help="folder with downloaded Kaggle run folders (each with metrics.json)")
    parser.add_argument("--adapt-steps", type=int, default=ADAPT_STEPS)
    parser.add_argument("--no-gifs", action="store_true", help="skip downloading videos and making GIFs")
    parser.add_argument("--no-start-check", action="store_true", help="use the known solved FetchPush starts "
                        "instead of recomputing them with fetch_start_check.py")
    parser.add_argument("--no-logs", action="store_true", help="do not download console logs (no approx. bootstrap)")
    args = parser.parse_args(argv)

    load_dotenv_if_local()
    import wandb

    api = wandb.Api(timeout=60)
    project = f"{api.default_entity}/{WANDB_PROJECT}"
    runs = list(api.runs(project))
    print(f"W&B project {project}: {len(runs)} runs")
    rows = match_runs(runs, expected_runs(args.adapt_steps))
    solved = solved_starts(not args.no_start_check)
    local = load_local_metrics(args.episodes_dir)
    print(f"Local metrics.json files in {args.episodes_dir}: {len(local)}")
    ADAPT_STEPS = args.adapt_steps  # table titles use it
    add_episode_data(rows, local, solved, use_logs=not args.no_logs)

    RESULTS_DIR.mkdir(exist_ok=True)
    write_csv(rows, RESULTS_DIR / "results.csv")

    t1, cond = table_conditioning(rows)
    t2 = table_fetch(rows)
    t3, base_pairs = table_baselines(rows)
    t4, adapt = table_adaptation(rows)
    bc_text, curves = bc_mlp_check(rows, solved)
    (RESULTS_DIR / "bc_mlp_fetch.md").write_text(bc_text)
    if curves:
        with (RESULTS_DIR / "bc_mlp_fetch_loss.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", "wandb_id", "step", "train_loss"])
            w.writeheader()
            w.writerows(curves)

    sources = [] if args.no_gifs else make_gifs(rows, solved, RESULTS_DIR / "gifs")
    if args.no_gifs and (RESULTS_DIR / "gifs" / "sources.json").is_file():
        sources = json.loads((RESULTS_DIR / "gifs" / "sources.json").read_text())

    import plots

    plot_paths = plots.make_all(RESULTS_DIR / "results.csv", RESULTS_DIR / "plots",
                                RESULTS_DIR / "bc_mlp_fetch_loss.csv")

    n_exact = sum(r["episodes_source"] in ("metrics.json", "wandb") for r in rows if r["status"] == "finished")
    n_fin = sum(r["status"] == "finished" for r in rows)
    header = [f"Generated by `scripts/aggregate.py` on {time.strftime('%Y-%m-%d %H:%M')} from W&B project "
              f"`{WANDB_PROJECT}`. Every row lists the W&B run IDs it was computed from; all per-run numbers, "
              "commits and links are in [results/results.csv](results/results.csv). Success rates are over 50 "
              "rollouts per robot per run from the fixed start seeds 100000–100049. Mean ± std is over training "
              f"seeds. Per-seed results available for {n_exact} of {n_fin} finished runs.", "",
              "Run status:"] + status_lines(rows) + [""]
    tables = header + t1 + t2 + t3 + t4
    (RESULTS_DIR / "tables.md").write_text("# Results tables\n\n" + "\n".join(tables).replace("](results/", "](")
                                           + "\n")
    plot_md = ["### Plots", ""] + [f"![{p.stem}](results/plots/{p.name})" for p in plot_paths] + [""]
    readme_block = (["## Results", ""] + tables + plot_md + gif_markdown(sources)
                    + ["Findings in plain language: [results/findings.md](results/findings.md). "
                       "BC-MLP on FetchPush: [results/bc_mlp_fetch.md](results/bc_mlp_fetch.md)."])
    README.write_text(replace_readme_block(README.read_text(), "\n".join(readme_block)))

    # --- headline ---
    print("\n=== Run status ===")
    print("\n".join(status_lines(rows)))
    print("\n=== Conditioning (mean ± std over seeds) ===")
    for v, s in cond["summaries"].items():
        p, f = s.get("pusht"), s.get("fetch")
        print(f"{v:10s} {_seeds(s)} seeds | PushT {_ms(*p['mean_std']) if p else '-'} | "
              f"FetchPush {_ms(*f['mean_std']) if f else '-'} | mean {_ms(*s['mean_success'])}")
    for name, cmp in cond["pairs"] + base_pairs + adapt["pairs"]:
        print(f"  {name:28s} " + " | ".join(f"{r}: {_gap(c)} ({c['method'] if c else '-'})" for r, c in cmp.items()))
    print("\n" + "\n".join(t2[-(len(_fetch_configs(rows)) + 3):-2]))
    print("\n" + "\n".join(l for l in t4 if l.startswith("| ") or l.startswith("LoRA") or l.startswith("Not")))
    print("\nBC-MLP FetchPush: " + next((l for l in bc_text.splitlines() if l.startswith("**Verdict")), "not checked"))
    print(f"\nWrote {RESULTS_DIR / 'results.csv'} ({len(rows)} rows), tables.md, bc_mlp_fetch.md, "
          f"{len(sources)} GIFs, {len(plot_paths)} plots, and the README results block.")


if __name__ == "__main__":
    main()
