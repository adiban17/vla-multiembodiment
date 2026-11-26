"""Phase 7 plots, drawn from results/results.csv (written by scripts/aggregate.py).

    .venv/bin/python scripts/plots.py        # redraw from the existing CSV files

1. conditioning_success.png   success per conditioning variant, one bar per robot (mean over seeds,
                              error bar = std over seeds, dots = single seeds), with the FetchPush 10% floor
2. adaptation_memory.png      mean success vs peak GPU memory at step 200 for frozen_matched / lora16 / full_ft
3. bc_mlp_fetch_loss.png      training loss of BC-MLP vs ResNet-18 on FetchPush (did BC-MLP learn?)

Colors: the first three slots of the reference categorical palette (blue, orange, aqua), which stay
distinguishable for color-blind readers; every series is also named in a legend or a direct label.
"""

import csv
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated all-pairs)
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
CONDITIONING = ("token", "film", "sep_heads")
ADAPT = ("frozen_matched", "lora16", "full_ft")
FETCH_FLOOR = 0.10


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(w=7.0, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    return fig, ax


def _title(ax, title, subtitle, width=88):
    sub = textwrap.fill(subtitle, width)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=10 + 12 * (sub.count("\n") + 1))
    ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK2, fontsize=8.5, va="bottom")


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_rows(csv_path) -> list[dict]:
    with open(csv_path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["status"] == "finished"]


def conditioning_plot(rows, out: Path) -> Path:
    fig, ax = _fig()
    robots = [("pusht", "PushT"), ("fetch", "FetchPush")]
    width, gap = 0.36, 0.02
    x = np.arange(len(CONDITIONING))
    for j, (robot, name) in enumerate(robots):
        offset = (j - 0.5) * (width + gap)
        for i, variant in enumerate(CONDITIONING):
            vals = [_num(r[f"{robot}_success"]) for r in rows
                    if r["group"] == "conditioning" and r["label"] == variant and _num(r[f"{robot}_success"]) is not None]
            if not vals:
                continue
            m = float(np.mean(vals))
            s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            ax.bar(x[i] + offset, 100 * m, width, color=SERIES[j], edgecolor=SURFACE, linewidth=2,
                   label=name if i == 0 else None, zorder=2)
            ax.errorbar(x[i] + offset, 100 * m, yerr=100 * s, color=INK, linewidth=1.2, capsize=4, zorder=3)
            ax.scatter(np.full(len(vals), x[i] + offset), [100 * v for v in vals], s=14, color=SURFACE,
                       edgecolor=INK, linewidth=0.9, zorder=4)
            ax.text(x[i] + offset, 100 * (m + s) + 2.5, f"{100 * m:.0f}%", ha="center", color=INK2, fontsize=8.5)
    ax.axhline(100 * FETCH_FLOOR, color=SERIES[1], linestyle=(0, (4, 3)), linewidth=1.2, zorder=1,
               label="FetchPush floor: 10% (5/50 starts already solved)")
    ax.set_xticks(x, CONDITIONING, color=INK, fontsize=10)
    ax.set_ylabel("success rate (%)", color=INK2, fontsize=9)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK)
    _title(ax, "Success by conditioning variant",
           "Frozen encoders, 4,000 steps. Bar = mean over 3 seeds, error bar = std over seeds, dots = single seeds "
           "(50 rollouts each).")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def adaptation_plot(rows, out: Path) -> Path:
    fig, ax = _fig()
    drawn, max_mem = 0, 0.0
    for j, mode in enumerate(ADAPT):
        pts = [(_num(r["peak_gpu_mem_gb_at_200"]), _num(r["mean_success"])) for r in rows
               if r["group"] == "adaptation" and r["label"] == mode]
        pts = [(m, s) for m, s in pts if m is not None and s is not None]
        if not pts:
            continue
        drawn += 1
        mem, succ = np.array(pts).T
        max_mem = max(max_mem, float(mem.max()))
        ax.scatter(mem, 100 * succ, s=22, color=SERIES[j], alpha=0.35, linewidth=0, zorder=2)
        mm, ms = mem.mean(), 100 * succ.mean()
        sd = 100 * succ.std(ddof=1) if len(succ) > 1 else 0.0
        ax.errorbar(mm, ms, yerr=sd, fmt="o", markersize=8, color=SERIES[j], markeredgecolor=SURFACE,
                    markeredgewidth=2, ecolor=SERIES[j], elinewidth=1.4, capsize=4, zorder=3, label=mode)
        ax.annotate(f"{mode}  ({len(pts)}/3 seeds)", (mm, ms), xytext=(10, 6), textcoords="offset points",
                    color=INK, fontsize=9)
    if not drawn:
        ax.text(0.5, 0.5, "No adaptation runs finished yet.\nRerun scripts/aggregate.py later.",
                transform=ax.transAxes, ha="center", va="center", color=INK2, fontsize=11)
        ax.set_xlim(0, 1)
    else:
        ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK)
        ax.set_xlim(0, 1.3 * max_mem)  # room for the direct labels
    ax.set_ylim(0, 100)
    ax.set_xlabel("peak GPU memory at training step 200 (GB allocated)", color=INK2, fontsize=9)
    ax.set_ylabel("mean success, PushT and FetchPush (%)", color=INK2, fontsize=9)
    _title(ax, "Success vs GPU memory by adaptation mode",
           "sep_heads, 2,500 steps, batch 64, T4. Big marker = mean over finished seeds (± std); faint = single seeds.")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def bc_loss_plot(curve_csv: Path, out: Path) -> Path | None:
    if not Path(curve_csv).is_file():
        return None
    curves = {}
    with open(curve_csv, newline="") as f:
        for r in csv.DictReader(f):
            curves.setdefault((r["model"], r["wandb_id"]), []).append((int(r["step"]), float(r["train_loss"])))
    fig, ax = _fig()
    names = {"bc_mlp": ("BC-MLP", SERIES[0]), "resnet18": ("ResNet-18", SERIES[1])}
    seen = set()
    for (model, rid), pts in sorted(curves.items()):
        pts.sort()
        steps, loss = zip(*pts)
        name, color = names.get(model, (model, MUTED))
        ax.plot(steps, loss, color=color, linewidth=1.6, alpha=0.9, label=None if model in seen else name)
        seen.add(model)
    ax.axvspan(0, 300, color=GRID, alpha=0.6, linewidth=0, zorder=0)
    ax.text(150, ax.get_ylim()[1], "warmup", ha="center", va="top", color=INK2, fontsize=8)
    ax.set_xlabel("training step", color=INK2, fontsize=9)
    ax.set_ylabel("training loss (masked L1, normalized)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    _title(ax, "FetchPush training loss: BC-MLP stays flat after warmup",
           "3 seeds each, logged every 50 steps. ResNet-18 keeps learning on the same data; BC-MLP does not.")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def make_all(results_csv, out_dir, curve_csv=None) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(results_csv)
    paths = [conditioning_plot(rows, out_dir / "conditioning_success.png"),
             adaptation_plot(rows, out_dir / "adaptation_memory.png")]
    if curve_csv is not None:
        p = bc_loss_plot(curve_csv, out_dir / "bc_mlp_fetch_loss.png")
        if p is not None:
            paths.append(p)
    return paths


if __name__ == "__main__":
    results = REPO / "results"
    for p in make_all(results / "results.csv", results / "plots", results / "bc_mlp_fetch_loss.csv"):
        print(f"Wrote {p}")
