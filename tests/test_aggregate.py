"""Phase 7 checks: scripts/aggregate.py (results CSV, statistics, README tables). Written before the code.

Gate (docs/spec.md, Phase 7): every README number traces to a run ID. These checks use fake W&B run
objects, so they need no network and no W&B key.
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vla.train import run_key
from vla.utils.config import load_config

REPO = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agg = _load("aggregate")


def fake_run(name, key, state="finished", job_type="train", debug=False, created="2026-11-25T10:00:00",
             summary=None, run_id=None, robots=("pusht", "fetch")):
    s = {"eval/mean_success": 0.3, "git_commit": "abcdef1234567", "total_minutes": 30.0}
    for r in robots:
        s.update({f"eval/{r}_successes": 5, f"eval/{r}_n": 50, f"eval/{r}_success": 0.1})
    if summary is not None:
        s.update(summary)
    return SimpleNamespace(
        id=run_id or name[:8], name=name, state=state, job_type=job_type, created_at=created, group=None,
        url=f"https://wandb.ai/x/vla-multiembodiment/runs/{run_id or name[:8]}",
        config={"run": {"key": key, "debug_data": debug, "git_commit": "abcdef1234567"}}, summary=s)


# --- which runs count --------------------------------------------------------------------


def test_eligible_filter():
    good = fake_run("film-frozen-s0-75a68d5", "film-frozen-full-s0-4000steps")
    assert agg.is_eligible(good)
    assert not agg.is_eligible(fake_run("smoke-film-frozen-s0-x", "film-frozen-full-s0-200steps"))
    assert not agg.is_eligible(fake_run("film-frozen-s0-y", "film-frozen-full-s0-4000steps", debug=True))
    assert not agg.is_eligible(fake_run("film-frozen-s0-z", "film-frozen-full-s0-4000steps", state="running"))
    assert not agg.is_eligible(fake_run("hello-9ee7303", None, job_type="hello"))
    unevaluated = fake_run("film-frozen-s1-a", "film-frozen-full-s1-4000steps")
    del unevaluated.summary["eval/mean_success"]
    assert not agg.is_eligible(unevaluated)


def test_expected_matrix_matches_run_key():
    expected = agg.expected_runs(adapt_steps=2500)
    keys = [e["key"] for e in expected]
    assert len(keys) == 30 and len(set(keys)) == 30
    assert sum(e["group"] == "conditioning" for e in expected) == 9
    assert sum(e["group"] == "baseline" for e in expected) == 12
    assert sum(e["group"] == "adaptation" for e in expected) == 9
    for e in expected:  # the names come from the training code itself, so they can never drift apart
        assert e["key"] == run_key(load_config(e["overrides"]))
    assert "sep_heads-lora16-full-s2-2500steps" in keys
    assert "bc_mlp-fetch-full-s1-4000steps" in keys
    assert "token-frozen-full-s0-4000steps" in keys


def test_match_marks_missing_running_duplicates_and_extras():
    expected = agg.expected_runs(adapt_steps=2500)
    runs = [
        fake_run("film-frozen-s0-old", "film-frozen-full-s0-4000steps", created="2026-11-25T09:00:00", run_id="old1"),
        fake_run("film-frozen-s0-new", "film-frozen-full-s0-4000steps", created="2026-11-25T12:00:00", run_id="new1"),
        fake_run("sep_heads-lora16-s0-bd", "sep_heads-lora16-full-s0-2500steps", state="running", run_id="run1"),
        fake_run("film-frozen-s0-8000", "film-frozen-full-s0-8000steps", run_id="xtra"),
        fake_run("smoke-film-frozen-s0", "film-frozen-full-s0-200steps", run_id="smk1"),
    ]
    rows = agg.match_runs(runs, expected)
    by_key = {}
    for r in rows:
        by_key.setdefault(r["run_key"], []).append(r)
    film = by_key["film-frozen-full-s0-4000steps"]
    used = [r for r in film if r["status"] == "finished"]
    assert len(used) == 1 and used[0]["wandb_id"] == "new1"  # the newest wins
    assert [r["wandb_id"] for r in film if r["status"] == "duplicate"] == ["old1"]
    assert by_key["sep_heads-lora16-full-s0-2500steps"][0]["status"] == "running"
    assert by_key["sep_heads-lora16-full-s0-2500steps"][0]["wandb_id"] == "run1"
    assert by_key["token-frozen-full-s1-4000steps"][0]["status"] == "missing"
    assert by_key["film-frozen-full-s0-8000steps"][0]["group"] == "other"
    assert "film-frozen-full-s0-200steps" not in by_key  # smoke runs never appear
    expected_keys = {e["key"] for e in expected}
    assert expected_keys <= set(by_key)


def test_row_fields_from_summary():
    run = fake_run("film-frozen-s0-75a68d5", "film-frozen-full-s0-4000steps", run_id="i0senai6",
                   summary={"eval/pusht_successes": 5, "eval/fetch_successes": 32, "eval/fetch_success": 0.64,
                            "eval/pusht_mean_max_coverage": 0.47, "peak_gpu_mem_gb_at_200": 1.5})
    row = agg.match_runs([run], agg.expected_runs(2500))
    row = next(r for r in row if r["wandb_id"] == "i0senai6")
    assert row["pusht_successes"] == 5 and row["fetch_successes"] == 32 and row["fetch_n"] == 50
    assert row["fetch_success"] == pytest.approx(0.64)
    assert row["pusht_mean_max_coverage"] == pytest.approx(0.47)
    assert row["git_commit"] == "abcdef1"
    assert row["cond"] == "film" and row["seed"] == 0 and row["label"] == "film"


# --- statistics ---------------------------------------------------------------------------


def test_mean_std_and_pooled_wilson():
    m, s = agg.mean_std([0.1, 0.0, 0.12])
    assert m == pytest.approx(0.0733333, abs=1e-6) and s == pytest.approx(np.std([0.1, 0.0, 0.12], ddof=1))
    assert agg.mean_std([0.5]) == (0.5, 0.0)
    assert agg.mean_std([]) == (None, None)
    k, n, lo, hi = agg.pooled_wilson([5, 0, 6], [50, 50, 50])
    assert (k, n) == (11, 150)
    from vla.evaluate import wilson_interval

    assert (lo, hi) == pytest.approx(wilson_interval(11, 150))


def test_paired_bootstrap_identical_and_different():
    rng = np.random.default_rng(1)
    a = (rng.random((3, 50)) < 0.5).astype(float)
    same = agg.paired_bootstrap(a, a.copy())
    assert same["gap_pp"] == 0 and same["lo_pp"] <= 0 <= same["hi_pp"]
    b = np.zeros((3, 50))
    hi = np.ones((3, 50))
    hi[:, :10] = 0  # 80% vs 40%: a 40 pp gap
    lo = np.zeros((3, 50))
    lo[:, :20] = 1
    diff = agg.paired_bootstrap(hi, lo)
    assert diff["gap_pp"] == pytest.approx(40.0)
    assert diff["lo_pp"] > 0 and diff["p_le_0"] < 0.01
    again = agg.paired_bootstrap(hi, lo)
    assert again == diff  # fixed rng seed: the same numbers every run
    assert agg.paired_bootstrap(b, b)["gap_pp"] == 0


def test_paired_bootstrap_unequal_seed_counts():
    a = np.ones((2, 50))
    b = np.zeros((3, 50))
    assert agg.paired_bootstrap(a, b)["gap_pp"] == pytest.approx(100.0)


def test_unsolved_starts_exact_and_range():
    solved = {100004, 100010, 100021, 100036, 100049}
    episodes = "".join("1" if (100000 + i) in solved or i < 20 else "0" for i in range(50))
    k, n = agg.unsolved_successes(episodes, solved)
    assert n == 45 and k == 20 - 2  # seeds 100004 and 100010 are inside the first 20 and are solved
    assert agg.unsolved_range(32, n_total=50, n_solved=5) == (27, 32)
    assert agg.unsolved_range(3, n_total=50, n_solved=5) == (0, 3)
    assert agg.unsolved_range(50, n_total=50, n_solved=5) == (45, 45)


def test_parse_eval_blocks():
    log = "\n".join([
        "step 4000 | VAL loss 0.08",
        "  pusht: 10/50 episodes, 1 successes, 22s",
        "  pusht: 20/50 episodes, 1 successes, 42s",
        "  pusht: 30/50 episodes, 2 successes, 61s",
        "  pusht: 40/50 episodes, 4 successes, 79s",
        "  pusht: 50/50 episodes, 5 successes, 97s",
        "  fetch: 10/50 episodes, 8 successes, 10s",
        "  fetch: 20/50 episodes, 14 successes, 21s",
        "  fetch: 30/50 episodes, 21 successes, 31s",
        "  fetch: 40/50 episodes, 27 successes, 41s",
        "  fetch: 50/50 episodes, 32 successes, 51s",
    ])
    blocks = agg.parse_eval_blocks(log)
    assert blocks["pusht"] == [1, 0, 1, 2, 1]
    assert blocks["fetch"] == [8, 6, 7, 6, 5]
    assert agg.parse_eval_blocks("no evaluation here") == {}


def test_episodes_from_local_metrics(tmp_path):
    run_dir = tmp_path / "kaggle" / "outputs" / "film-frozen-s0-75a68d5"
    run_dir.mkdir(parents=True)
    eps = lambda pattern: [{"seed": 100000 + i, "success": c == "1"} for i, c in enumerate(pattern)]
    pusht = "1" * 5 + "0" * 45
    fetch = "1" * 32 + "0" * 18
    (run_dir / "metrics.json").write_text(json.dumps({
        "run_id": "film-frozen-s0-75a68d5",
        "eval": {"pusht": {"successes": 5, "n": 50, "episodes": eps(pusht)},
                 "fetch": {"successes": 32, "n": 50, "episodes": eps(fetch)}}}))
    bad_dir = tmp_path / "kaggle" / "outputs" / "token-frozen-s0-75a68d5"
    bad_dir.mkdir()
    (bad_dir / "metrics.json").write_text(json.dumps({
        "run_id": "token-frozen-s0-75a68d5",
        "eval": {"pusht": {"successes": 9, "n": 50, "episodes": eps("1" * 9 + "0" * 41)}}}))
    local = agg.load_local_metrics(tmp_path / "kaggle")
    assert set(local) == {"film-frozen-s0-75a68d5", "token-frozen-s0-75a68d5"}

    good = {"run_name": "film-frozen-s0-75a68d5", "pusht_successes": 5, "pusht_n": 50,
            "fetch_successes": 32, "fetch_n": 50}
    got = agg.episodes_for(good, local, summary={})
    assert got == {"pusht": pusht, "fetch": fetch, "source": "metrics.json"}

    mismatch = {"run_name": "token-frozen-s0-75a68d5", "pusht_successes": 2, "pusht_n": 50,
                "fetch_successes": 30, "fetch_n": 50}
    assert agg.episodes_for(mismatch, local, summary={})["source"] == "none"  # totals disagree: ignored

    from_wandb = {"run_name": "x", "pusht_successes": 1, "pusht_n": 50, "fetch_successes": None, "fetch_n": None}
    summary = {"eval/pusht_episode_success": "1" + "0" * 49}
    assert agg.episodes_for(from_wandb, {}, summary=summary) == {"pusht": "1" + "0" * 49, "source": "wandb"}


# --- README and the BC-MLP check ---------------------------------------------------------


def test_readme_block_replacement_is_idempotent():
    readme = "# Title\n\nIntro.\n\n## Setup\nsteps\n"
    once = agg.replace_readme_block(readme, "TABLE v1")
    assert "TABLE v1" in once and once.startswith("# Title\n\nIntro.") and "## Setup\nsteps" in once
    twice = agg.replace_readme_block(once, "TABLE v2")
    assert "TABLE v2" in twice and "TABLE v1" not in twice
    assert twice.count(agg.README_START) == 1 and twice.count(agg.README_END) == 1
    assert agg.replace_readme_block(twice, "TABLE v2") == twice


def _curve(values, start=50, every=50):
    return [(start + i * every, v) for i, v in enumerate(values)]


def test_loss_verdict_flat_vs_learned():
    flat = [_curve([0.118, 0.11, 0.10, 0.098] + [0.099, 0.100] * 38)]  # drops during warmup, then flat
    learning = [_curve([0.13, 0.10] + list(np.linspace(0.095, 0.045, 78)))]
    v = agg.loss_verdict(flat, learning)
    assert v["verdict"] == "flat"
    assert abs(v["bc_change"]) < 0.05 and v["contrast_change"] < -0.30
    v2 = agg.loss_verdict(learning, learning)
    assert v2["verdict"] == "learned"


def test_log_eval_writes_episode_string():
    from vla.evaluate import log_eval

    class FakeWB:
        def __init__(self):
            self.summary, self.logged = {}, {}

        def log(self, d):
            self.logged.update(d)

    result = {"mean_success": 0.5, "eval_seconds": 1.0}
    for robot, pattern in (("pusht", "100"), ("fetch", "011")):
        eps = [{"seed": 100000 + i, "success": c == "1", "steps": 1, "max_coverage": None}
               for i, c in enumerate(pattern)]
        k = pattern.count("1")
        result[robot] = {"n": 3, "successes": k, "success_rate": k / 3, "wilson_low": 0, "wilson_high": 1,
                         "seconds": 1.0, "episodes": eps, "videos": []}
    wb = FakeWB()
    log_eval(wb, result)
    assert wb.summary["eval/pusht_episode_success"] == "100"
    assert wb.summary["eval/fetch_episode_success"] == "011"
