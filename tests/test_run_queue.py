"""Phase 4 checks: the run queue (scripts/run_queue.py) and the Phase 4 Kaggle jobs. Written before the code.

The queue must: skip runs already finished, keep going when one run crashes, run two at a time,
give each run 2 data-loader workers when two share the machine, and print a final table.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from conftest import fake_encode, fake_language
from vla.data.cache_features import build_cache
from vla.train import run_key
from vla.utils.config import load_config, to_dict

REPO = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rq = _load("run_queue")


# --- planning: keys and worker setting --------------------------------------------------


def test_plan_jobs_keys_and_workers():
    jobs = rq.plan_jobs([["cond=token", "seed=1"], ["cond=film"]], workers_per_run=2, extra=["train.steps=100"])
    assert [j["key"] for j in jobs] == ["token-frozen-full-s1-100steps", "film-frozen-full-s0-100steps"]
    for job in jobs:
        assert "data.num_workers=2" in job["overrides"] and "train.steps=100" in job["overrides"]
        assert load_config(job["overrides"]).data.num_workers == 2
    # no worker setting: the config's own value (4) is kept
    (job,) = rq.plan_jobs([["cond=film"]], workers_per_run=None)
    assert load_config(job["overrides"]).data.num_workers == 4
    assert rq.default_workers_per_run(slots=2) == 2 and rq.default_workers_per_run(slots=1) is None


def test_bad_config_is_a_failed_row_not_a_crash():
    (job,) = rq.plan_jobs([["cond=flim"]], workers_per_run=None)
    assert job["key"] is None and "flim" in job["error"]


# --- running: skip, crash, keep going -------------------------------------------------------


def _fake_command(tmp_path, crash_keys):
    """Stand-in for `python -m vla.train`: writes a run folder with metrics.json, or exits 1."""

    def command_for(job, result_file, slot):
        run_dir = tmp_path / "runs" / job["key"]
        metrics = {"steps_per_sec": 2.0, "pusht_success": 0.5, "fetch_success": 0.25, "mean_success": 0.375,
                   "val_loss": 0.1, "total_minutes": 1.0, "pusht_successes": 25, "fetch_successes": 12,
                   "pusht_n": 50, "fetch_n": 50}
        code = "import sys; print('crashing on purpose'); sys.exit(1)" if job["key"] in crash_keys else (
            "import json, pathlib, sys\n"
            f"d = pathlib.Path({str(run_dir)!r}); d.mkdir(parents=True)\n"
            f"(d / 'metrics.json').write_text(json.dumps({metrics!r}))\n"
            f"pathlib.Path({str(result_file)!r}).write_text(json.dumps({{'run_dir': str(d)}}))\n"
            f"print('workers', {job['overrides']!r})\n")
        return [sys.executable, "-c", code]

    return command_for


def test_queue_skips_finished_and_survives_a_crash(tmp_path, capsys):
    configs = [["cond=film", "seed=0"], ["cond=token", "seed=0"], ["cond=sep_heads", "seed=0"], ["cond=film", "seed=1"]]
    jobs = rq.plan_jobs(configs, workers_per_run=2)
    finished_key, crash_key = jobs[0]["key"], jobs[1]["key"]
    looked_up = []

    def is_finished(key):
        looked_up.append(key)
        return {"run": "film-frozen-s0-abc1234", "commit": "abc1234", "mean_success": 0.4} if key == finished_key else None

    rows = rq.run_queue(configs, slots=2, workers_per_run=2, is_finished=is_finished,
                        command_for=_fake_command(tmp_path, {crash_key}), queue_dir=tmp_path / "queue")
    status = {r["key"]: r["status"] for r in rows}
    assert status == {finished_key: "skipped", crash_key: "failed",
                      jobs[2]["key"]: "finished", jobs[3]["key"]: "finished"}
    assert not (tmp_path / "runs" / finished_key).exists()  # never started
    assert sorted(looked_up) == sorted(j["key"] for j in jobs)
    done = next(r for r in rows if r["key"] == jobs[2]["key"])
    assert done["mean_success"] == pytest.approx(0.375) and done["slot"] in (0, 1)

    out = capsys.readouterr().out
    assert "data-loader workers per run: 2" in out
    for job in jobs:
        assert job["key"] in out  # every run appears in the final table
    assert "crashing on purpose" in out  # the run's output is streamed with a prefix
    log = (tmp_path / "queue" / "logs" / f"{crash_key}.log").read_text()
    assert "crashing on purpose" in log
    saved = json.loads((tmp_path / "queue" / "queue_results.json").read_text())
    assert saved["workers_per_run"] == 2 and len(saved["rows"]) == 4

    rows = rq.run_queue(configs[:1], slots=1, rerun=True, is_finished=is_finished,
                        command_for=_fake_command(tmp_path / "again", set()), queue_dir=tmp_path / "queue2")
    assert rows[0]["status"] == "finished"  # --rerun ignores W&B


def test_queue_runs_real_tiny_trainings(tmp_path, synthetic_data, monkeypatch):
    """Two real tiny trainings as subprocesses, two at a time on CPU, plus one that crashes."""
    build_cache(tmp_path / "inputs" / "cache", synthetic_data, fake_encode, fake_language)
    monkeypatch.setenv("VLA_OUTPUTS_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("VLA_INPUTS_DIR", str(tmp_path / "inputs"))
    monkeypatch.setenv("WANDB_MODE", "disabled")
    tiny = ["size=tiny", "train.steps=4", "train.val_every=2", "train.log_every=2", "eval.enabled=false"]
    configs = [[*tiny, "seed=0"], [*tiny, "seed=1"], [*tiny, "seed=2", "data.cache_dir=/does/not/exist"]]
    rows = rq.run_queue(configs, slots=2, workers_per_run=0, allow_dirty=True, is_finished=lambda key: None,
                        queue_dir=tmp_path / "queue")
    assert [r["status"] for r in rows] == ["finished", "finished", "failed"]
    for r in rows[:2]:
        metrics = json.loads((Path(r["run_dir"]) / "metrics.json").read_text())
        assert r["val_loss"] == pytest.approx(metrics["val_loss"])
        config = (Path(r["run_dir"]) / "config.yaml").read_text()
        assert "num_workers: 0" in config  # the queue's worker setting reached the run


# --- Kaggle jobs ----------------------------------------------------------------


def test_phase4_jobs_registered_and_batch():
    kj = _load("kaggle_job")
    assert {"eval_sanity", "batch_conditioning"} <= set(kj.JOBS)
    configs = kj.conditioning_configs()
    assert configs == [[f"cond={c}", f"seed={s}"] for s in (0, 1, 2) for c in ("token", "film", "sep_heads")]
    assert kj.BATCH_WORKERS_PER_RUN == 2


def test_eval_sanity_film_run_is_the_batch_film_s0_run():
    """eval_sanity's FiLM run must match the queue's film-s0 exactly, so the batch skips it later."""
    kj = _load("kaggle_job")
    batch = rq.plan_jobs(kj.conditioning_configs(), workers_per_run=kj.BATCH_WORKERS_PER_RUN)
    film_s0 = next(j for j in batch if j["key"].startswith("film-") and "-s0-" in j["key"])
    sanity_cfg = load_config(kj.eval_sanity_film_overrides())
    assert run_key(sanity_cfg) == film_s0["key"] == "film-frozen-full-s0-4000steps"
    assert to_dict(sanity_cfg) == to_dict(load_config(film_s0["overrides"]))
