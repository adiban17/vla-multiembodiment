"""Phase 5 checks: the two per-robot baselines (BC-MLP and ResNet-18 BC). Written before the code.

Gate (docs/spec.md, Phase 5): both baselines overfit one batch; parameter counts printed; they use
the same chunks, loss, normalization and evaluation as the main model; the queue builds the 12
baseline runs. Plus the FetchPush "already solved at the start" check (scripts/fetch_start_check.py).

Everything runs on the Mac CPU. ResNet-18 uses random weights here (no download); Kaggle runs use
the ImageNet weights.
"""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import fake_encode, fake_language
from vla.data.cache_features import build_cache
from vla.data.dataset import WindowDataset
from vla.data.normalize import Normalizer
from vla.models.encoders import IMAGENET_MEAN, IMAGENET_STD, preprocess_image
from vla.models.policy import count_params
from vla.train import make_optimizer, make_scheduler, run_key, train
from vla.utils.config import load_config

REPO = Path(__file__).resolve().parents[1]
BASELINES = ("bc_mlp", "resnet18")
ROBOTS = ("pusht", "fetch")
IMAGE = {"pusht": 96, "fetch": 128}  # native camera sizes
EMB = {"pusht": 0, "fetch": 1}


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- config ---------------------------------------------------------------------


def test_baseline_configs():
    mlp = load_config(["baseline=bc_mlp", "robot=pusht"])
    assert (mlp.baseline, mlp.robot) == ("bc_mlp", "pusht")
    assert mlp.data.mode == "pixels" and mlp.data.batch_size == 128
    assert mlp.optim.lr == pytest.approx(1e-3) and mlp.bc.image_size == 64 and mlp.bc.hidden == 512
    assert mlp.train.steps == 4000 and mlp.optim.warmup_steps == 300  # same budget as the main model

    res = load_config(["baseline=resnet18", "robot=fetch"])
    assert res.data.mode == "pixels" and res.data.batch_size == 128
    assert res.optim.lr == pytest.approx(1e-4) and res.bc.image_size == 224 and res.bc.pretrained

    tiny = load_config(["baseline=resnet18", "robot=fetch", "size=tiny"])
    assert tiny.data.batch_size == 8 and not tiny.bc.pretrained  # the Mac never downloads weights

    vla = load_config()
    assert (vla.baseline, vla.robot) == ("none", "both")


def test_baseline_config_errors():
    with pytest.raises(ValueError, match="robot=pusht"):
        load_config(["baseline=bc_mlp"])  # a baseline trains on ONE robot
    with pytest.raises(ValueError, match="both"):
        load_config(["robot=pusht"])  # the VLA trains on both
    with pytest.raises(ValueError, match="resnet18"):
        load_config(["baseline=resnet"])  # lists the valid choices
    with pytest.raises(ValueError, match="robot"):
        load_config(["baseline=bc_mlp", "robot=ur5"])


def test_run_keys():
    assert run_key(load_config(["baseline=resnet18", "robot=fetch", "seed=1"])) == "resnet18-fetch-full-s1-4000steps"
    assert run_key(load_config(["baseline=bc_mlp", "robot=pusht"])) == "bc_mlp-pusht-full-s0-4000steps"
    # the main model's keys do not change, so finished conditioning runs are still skipped
    assert run_key(load_config(["cond=film"])) == "film-frozen-full-s0-4000steps"


# --- preprocessing: still one function ------------------------------------------------


@pytest.mark.parametrize("size", [96, 128])
def test_preprocess_image_224_is_unchanged(size):
    """The DINOv2 path (224) must give exactly the bytes it gave before the size argument existed."""
    batch = np.random.default_rng(0).integers(0, 256, size=(2, size, size, 3), dtype=np.uint8)
    x = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False).clamp_(0.0, 1.0)
    old = (x - torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)) / torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    torch.testing.assert_close(preprocess_image(batch), old, rtol=0, atol=0)
    torch.testing.assert_close(preprocess_image(batch, size=224), old, rtol=0, atol=0)


@pytest.mark.parametrize("size", [96, 128])
def test_preprocess_image_small_size(size):
    batch = np.random.default_rng(1).integers(0, 256, size=(3, size, size, 3), dtype=np.uint8)
    out = preprocess_image(batch, size=64)
    assert out.shape == (3, 3, 64, 64) and out.dtype == torch.float32
    flat = np.full((size, size, 3), 200, dtype=np.uint8)  # a flat colour stays flat when shrunk
    expected = (200 / 255 - torch.tensor(IMAGENET_MEAN)) / torch.tensor(IMAGENET_STD)
    torch.testing.assert_close(preprocess_image(flat, size=64)[0, :, 30, 30], expected)


# --- models ---------------------------------------------------------------------


def fake_batch(robot: str, b: int = 8, image: int | None = None, seed: int = 0) -> dict:
    """One robot's batch, shaped like WindowDataset(mode="pixels") batches."""
    g = torch.Generator().manual_seed(seed)
    image = image or IMAGE[robot]
    dims = 2 if robot == "pusht" else 4
    dim_mask = torch.zeros(b, 4, dtype=torch.bool)
    dim_mask[:, :dims] = True
    is_pad = torch.zeros(b, 16, dtype=torch.bool)
    is_pad[-1, 10:] = True
    actions = (torch.rand(b, 16, 4, generator=g) * 2 - 1) * dim_mask[:, None, :] * ~is_pad[:, :, None]
    state = torch.rand(b, 8, generator=g) * 2 - 1
    if robot == "pusht":
        state[:, 2:] = 0
    return {
        "pixels": torch.randint(0, 256, (b, 2, image, image, 3), generator=g, dtype=torch.uint8),
        "state": state,
        "embodiment_id": torch.full((b,), EMB[robot], dtype=torch.int64),
        "actions": actions,
        "action_dim_mask": dim_mask,
        "action_is_pad": is_pad,
    }


def _model(name: str, *overrides):
    from vla.models.baselines import build_baseline

    torch.manual_seed(0)
    return build_baseline(load_config([f"baseline={name}", "robot=pusht", "bc.pretrained=false", *overrides]))


@pytest.mark.parametrize("name", BASELINES)
@pytest.mark.parametrize("robot", ROBOTS)
def test_forward_shapes(name, robot):
    model = _model(name).eval()
    with torch.no_grad():
        out = model(fake_batch(robot, b=2))
    assert out.shape == (2, 16, 4) and out.dtype == torch.float32 and torch.isfinite(out).all()


def test_parameter_counts(capsys):
    counts = {name: count_params(_model(name)) for name in BASELINES}
    with capsys.disabled():
        print("\nTrainable parameters (baselines, full size):")
        for name, n in counts.items():
            print(f"  {name:9s} {n:>12,d}  ({n / 1e6:.2f}M)")
    # 2x3x64x64 + 8 -> 512 -> 512 -> 16x4
    assert counts["bc_mlp"] == (24584 * 512 + 512) + (512 * 512 + 512) + (512 * 64 + 64) == 12_883_008
    # ResNet-18 without its classifier (11,176,512) + 1032 -> 512 -> 512 -> 64
    assert counts["resnet18"] == 11_176_512 + (1032 * 512 + 512) + (512 * 512 + 512) + (512 * 64 + 64)


def test_conv_weights_get_weight_decay():
    model = _model("resnet18")
    cfg = load_config(["baseline=resnet18", "robot=pusht"])
    decay, no_decay = make_optimizer(model, cfg.optim).param_groups
    convs = [m.weight for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    decayed = {id(p) for p in decay["params"]}
    assert convs and all(id(w) in decayed for w in convs)
    assert decay["weight_decay"] == pytest.approx(0.01) and no_decay["weight_decay"] == 0.0


@pytest.mark.parametrize("name", BASELINES)
def test_each_baseline_overfits_one_batch(name):
    """Full width, 64x64 images, random init, one tiny fake FetchPush batch."""
    model = _model(name, "bc.image_size=64")
    cfg = load_config([f"baseline={name}", "robot=fetch", "optim.lr=1e-3", "optim.warmup_steps=10"])
    steps = 150  # enough: both reach ~1e-4 well before this (keeps the Mac suite fast)
    optimizer = make_optimizer(model, cfg.optim)
    scheduler = make_scheduler(optimizer, cfg.optim.warmup_steps, steps)
    batch = fake_batch("fetch", b=8, image=64)
    model.train()
    losses = []
    for _ in range(steps):
        loss = model.loss(batch)["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    print(f"{name}: loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < 0.02
    assert losses[-1] < 0.05 * losses[0]


def test_loss_is_that_robots_masked_l1():
    model = _model("bc_mlp")
    batch = fake_batch("pusht", b=4)
    out = model.loss(batch)
    pred = out["pred"]
    valid = batch["action_dim_mask"][:, None, :] & ~batch["action_is_pad"][:, :, None]
    expected = (pred - batch["actions"]).abs()[valid].mean()
    assert out["loss"].item() == pytest.approx(expected.item(), rel=1e-5)
    assert out["loss_pusht"].item() == pytest.approx(expected.item(), rel=1e-5)
    assert out["counts"][1].item() == 0  # no FetchPush samples


# --- data and evaluation see the same frames ------------------------------------------


@pytest.fixture
def cache_run(tmp_path, synthetic_data):
    run = tmp_path / "inputs" / "my-cache" / "cache_features-0000000-20261126-000000"
    build_cache(run, synthetic_data, fake_encode, fake_language)
    return run


@pytest.mark.parametrize("robot", ROBOTS)
def test_pixels_mode_gives_raw_frames(cache_run, robot):
    ds = WindowDataset(cache_run / robot, split="all", mode="pixels", require_training_images=False)
    images = np.load(cache_run / robot / "images.npy")
    for i in (0, 1, 7):  # frame 0 is the first step of an episode: frame t repeated
        s = ds[i]
        t = int(s["frame"])
        start = int(ds.episode_starts[int(s["episode"])])
        assert s["pixels"].dtype == torch.uint8 and s["pixels"].shape == (2, IMAGE[robot], IMAGE[robot], 3)
        np.testing.assert_array_equal(s["pixels"].numpy(), images[[max(t - 1, start), t]])
        assert "vision" not in s and "images" not in s


@pytest.mark.parametrize("robot", ROBOTS)
def test_eval_batch_equals_training_sample(cache_run, robot):
    from vla.evaluate import BaselineRolloutPolicy

    ds = WindowDataset(cache_run / robot, split="all", mode="pixels", require_training_images=False)
    images = np.load(cache_run / robot / "images.npy")
    raw_state = np.load(cache_run / robot / "lowdim.npz")["state"]
    policy = BaselineRolloutPolicy(_model("bc_mlp").eval(), Normalizer.load(cache_run / "stats.json"), "cpu")
    for i in (0, 1, 7):
        sample = ds[i]
        t = int(sample["frame"])
        start = int(ds.episode_starts[int(sample["episode"])])
        obs = {"image": images[t], "state": raw_state[t]}
        prev = obs if t == start else {"image": images[t - 1], "state": raw_state[t - 1]}
        policy.reset(robot, seed=0)
        batch = policy.make_batch(prev, obs, t - start)
        for key in ("pixels", "state", "embodiment_id"):
            torch.testing.assert_close(batch[key][0], sample[key], rtol=0, atol=0, msg=f"{key} differs")


def test_baseline_actions_are_unnormalized_and_clipped(cache_run):
    from vla.evaluate import BaselineRolloutPolicy

    policy = BaselineRolloutPolicy(_model("bc_mlp").eval(), Normalizer.load(cache_run / "stats.json"), "cpu")
    for robot, dims, (lo, hi) in (("pusht", 2, (0.0, 512.0)), ("fetch", 4, (-1.0, 1.0))):
        images = np.load(cache_run / robot / "images.npy")
        state = np.load(cache_run / robot / "lowdim.npz")["state"]
        policy.reset(robot, seed=0)
        obs = {"image": images[0], "state": state[0]}
        chunk = policy.act(obs, obs, 0)
        assert chunk.shape == (16, dims) and chunk.dtype == np.float32
        assert chunk.min() >= lo and chunk.max() <= hi


# --- training end to end (tiny, CPU) -------------------------------------------------------


@pytest.fixture
def local_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VLA_OUTPUTS_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("VLA_INPUTS_DIR", str(tmp_path / "inputs"))
    monkeypatch.setenv("WANDB_MODE", "disabled")


def _tiny(name, robot, *extra):
    return load_config([f"baseline={name}", f"robot={robot}", "size=tiny", "train.steps=6", "train.val_every=3",
                        "train.log_every=2", "eval.enabled=false", *extra])


@pytest.mark.parametrize("name, robot", [("bc_mlp", "pusht"), ("resnet18", "fetch")])
def test_tiny_baseline_training_run(cache_run, local_env, name, robot):
    from vla.evaluate import load_run

    run_dir = train(_tiny(name, robot), allow_dirty=True)
    assert run_dir.name.startswith(f"{name}-{robot}-s0-")
    for f in ("config.yaml", "git.txt", "metrics.json", "ckpt.pt", "stats.json"):
        assert (run_dir / f).is_file(), f
    m = json.loads((run_dir / "metrics.json").read_text())
    assert m["run_key"] == f"{name}-{robot}-tiny-s0-6steps"
    assert m["model"] == name and m["robot"] == robot and m["steps"] == 6
    assert [v["step"] for v in m["val_curve"]] == [3, 6]
    other = "fetch" if robot == "pusht" else "pusht"
    assert math.isfinite(m["val_loss"]) and m["val_loss"] == pytest.approx(m[f"val_loss_{robot}"])
    assert f"val_loss_{other}" not in m and f"train_loss_{other}" not in m
    assert set(m["train_windows"]) == {robot}

    model, cfg = load_run(run_dir, "cpu")
    assert type(model).__name__ == {"bc_mlp": "BCMLP", "resnet18": "ResNet18BC"}[name]
    assert cfg.baseline == name


def test_baseline_run_evaluates_its_robot_without_dinov2(cache_run, local_env, monkeypatch):
    import vla.evaluate

    def no_encoders(device):
        raise AssertionError("a baseline must not load DINOv2 or MiniLM")

    monkeypatch.setattr(vla.evaluate, "load_encoders", no_encoders)
    run_dir = train(_tiny("bc_mlp", "pusht", "eval.enabled=true", "eval.n_episodes=1", "eval.n_videos=1"),
                    allow_dirty=True)
    m = json.loads((run_dir / "metrics.json").read_text())
    assert m["eval"]["pusht"]["n"] == 1 and "fetch" not in m["eval"]
    assert "pusht_success" in m and "fetch_success" not in m
    assert m["mean_success"] == pytest.approx(m["pusht_success"])
    assert len(list((run_dir / "videos").glob("eval_pusht_*.mp4"))) == 1


# --- Kaggle jobs and the queue ---------------------------------------------------------------


def test_queue_builds_the_12_baseline_runs():
    kj = _load("kaggle_job")
    rq = _load("run_queue")
    assert {"smoke_baselines", "batch_baselines"} <= set(kj.JOBS)
    configs = kj.baseline_configs()
    assert configs == [[f"baseline={b}", f"robot={r}", f"seed={s}"]
                       for s in (0, 1, 2) for b in BASELINES for r in ROBOTS]
    jobs = rq.plan_jobs(configs, workers_per_run=kj.BATCH_WORKERS_PER_RUN)
    assert all(j["error"] is None for j in jobs), [j["error"] for j in jobs]
    keys = [j["key"] for j in jobs]
    assert len(set(keys)) == 12
    assert set(keys) == {f"{b}-{r}-full-s{s}-4000steps" for b in BASELINES for r in ROBOTS for s in (0, 1, 2)}
    for job in jobs:
        cfg = load_config(job["overrides"])
        assert cfg.data.num_workers == 2 and cfg.data.batch_size == 128 and cfg.data.mode == "pixels"
    print("\n".join(keys))


# --- FetchPush: starts that are already solved -------------------------------------------------


def test_fetch_start_check_on_two_seeds():
    fsc = _load("fetch_start_check")
    a = fsc.start_distances([100_000, 100_001])
    b = fsc.start_distances([100_000, 100_001])
    assert [r["seed"] for r in a] == [100_000, 100_001]
    for r, r2 in zip(a, b):
        assert math.isfinite(r["distance"]) and r["distance"] >= 0
        assert r["distance"] == r2["distance"]  # same seed, same start
        assert r["threshold"] == pytest.approx(0.05)
        assert r["solved_at_start"] == (r["distance"] < r["threshold"])
