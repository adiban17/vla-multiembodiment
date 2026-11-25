"""Phase 6 checks: the adaptation regimes (frozen vs full fine-tuning vs LoRA). Written before the code.

Gate (docs/spec.md, Phase 6, plus the user's Phase 6 request):
- trainable counts printed: about 0.44M LoRA parameters in DINOv2, about 22M for full fine-tuning;
- a forward pass on 2 images on CPU for both modes;
- LoRA at step 0 equals the frozen model (its B matrices start at zero);
- evaluating an adapted run uses that run's own adapted DINOv2 (not the frozen one from the cache);
- frozen_matched: a frozen control with the adaptation runs' batch size and step count;
- the queue builds the 9 adaptation configs, with --steps N in every run key.

Everything runs on the Mac CPU with the real DINOv2-small (it is in the local Hugging Face cache).
"""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import fake_encode, fake_language
from vla.data.cache_features import build_cache
from vla.data.dataset import WindowDataset, collate_mixed_pixels
from vla.data.instructions import all_instructions, paraphrases
from vla.data.normalize import Normalizer
from vla.models.encoders import DinoV2Encoder, preprocess_image
from vla.models.policy import AdaptedVLAPolicy, VLAPolicy, build_policy, count_params, encoder_mode, encoder_params
from vla.train import build_datasets, make_optimizer, make_train_loader, run_key, train
from vla.utils.config import load_config

REPO = Path(__file__).resolve().parents[1]
ROBOTS = ("pusht", "fetch")
IMAGE = {"pusht": 96, "fetch": 128}
LORA_PARAMS = 12 * 3 * (16 * 384 + 384 * 16)  # 12 blocks x (query, key, value) x (A + B) = 442,368


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model(adapt, *extra, size="full"):
    torch.manual_seed(0)
    return build_policy(load_config(["cond=sep_heads", f"adapt={adapt}", f"size={size}", *extra]))


def _pixels(b=1, robot="pusht", seed=0):
    """(b, 2, H, W, 3) random uint8 frames t-1 and t."""
    g = np.random.default_rng(seed)
    return torch.from_numpy(g.integers(0, 256, size=(b, 2, IMAGE[robot], IMAGE[robot], 3), dtype=np.uint8))


def _batch(pixels, robot="pusht"):
    b = len(pixels) if isinstance(pixels, torch.Tensor) else sum(len(p) for p in pixels)
    g = torch.Generator().manual_seed(1)
    return {"pixels": pixels, "lang": torch.randn(b, 384, generator=g), "state": torch.randn(b, 8, generator=g),
            "embodiment_id": torch.full((b,), 0 if robot == "pusht" else 1, dtype=torch.int64)}


# --- 1. configs ---------------------------------------------------------------------------


def test_adaptation_configs():
    full = load_config(["cond=sep_heads", "adapt=full_ft"])
    lora = load_config(["cond=sep_heads", "adapt=lora16"])
    matched = load_config(["cond=sep_heads", "adapt=frozen_matched"])
    for cfg in (full, lora):
        assert cfg.data.mode == "pixels" and cfg.data.batch_size == 64
        assert cfg.optim.lr == pytest.approx(3e-4)  # the head keeps the frozen runs' learning rate
    assert (encoder_mode(full), full.encoder.lr) == ("full", pytest.approx(1e-5))
    assert (encoder_mode(lora), lora.encoder.lr) == ("lora", pytest.approx(1e-4))
    assert (lora.encoder.lora_r, lora.encoder.lora_alpha, lora.encoder.lora_dropout) == (16, 32, pytest.approx(0.05))
    assert list(lora.encoder.lora_targets) == ["q_proj", "k_proj", "v_proj"]  # query, key, value in transformers 5
    assert encoder_mode(matched) == "frozen" and matched.data.mode == "cached" and matched.data.batch_size == 64
    assert matched.optim.lr == pytest.approx(3e-4)


def test_run_keys_include_mode_and_steps():
    for adapt in ("full_ft", "lora16", "frozen_matched"):
        cfg = load_config(["cond=sep_heads", f"adapt={adapt}", "seed=2", "train.steps=2500"])
        assert run_key(cfg) == f"sep_heads-{adapt}-full-s2-2500steps"
    # frozen_matched never collides with batch_conditioning's frozen sep_heads run
    assert run_key(load_config(["cond=sep_heads", "adapt=frozen_matched"])) != \
        run_key(load_config(["cond=sep_heads"])) == "sep_heads-frozen-full-s0-4000steps"


def test_frozen_defaults_unchanged():
    cfg = load_config()
    assert (cfg.adapt, encoder_mode(cfg), cfg.data.mode, cfg.data.batch_size) == ("frozen", "frozen", "cached", 256)
    assert (cfg.train.steps, cfg.optim.warmup_steps, cfg.optim.lr) == (4000, 300, pytest.approx(3e-4))
    assert type(build_policy(load_config(["size=tiny"]))) is VLAPolicy
    assert encoder_mode({"cond": "film"}) == "frozen"  # checkpoints saved before Phase 6 have no encoder block


# --- 2. trainable counts ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def lora_model():
    return _model("lora16").eval()


@pytest.fixture(scope="module")
def full_model():
    return _model("full_ft").eval()


def test_trainable_counts(lora_model, full_model, capsys):
    head = count_params(_model("frozen_matched"))
    lora_enc, full_enc = encoder_params(lora_model), encoder_params(full_model)
    with capsys.disabled():
        print(f"\nTrainable parameters (sep_heads, full size):")
        print(f"  frozen_matched  encoder {0:>12,d}  head {head:>12,d}  total {head:>12,d}")
        print(f"  lora16          encoder {lora_enc:>12,d}  head {count_params(lora_model) - lora_enc:>12,d}  "
              f"total {count_params(lora_model):>12,d}")
        print(f"  full_ft         encoder {full_enc:>12,d}  head {count_params(full_model) - full_enc:>12,d}  "
              f"total {count_params(full_model):>12,d}")
    assert lora_enc == LORA_PARAMS == 442_368
    assert 21e6 < full_enc < 23e6
    assert count_params(lora_model) - lora_enc == count_params(full_model) - full_enc == head
    trainable = [n for n, p in lora_model.encoder.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
    lora_b = [p for n, p in lora_model.encoder.named_parameters() if "lora_B" in n]
    assert len(lora_b) == 36 and all(torch.count_nonzero(p) == 0 for p in lora_b)
    assert all(p.requires_grad for p in full_model.encoder.parameters())


# --- 3. forward on 2 images, gradients ----------------------------------------------------------


@pytest.mark.parametrize("adapt", ["full_ft", "lora16"])
def test_forward_two_images_on_cpu_and_gradients(adapt):
    model = _model(adapt).train()
    out = model(_batch(_pixels(1)))  # one sample = frames t-1 and t = 2 images
    assert out.shape == (1, 16, 4) and torch.isfinite(out).all()
    out.abs().mean().backward()
    named = dict(model.encoder.named_parameters())
    if adapt == "full_ft":
        patch = next(p for n, p in named.items() if "patch_embeddings" in n and n.endswith("weight"))
        assert patch.grad is not None and patch.grad.abs().sum() > 0
    else:
        b_grads = [p.grad for n, p in named.items() if "lora_B" in n]
        assert all(g is not None for g in b_grads) and sum(g.abs().sum() for g in b_grads) > 0
        assert all(p.grad is None for n, p in named.items() if "lora_" not in n)


# --- 4. LoRA at init equals the frozen model ----------------------------------------------------


def test_lora_at_init_equals_frozen_model(lora_model):
    frozen = build_policy(load_config(["cond=sep_heads"])).eval()
    head_state = {k: v for k, v in lora_model.state_dict().items() if not k.startswith("encoder.")}
    frozen.load_state_dict(head_state)
    dino = DinoV2Encoder().eval()
    for robot in ROBOTS:
        pixels = _pixels(2, robot, seed=3)
        batch = _batch(pixels, robot)
        with torch.no_grad():
            vision = dino(preprocess_image(pixels.reshape(4, *pixels.shape[2:]))).reshape(2, 2, 65, 384)
            expected = frozen({**batch, "vision": vision})
            got = lora_model(batch)
        torch.testing.assert_close(got, expected, rtol=0, atol=1e-6)


# --- data: mixed image sizes in one batch --------------------------------------------------------


@pytest.fixture
def cache_run(tmp_path, synthetic_data):
    run = tmp_path / "inputs" / "my-cache" / "cache_features-0000000-20261126-000000"
    build_cache(run, synthetic_data, fake_encode, fake_language)
    return run


def test_collate_keeps_order_and_groups_by_size():
    samples = [{"pixels": _pixels(1, r)[0], "x": torch.tensor(i)} for i, r in enumerate(["pusht", "pusht", "fetch"])]
    batch = collate_mixed_pixels(samples)
    assert batch["x"].tolist() == [0, 1, 2]
    assert [tuple(p.shape) for p in batch["pixels"]] == [(2, 2, 96, 96, 3), (1, 2, 128, 128, 3)]
    one_size = collate_mixed_pixels(samples[:2])
    assert isinstance(one_size["pixels"], list) and len(one_size["pixels"]) == 1


def test_mixed_train_batch_forward_equals_per_sample(cache_run):
    cfg = load_config(["cond=sep_heads", "adapt=lora16", "size=tiny"])
    train_sets, _, _ = build_datasets(cache_run, cfg, require_training_images=False)
    batch = next(iter(make_train_loader(train_sets, cfg, "cpu")))
    assert [tuple(p.shape) for p in batch["pixels"]] == [(4, 2, 96, 96, 3), (4, 2, 128, 128, 3)]
    model = _model("lora16", size="tiny").eval()
    with torch.no_grad():
        together = model(batch)
        rows = []
        for group_start, group in zip((0, 4), batch["pixels"]):
            for j in range(len(group)):
                i = group_start + j
                single = {k: v[i:i + 1] for k, v in batch.items() if isinstance(v, torch.Tensor)}
                rows.append(model({**single, "pixels": group[j:j + 1]}))
    torch.testing.assert_close(together, torch.cat(rows), rtol=0, atol=1e-5)


# --- 6. optimizer groups ---------------------------------------------------------------------------


def test_optimizer_learning_rates():
    for adapt, enc_lr in (("full_ft", 1e-5), ("lora16", 1e-4)):
        cfg = load_config(["cond=sep_heads", f"adapt={adapt}", "size=tiny"])
        model = _model(adapt, size="tiny")
        opt = make_optimizer(model, cfg.optim, encoder_lr=cfg.encoder.lr)
        enc_ids = {id(p) for p in model.encoder.parameters() if p.requires_grad}
        seen = set()
        for g in opt.param_groups:
            ids = {id(p) for p in g["params"]}
            seen |= ids
            if ids & enc_ids:
                assert ids <= enc_ids and g["lr"] == pytest.approx(enc_lr)
            else:
                assert g["lr"] == pytest.approx(3e-3)  # size=tiny's head lr
        assert seen == {id(p) for p in model.parameters() if p.requires_grad}
    frozen = build_policy(load_config(["size=tiny"]))
    groups = make_optimizer(frozen, load_config(["size=tiny"]).optim).param_groups
    assert len(groups) == 2 and [g["weight_decay"] for g in groups] == [0.01, 0.0]


# --- 7, 8. evaluation uses the run's own adapted encoder -----------------------------------------


def _lang_vectors(cache_run):
    language = np.load(cache_run / "language.npy")
    order = all_instructions()
    return {r: language[order.index(paraphrases(r)[0])] for r in ROBOTS}


@pytest.mark.parametrize("robot", ROBOTS)
def test_adapted_eval_batch_equals_training_sample(cache_run, robot):
    from vla.evaluate import AdaptedVLARolloutPolicy

    ds = WindowDataset(cache_run / robot, split="all", mode="pixels", instructions=0, require_training_images=False)
    images = np.load(cache_run / robot / "images.npy")
    raw_state = np.load(cache_run / robot / "lowdim.npz")["state"]
    policy = AdaptedVLARolloutPolicy(_model("lora16", size="tiny").eval(), Normalizer.load(cache_run / "stats.json"),
                                     _lang_vectors(cache_run), "cpu")
    assert policy.needs_images
    for i in (0, 1, 7):
        sample = ds[i]
        t = int(sample["frame"])
        start = int(ds.episode_starts[int(sample["episode"])])
        obs = {"image": images[t], "state": raw_state[t]}
        prev = obs if t == start else {"image": images[t - 1], "state": raw_state[t - 1]}
        policy.reset(robot, seed=0)
        batch = policy.make_batch(prev, obs, t - start)
        assert "vision" not in batch
        for key in ("pixels", "lang", "state", "embodiment_id"):
            torch.testing.assert_close(batch[key][0], sample[key], rtol=0, atol=0, msg=f"{key} differs")


@pytest.mark.parametrize("adapt", ["lora16", "full_ft"])
def test_changing_encoder_weights_changes_rollout_actions(cache_run, adapt):
    from vla.evaluate import AdaptedVLARolloutPolicy

    model = _model(adapt, size="tiny").eval()
    policy = AdaptedVLARolloutPolicy(model, Normalizer.load(cache_run / "stats.json"), _lang_vectors(cache_run), "cpu")
    images = np.load(cache_run / "fetch" / "images.npy")
    state = np.load(cache_run / "fetch" / "lowdim.npz")["state"]
    obs, prev = {"image": images[5], "state": state[5]}, {"image": images[4], "state": state[4]}
    policy.reset("fetch", seed=0)
    before = policy.act(prev, obs, 5)
    with torch.no_grad():
        g = torch.Generator().manual_seed(0)
        for name, p in model.encoder.named_parameters():
            if (adapt == "lora16" and "lora_B" in name) or (adapt == "full_ft" and "patch_embeddings" in name):
                p.add_(0.5 * torch.randn(p.shape, generator=g))
    after = policy.act(prev, obs, 5)
    assert before.shape == after.shape == (16, 4)
    assert np.abs(before - after).max() > 1e-4, "the rollout policy ignored the adapted encoder"


# --- 9. tiny end-to-end runs -------------------------------------------------------------------------


@pytest.fixture
def local_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VLA_OUTPUTS_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("VLA_INPUTS_DIR", str(tmp_path / "inputs"))
    monkeypatch.setenv("WANDB_MODE", "disabled")


def test_tiny_lora_run_end_to_end_uses_its_own_encoder(cache_run, local_env, monkeypatch):
    import vla.evaluate
    from vla.evaluate import load_run

    def no_frozen_encoders(device):
        raise AssertionError("an adapted run must be evaluated with its own DINOv2, not the frozen one")

    monkeypatch.setattr(vla.evaluate, "load_encoders", no_frozen_encoders)
    cfg = load_config(["cond=sep_heads", "adapt=lora16", "size=tiny", "train.steps=3", "train.val_every=3",
                       "train.log_every=1", "eval.n_episodes=1", "eval.n_videos=0"])
    run_dir = train(cfg, allow_dirty=True)
    assert run_dir.name.startswith("sep_heads-lora16-s0-")
    m = json.loads((run_dir / "metrics.json").read_text())
    assert m["run_key"] == "sep_heads-lora16-tiny-s0-3steps" and m["adapt"] == "lora16"
    assert m["encoder_trainable_params"] == LORA_PARAMS
    assert m["head_trainable_params"] == m["trainable_params"] - LORA_PARAMS
    assert "peak_gpu_mem_gb_at_200" in m and m["peak_gpu_mem_gb_at_200"] is None  # CPU
    assert m["val_seconds"] > 0 and math.isfinite(m["val_loss"])
    assert m["eval"]["pusht"]["n"] == 1 and m["eval"]["fetch"]["n"] == 1 and "mean_success" in m

    model, loaded_cfg = load_run(run_dir, "cpu")
    assert type(model) is AdaptedVLAPolicy and encoder_mode(loaded_cfg) == "lora"
    ckpt = torch.load(run_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    assert any("lora_B" in k for k in ckpt["model"])
    trained_b = [v for k, v in ckpt["model"].items() if "lora_B" in k]
    assert any(torch.count_nonzero(v) > 0 for v in trained_b)  # LoRA really trained (B left zero)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, ckpt["model"][k], rtol=0, atol=0)


def test_tiny_frozen_matched_run(cache_run, local_env):
    cfg = load_config(["cond=sep_heads", "adapt=frozen_matched", "size=tiny", "train.steps=3",
                       "train.val_every=3", "eval.enabled=false"])
    run_dir = train(cfg, allow_dirty=True)
    m = json.loads((run_dir / "metrics.json").read_text())
    assert m["run_key"] == "sep_heads-frozen_matched-tiny-s0-3steps"
    assert m["encoder_trainable_params"] == 0 and m["head_trainable_params"] == m["trainable_params"]


# --- 10. Kaggle jobs and the queue -------------------------------------------------------------------


def test_queue_builds_the_9_adaptation_runs():
    kj = _load("kaggle_job")
    rq = _load("run_queue")
    assert {"smoke_adapt", "batch_adaptation"} <= set(kj.JOBS)
    configs = kj.adaptation_configs(2500)
    modes = ("full_ft", "lora16", "frozen_matched")
    assert configs == [["cond=sep_heads", f"adapt={a}", f"seed={s}", "train.steps=2500"]
                       for s in (0, 1, 2) for a in modes]
    jobs = rq.plan_jobs(configs, workers_per_run=kj.BATCH_WORKERS_PER_RUN)
    assert all(j["error"] is None for j in jobs), [j["error"] for j in jobs]
    keys = [j["key"] for j in jobs]
    assert len(set(keys)) == 9
    assert set(keys) == {f"sep_heads-{a}-full-s{s}-2500steps" for a in modes for s in (0, 1, 2)}
    for job in jobs:
        cfg = load_config(job["overrides"])
        assert cfg.data.num_workers == 2 and cfg.data.batch_size == 64 and cfg.train.steps == 2500
        assert cfg.data.mode == ("cached" if cfg.adapt == "frozen_matched" else "pixels")
    print("\n" + "\n".join(keys))


def test_steps_argument():
    kj = _load("kaggle_job")
    with pytest.raises(ValueError, match="--steps"):
        kj.run_job("batch_adaptation", steps=None)  # must be given explicitly
    with pytest.raises(ValueError, match="--steps"):
        kj.run_job("hello", steps=100)  # a job that does not take it
    with pytest.raises(ValueError, match="warmup"):
        kj.adaptation_configs(200)  # shorter than the 300 warmup steps


def test_max_steps_within():
    kj = _load("kaggle_job")
    # 0.5 s/step, 30 s per validation (every 1,000 steps), 600 s of evaluation and start-up, 40 min budget:
    # 3,300 steps = 1650 + 4*30 + 600 = 2370 s fits; 3,400 steps = 1700 + 120 + 600 = 2420 s does not.
    assert kj.max_steps_within(2400, 0.5, 30, 1000, 600) == 3300
    assert kj.max_steps_within(2400, 0.5, 30, 1000, 2500) == 0  # evaluation alone is over budget
    assert kj.max_steps_within(2400, 0.1, 0, 1000, 0) % 100 == 0


# --- 11. peft pins ---------------------------------------------------------------------------------------


def test_peft_pinned_and_installed():
    import peft

    for name in ("requirements.txt", "requirements-kaggle.txt"):
        lines = (REPO / name).read_text().splitlines()
        assert f"peft=={peft.__version__}" in lines, f"{name} must pin the installed peft {peft.__version__}"
