# vla-multiembodiment

A learning project: one vision-language-action (VLA) policy that drives two simulated robots,
PushT (2-D actions) and FetchPush (4-D actions), trained by behavior cloning on expert demos.
Frozen DINOv2-small (vision) and MiniLM (language) feed a ~15M-parameter transformer that outputs
16-step action chunks. The research questions compare three ways to tell the model which robot
it drives (learned token, FiLM, separate heads), and full fine-tuning against LoRA for the
vision encoder, with BC-MLP and ResNet-18 baselines. The full design is in `docs/spec.md`;
read the relevant section before starting a phase.

The user is new to Claude Code, Python packaging and robotics: explain what you do in simple words.

## Constraints (these override docs/spec.md)
- **Mac = CPU only.** The laptop is an M1 Mac with no NVIDIA GPU. On the Mac, run only small
  CPU tests. **Never use the `"mps"` device**; use `vla.utils.runtime.get_device()`, which
  returns `"cuda"` or `"cpu"`.
- **All GPU work runs on Kaggle** (free T4 x2): data collection, training and evaluation.
  Kaggle clones this repo from a public GitHub repo. Colab is only a backup.
  The spec's "rent a GPU machine" setup does not apply.
- **The Kaggle notebook has exactly two cells** (see `notebooks/kaggle_cells.md`):
  1. Clone the repo to `/tmp/repo`, `pip install -r requirements-kaggle.txt`, load
     `WANDB_API_KEY` from Kaggle Secrets.
  2. `python scripts/kaggle_job.py --job <name>`.
  All logic lives in the repo. A new piece of work is a new job in `scripts/kaggle_job.py`, never
  new notebook code. This replaces the spec's `scripts/sweep.sh`.
- **Paths on Kaggle:** write outputs to `/kaggle/working/outputs/`. Read earlier outputs (attached
  as datasets) from `/kaggle/input/`. Use `vla.utils.runtime.outputs_dir()` and `inputs_dir()`.
  On the Mac, outputs go to `./outputs/` (gitignored).
- **Frozen runs train 4,000 steps with 300 warmup steps** (spec: 20k and 500). Reason: Kaggle job
  `smoke_train` measured 2.0 steps/s on a T4 at batch 256, so 20k steps would take about 3 hours.
  The budget is 45 minutes per run including evaluation; 4,000 steps is about 33 minutes of training.
  Validation stays every 1,000 steps. The defaults live in `configs/base.yaml`.
- **Secrets never go in git.** On the Mac, use the gitignored `.env` (template: `.env.example`).
  On Kaggle, use Kaggle Secrets.

## Rules
1. **Pin dependency versions** with `==`. `requirements.txt` is for the Mac (CPU).
   `requirements-kaggle.txt` must NOT list torch, torchvision, transformers or numpy, because
   Kaggle already has them. Add a package in the phase that first needs it.
2. **Every run saves its resolved config and git commit** into its run folder
   (`vla.utils.tracking.save_run_info`) and into W&B (project `vla-multiembodiment`).
3. **One `preprocess_image()`** (in `src/vla/models/encoders.py`) is shared by feature caching,
   training and evaluation. Never write a second copy of image preprocessing.
4. **Write each phase's checks before its code**, as tests in `tests/` or scripts, using the gate
   from the roadmap in `docs/spec.md`. Then implement until they pass.
5. **Training images come from the evaluation renderer only.** FetchPush training images must be
   rendered with EGL on Kaggle; PushT training images come from the LeRobot dataset. Mac-rendered
   images are debug-only (save them under `runs/debug/`). Every dataset's `meta.json` records
   `image_source` and `images_for_training`; call `vla.data.shards.assert_training_images_ok(meta)`
   before training on a dataset.
6. **lerobot lives in its own env.** lerobot 0.7.0 needs Python 3.12 and pins its own torch/numpy, so
   it is only in `requirements-lerobot.txt`, installed by `vla.data.lerobot_env` into a separate uv
   env. Never add it to `requirements.txt` or `requirements-kaggle.txt`.
7. **Stop after each phase and report**: show the check output, list the files changed, and say
   what the user must run on Kaggle. Do not start the next phase.

## Layout
```
configs/                 YAML configs (base, cond/, adapt/, baseline/)
src/vla/envs/            PushT and FetchPush wrappers
src/vla/data/            shards.py (.npz format), fetch_expert, collect_fetch, pusht_loader (lerobot env),
                         lerobot_env, pusht_compare, normalize, instructions, dataset, cache_features
src/vla/models/          encoders (+ preprocess_image), trunk, conditioning, policy, baselines
src/vla/utils/           runtime.py (paths, device), tracking.py (git, run info), config, seed
src/vla/train.py         trains, then evaluates the final model (cfg.eval)
src/vla/evaluate.py      rollout harness: expert/random/VLA policies, Wilson intervals, videos
scripts/kaggle_job.py    registry of named jobs run on Kaggle (and locally on CPU)
scripts/run_queue.py     runs configs two at a time (one per GPU), skips runs finished in W&B
tests/                   checks, written before each phase's code
notebooks/kaggle_cells.md   the two Kaggle cells
```

## Commands (Mac)
- Tests: `.venv/bin/python -m pytest -q`
- List jobs: `.venv/bin/python scripts/kaggle_job.py --list`
- Run a job on CPU without W&B: `WANDB_MODE=disabled .venv/bin/python scripts/kaggle_job.py --job hello`
- Always call `.venv/bin/python` directly (a conda base env is also on PATH).
- Add or upgrade a package: `uv pip install --python .venv/bin/python <pkg>==<version>`, then pin it
  in the requirements file(s).

## Progress
- Phase 0 (setup): done. The Kaggle `hello` job is the GPU gate.
- Phase 1 (envs + data): built and tested on the Mac. Kaggle jobs `data_pusht`, `data_fetch_smoke`,
  `data_fetch` not run yet. Pins: mujoco 3.9.0 (gymnasium-robotics 1.4.2 breaks on mujoco 4.x).
  FetchPush camera: `vla.envs.fetch.CAMERA` (45° front view, 128x128); the arm sometimes hides the cube.
- Phase 2 (dataset + cache): built and tested on the Mac. Kaggle jobs `cache_features_smoke`, `cache_features`
  not run yet. A cache run folder is self-contained (stats.json, language.npy, `<robot>/` with lowdim.npz,
  vision.npy fp16, images.npy uint8); `WindowDataset(run/<robot>, mode="cached"|"raw")` reads it.
  Normalization: dims with train range < 0.01 are constant (-> 0); FetchPush actions are identity.
  Mac debug data lives in `runs/debug/pusht` and `runs/debug/fetch_mac_render`
  (`VLA_INPUTS_DIR=runs/debug` runs the cache jobs locally).
- Phase 3 (model + training loop): built and tested on the Mac. Kaggle job `smoke_train` not run yet.
  Trainable params: none 14.68M, token 14.68M, film 15.48M, sep_heads 14.83M (`scripts/count_params.py`).
  Config groups: `cond=`, `adapt=`, `size=full|tiny` (`vla.utils.config.load_config`). Loss = average of the
  PushT and FetchPush masked-L1 means (equal weight per robot; same rule for val loss).
  `python -m vla.train size=tiny --allow-dirty` trains locally (Mac debug caches allowed only with size=tiny).
- Phase 4 (evaluation + first batch): built and tested on the Mac. Kaggle jobs `eval_sanity`, `batch_conditioning`
  not run yet. Rollouts use env seeds 100000-100049 and execute 8 of 16 actions per call; eval features are rounded
  to fp16 like the cache. Every run folder holds copies of stats.json and language.npy. `run_key` =
  `{cond}-{adapt}-{size}-s{seed}-{steps}steps` (no commit) is what the queue uses to skip finished W&B runs.
  The queue gives each run 2 data-loader workers when 2 run at once (`data.num_workers` affects which instruction
  each sample gets, so it is part of the config). `python -m vla.evaluate --policy expert|random|<run_dir>`.
