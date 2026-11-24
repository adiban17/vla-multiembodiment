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
5. **Stop after each phase and report**: show the check output, list the files changed, and say
   what the user must run on Kaggle. Do not start the next phase.

## Layout
```
configs/                 YAML configs (base, cond/, adapt/, baseline/)
src/vla/envs/            PushT and FetchPush wrappers
src/vla/data/            loaders, scripted expert, dataset, normalization, feature cache
src/vla/models/          encoders (+ preprocess_image), trunk, conditioning, policy, baselines
src/vla/utils/           runtime.py (paths, device), tracking.py (git, run info), config, seed
src/vla/train.py, evaluate.py
scripts/kaggle_job.py    registry of named jobs run on Kaggle (and locally on CPU)
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
