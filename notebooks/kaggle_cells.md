# Kaggle notebook: the two cells

All logic lives in this repo. The notebook only fetches the latest code (cell 1) and runs one
named job (cell 2). Replace `YOUR-USERNAME` with your GitHub username.

## One-time setup in Kaggle
1. **Verify your phone number** (Kaggle → Settings → Phone verification). Without it, notebooks
   have no Internet and no GPU.
2. Create a notebook: kaggle.com → **Create → New Notebook**.
3. In the right-hand panel, open **Session options**:
   - **Accelerator:** `GPU T4 x2`
   - **Internet:** on
4. Add your W&B key as a secret: menu **Add-ons → Secrets → Add Secret**.
   Label: `WANDB_API_KEY`, value: your key from https://wandb.ai/authorize.
   Make sure the checkbox next to it is ticked, which attaches it to this notebook.

## Cell 1: get the code, install, load the secret
Run this again after every `git push`, so the notebook uses your newest code.

```python
import os
from kaggle_secrets import UserSecretsClient

REPO_URL = "https://github.com/YOUR-USERNAME/vla-multiembodiment.git"

!rm -rf /tmp/repo && git clone --depth 1 {REPO_URL} /tmp/repo
%cd /tmp/repo
!git log -1 --oneline
!pip install -q -r requirements-kaggle.txt

os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
print("W&B key loaded:", bool(os.environ["WANDB_API_KEY"]))
```

## Cell 2: run one job
Change `hello` to the job you want. `python scripts/kaggle_job.py --list` shows them all.

```python
%cd /tmp/repo
!python scripts/kaggle_job.py --job hello
```

`batch_adaptation` also needs a step count, taken from the `smoke_adapt` summary, for example
`!python scripts/kaggle_job.py --job batch_adaptation --steps 3000`. It is a command-line argument,
so changing it never needs a new push.

## Jobs
| Job | Accelerator | What it does | Time (estimate) |
| --- | --- | --- | --- |
| `hello` | GPU T4 x2 | prints GPUs + commit, tiny W&B run | 1 min |
| `data_pusht` | None (CPU) is enough | builds a separate Python 3.12 lerobot env, exports lerobot/pusht (206 episodes) to .npz, 3 videos, `pusht_compare.png` | 15-40 min |
| `data_fetch_smoke` | GPU T4 x2 (EGL rendering) | 5 FetchPush demos with images, 3 videos | 1 min |
| `data_fetch` | GPU T4 x2 (EGL rendering) | 500 FetchPush demos with images, 3 videos | 15-30 min |
| `cache_features_smoke` | GPU T4 x2 | feature cache of the first 3 episodes per robot; prints the datasets found, stats, sizes and the cached-vs-live check | 2-3 min |
| `cache_features` | GPU T4 x2 | DINOv2 + MiniLM features (fp16), raw images (uint8), `stats.json`, `size_report.json` for all episodes | 10-30 min |
| `smoke_train` | GPU T4 x2 (uses one) | FiLM policy, 1,000 training steps; prints steps/sec, peak GPU memory, data-wait vs compute time per step, train and val loss | 5-15 min |
| `eval_sanity` | GPU T4 x2 (uses one) | scripted expert and random policy through the evaluation harness, then one full FiLM seed-0 run (4,000 steps + 50 rollouts per robot); prints the time per run | 45-60 min |
| `batch_conditioning` | GPU T4 x2 (both) | token, film, sep_heads x seeds 0, 1, 2, two runs at a time (2 data-loader workers each); skips runs already finished in W&B; prints a results table | about 4 h |
| `smoke_baselines` | GPU T4 x2 (uses one) | 200 steps of BC-MLP and ResNet-18 on each robot + 2 rollouts; prints steps/s, peak memory, time per full run | 10-20 min |
| `batch_baselines` | GPU T4 x2 (both) | bc_mlp, resnet18 x pusht, fetch x seeds 0, 1, 2, two at a time; skips finished runs | about 4-6 h |
| `smoke_adapt` | GPU T4 x2 (uses one) | sep_heads with full_ft and with lora16, 200 steps each (batch 64, 2 workers) + 2 rollouts per robot; prints trainable counts, s/step, data wait, peak GPU memory after 200 steps, and the largest `--steps` that keeps train + eval under 40 min | 15-25 min |
| `batch_adaptation --steps N` | GPU T4 x2 (both) | full_ft, lora16, frozen_matched x seeds 0, 1, 2 on sep_heads, N steps each, two at a time; skips runs finished in W&B (run key includes the mode and N) | about 3-3.5 h |

Run `data_fetch_smoke` before `data_fetch`, and watch its videos in W&B first.

**Before the cache jobs:** in the notebook's right panel, **Add Input** → the saved outputs of your
`data_pusht` and `data_fetch` notebooks. The job searches `/kaggle/input/` for folders with
`meta.json` and prints what it found, so the folder names do not matter. If both `data_fetch_smoke`
and `data_fetch` are attached, it uses the one with more episodes. Run `cache_features_smoke`
interactively first, then `cache_features` as **Save & Run All (Commit)** (about 4.5 GB of output).
For long jobs use **Save Version → Save & Run All (Commit)** so the outputs are kept.

**Before `smoke_train`:** **Add Input** → the saved output of your `cache_features` notebook, and
nothing else is needed. The job searches `/kaggle/input/` for a folder with `stats.json` and `pusht/`,
`fetch/` feature subfolders, prints what it found, and uses the one with the most frames. It refuses
caches with debug-only (Mac-rendered) images, and refuses to start if the cloned code has uncommitted
changes. Send back the `=== smoke_train summary ===` block: it fills in the compute table.

**Before `eval_sanity` and `batch_conditioning`:** **Add Input** → the saved output of your
`cache_features` notebook (as for `smoke_train`). Run both as **Save Version → Save & Run All (Commit)**.
Run `eval_sanity` first and send back its `=== eval_sanity summary ===` block. Its FiLM run is exactly
`film-s0` of the batch (same config and run key), so `batch_conditioning` skips it and trains the
other 8. If a batch version stops early or a run fails, just run a new version: runs already finished
in W&B are skipped. Each run writes `metrics.json` with success rates, 95% Wilson intervals and the
evaluation time, and logs 3 rollout videos per robot to W&B.

**Before `smoke_adapt` and `batch_adaptation`:** **Add Input** → the saved output of your `cache_features`
notebook (the adaptation runs read its raw `images.npy` frames and `language.npy`). Run `smoke_adapt` first
and send back its `=== smoke_adapt summary ===` block. Then run `batch_adaptation --steps N` (N from that
summary) as **Save Version → Save & Run All (Commit)**. `frozen_matched` is the fair frozen control: frozen
DINOv2 and cached features, but the same batch (64) and steps (N) as the adapted runs.

## Where results go
- Each run writes a folder to `/kaggle/working/outputs/<run_id>/` with `config.yaml`, `git.txt`
  and `metrics.json`, and logs to the W&B project `vla-multiembodiment`.
- `/kaggle/working` is kept only when you **Save Version → Save & Run All (Commit)**. An interactive
  session loses it when it stops. Long jobs should be run as a saved version.
- To use one run's outputs in a later notebook: **Add Input** → pick that notebook's output. It
  appears under `/kaggle/input/`.

## Expected output of `hello`
Two lines like `GPUs: [0] Tesla T4, [1] Tesla T4` and `Git commit: <40 characters>`, a tiny run whose
loss falls close to 0, and a new run in W&B under project `vla-multiembodiment`.
