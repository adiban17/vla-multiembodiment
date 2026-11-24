# vla-multiembodiment

One vision-language-action policy for two simulated robots, PushT and FetchPush, with ablations on
how the model is told which robot it controls (learned token, FiLM, separate heads) and on how the
vision encoder is adapted (frozen, full fine-tuning, LoRA). The design is in [docs/spec.md](docs/spec.md).

Results: none yet. Every number added here will trace back to a run ID.

## Setup on a Mac (CPU-only tests)
```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env        # then paste your W&B key into .env
.venv/bin/python -m pytest -q
```

## Running on Kaggle (GPU)
All data collection, training and evaluation run in a two-cell Kaggle notebook. See
[notebooks/kaggle_cells.md](notebooks/kaggle_cells.md). Available jobs:
`python scripts/kaggle_job.py --list`.
