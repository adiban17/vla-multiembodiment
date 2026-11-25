"""Experiment tracking: record exactly which code and config produced each run.

Rule: every run saves its resolved config and git commit into its run folder.
"""

import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

from vla.utils.runtime import REPO_ROOT, is_kaggle


def _git(args, repo):
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout


def git_commit(repo: Path = REPO_ROOT) -> dict:
    """Return {"commit": full hash, "dirty": True if there are uncommitted changes}.

    Returns commit "unknown" if the folder is not a git repo or has no commits yet.
    """
    try:
        commit = _git(["rev-parse", "HEAD"], repo).strip()
        dirty = _git(["status", "--porcelain"], repo).strip() != ""
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "dirty": True}
    return {"commit": commit, "dirty": dirty}


def dirty_files(repo: Path = REPO_ROOT) -> str:
    """`git status --short` output: the files with uncommitted changes."""
    try:
        return _git(["status", "--short"], repo).rstrip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "(not a git repo)"


def save_run_info(run_dir: Path, config: dict, git: dict | None = None) -> None:
    """Write config.yaml (resolved) and git.txt into run_dir.

    If the code had uncommitted changes, also save them as git_diff.patch, so the run
    can still be reproduced.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    git = git or git_commit()

    OmegaConf.save(OmegaConf.create(config), run_dir / "config.yaml", resolve=True)
    (run_dir / "git.txt").write_text(f"commit: {git['commit']}\ndirty: {git['dirty']}\n")

    if git["dirty"] and git["commit"] != "unknown":
        (run_dir / "git_diff.patch").write_text(_git(["diff", "HEAD"], REPO_ROOT))


def check_wandb_key() -> None:
    """Fail early with a clear message instead of W&B asking for a key mid-run."""
    if os.environ.get("WANDB_MODE") in ("offline", "disabled") or os.environ.get("WANDB_API_KEY"):
        return
    if is_kaggle():
        sys.exit("WANDB_API_KEY is not set. Add it in Kaggle (Add-ons > Secrets) and re-run cell 1.")
    print("No WANDB_API_KEY in .env, so W&B logs offline (files only).")
    os.environ["WANDB_MODE"] = "offline"
