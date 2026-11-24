"""Experiment tracking: record exactly which code and config produced each run.

Rule: every run saves its resolved config and git commit into its run folder.
W&B logging helpers will be added here in Phase 3.
"""

import subprocess
from pathlib import Path

from omegaconf import OmegaConf

from vla.utils.runtime import REPO_ROOT


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
