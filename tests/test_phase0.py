"""Phase 0 checks. Written before the Phase 0 code, so "done" has a definition.

Run on the Mac (CPU only):  .venv/bin/python -m pytest -q
"""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_kaggle_job():
    """Import scripts/kaggle_job.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("kaggle_job", REPO / "scripts" / "kaggle_job.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _requirement_lines(name):
    lines = (REPO / name).read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


# --- torch on CPU -----------------------------------------------------------

def test_torch_imports_and_runs_on_cpu():
    import torch

    x = torch.ones(3, device="cpu") + 1
    assert x.device.type == "cpu"
    assert x.tolist() == [2.0, 2.0, 2.0]


def test_get_device_never_returns_mps(monkeypatch):
    import torch
    from vla.utils.runtime import get_device

    # Pretend this machine has MPS but no CUDA (like an M1 Mac).
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert get_device() == "cpu"


# --- pinned dependencies ----------------------------------------------------

@pytest.mark.parametrize("name", ["requirements.txt", "requirements-kaggle.txt"])
def test_requirements_are_pinned(name):
    for line in _requirement_lines(name):
        if line == "-e .":
            continue
        assert re.fullmatch(r"[A-Za-z0-9_.\-\[\]]+==[A-Za-z0-9_.+\-]+", line), f"not pinned: {line!r}"


def test_kaggle_requirements_skip_preinstalled_packages():
    names = {re.split(r"[=<>\[]", ln)[0].lower() for ln in _requirement_lines("requirements-kaggle.txt")}
    for preinstalled in ["torch", "torchvision", "transformers", "numpy"]:
        assert preinstalled not in names, f"{preinstalled} must not be in requirements-kaggle.txt"


# --- git commit tracking ----------------------------------------------------

def test_git_commit_returns_full_hash(tmp_path):
    from vla.utils.tracking import git_commit

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "a.txt").write_text("hi")
    git("add", "a.txt")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")

    info = git_commit(tmp_path)
    assert re.fullmatch(r"[0-9a-f]{40}", info["commit"])
    assert info["dirty"] is False


# --- job registry and the hello job ------------------------------------------

def test_hello_job_is_registered():
    kaggle_job = _load_kaggle_job()
    assert "hello" in kaggle_job.JOBS


def test_hello_job_saves_config_and_git(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("VLA_OUTPUTS_DIR", str(tmp_path))
    kaggle_job = _load_kaggle_job()

    run_dir = kaggle_job.JOBS["hello"]()

    assert Path(run_dir).parent == tmp_path
    assert (Path(run_dir) / "config.yaml").is_file()
    assert (Path(run_dir) / "git.txt").read_text().strip() != ""


def test_kaggle_job_cli_lists_jobs():
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "kaggle_job.py"), "--list"],
        capture_output=True, text=True, check=True,
    )
    assert "hello" in out.stdout


# --- secrets and big files stay out of git ------------------------------------

def test_gitignore_covers_secrets_and_outputs():
    ignored = set((REPO / ".gitignore").read_text().split())
    # Top-level only (leading /), so the code folder src/vla/data/ is still tracked.
    for entry in [".env", "/data/", "/outputs/", "/runs/", ".venv/"]:
        assert entry in ignored, f"{entry} missing from .gitignore"


def test_code_folders_are_not_gitignored():
    paths = ["src/vla/data/dataset.py", "src/vla/models/policy.py", "configs/base.yaml"]
    out = subprocess.run(["git", "check-ignore", *paths], cwd=REPO, capture_output=True, text=True)
    assert out.stdout.strip() == "", f"wrongly ignored: {out.stdout}"
