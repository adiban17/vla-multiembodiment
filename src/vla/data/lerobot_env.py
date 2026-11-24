"""Build the separate Python 3.12 env for lerobot, and run the PushT export inside it.

Why separate: lerobot 0.7.0 needs Python >= 3.12 and pins its own torch and numpy. Installing it
next to Kaggle's GPU PyTorch could replace that PyTorch. Instead uv makes a fresh env (with its
own Python download and a small CPU-only torch), and we call it as a subprocess.
"""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

from vla.utils.runtime import REPO_ROOT

REQUIREMENTS = REPO_ROOT / "requirements-lerobot.txt"
PYTHON_VERSION = "3.12"


def _uv() -> list:
    """The uv command: the pip-installed one on Kaggle, or the one on PATH on the Mac."""
    if importlib.util.find_spec("uv") is not None:
        return [sys.executable, "-m", "uv"]
    if shutil.which("uv"):
        return [shutil.which("uv")]
    raise RuntimeError("uv not found: it is installed by requirements-kaggle.txt (Kaggle) or ~/.local/bin/uv (Mac)")


def _run(cmd, **kwargs):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO_ROOT, **kwargs)


def ensure_lerobot_env(venv_dir) -> Path:
    """Create (or reuse) the env and install requirements-lerobot.txt. Returns its python."""
    venv_dir = Path(venv_dir)
    python = venv_dir / "bin" / "python"
    if not python.exists():
        _run([*_uv(), "venv", "--managed-python", "--python", PYTHON_VERSION, venv_dir])
    _run([*_uv(), "pip", "install", "--quiet", "--python", python, "--torch-backend", "cpu", "-r", REQUIREMENTS])
    return python


def run_pusht_export(out_dir, venv_dir, max_episodes: int | None = None) -> None:
    """Export lerobot/pusht to out_dir using the separate env."""
    python = ensure_lerobot_env(venv_dir)
    cmd = [python, "-m", "vla.data.pusht_loader", "--out", Path(out_dir)]
    if max_episodes:
        cmd += ["--max-episodes", max_episodes]
    _run(cmd)
