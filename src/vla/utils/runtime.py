"""Where am I running, where do files go, and which device to use.

The same code runs on the Mac (CPU tests) and on Kaggle (GPU). These helpers hide the
differences, so no other file needs to check "am I on Kaggle?".
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # src/vla/utils/runtime.py -> repo root

KAGGLE_OUTPUTS = Path("/kaggle/working/outputs")  # saved when a Kaggle notebook version is committed
KAGGLE_INPUTS = Path("/kaggle/input")  # earlier outputs, attached as Kaggle datasets


def is_kaggle() -> bool:
    """True inside a Kaggle notebook."""
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/working").is_dir()


def outputs_dir() -> Path:
    """Folder for new outputs. Kaggle: /kaggle/working/outputs. Mac: ./outputs (gitignored).

    The VLA_OUTPUTS_DIR environment variable overrides both (used by tests).
    """
    if os.environ.get("VLA_OUTPUTS_DIR"):
        path = Path(os.environ["VLA_OUTPUTS_DIR"])
    elif is_kaggle():
        path = KAGGLE_OUTPUTS
    else:
        path = REPO_ROOT / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def inputs_dir() -> Path:
    """Folder holding earlier outputs. Kaggle: /kaggle/input. Mac: ./outputs.

    The VLA_INPUTS_DIR environment variable overrides both.
    """
    if os.environ.get("VLA_INPUTS_DIR"):
        return Path(os.environ["VLA_INPUTS_DIR"])
    return KAGGLE_INPUTS if is_kaggle() else REPO_ROOT / "outputs"


def get_device() -> str:
    """"cuda" if an NVIDIA GPU is available, else "cpu". Never "mps" (project rule)."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_dotenv_if_local() -> None:
    """On the Mac, load secrets such as WANDB_API_KEY from the gitignored .env file.

    On Kaggle, secrets come from Kaggle Secrets instead, so this does nothing there.
    """
    env_file = REPO_ROOT / ".env"
    if is_kaggle() or not env_file.is_file():
        return
    from dotenv import load_dotenv  # Mac-only dependency

    load_dotenv(env_file, override=False)
