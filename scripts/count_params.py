"""Print the trainable parameter count of each full-size conditioning variant (CPU, no data).

    .venv/bin/python scripts/count_params.py
"""

import torch

from vla.models.conditioning import VARIANTS
from vla.models.policy import build_policy, count_params
from vla.utils.config import load_config


def main() -> None:
    counts = {}
    for cond in VARIANTS:
        torch.manual_seed(0)
        counts[cond] = count_params(build_policy(load_config([f"cond={cond}"])))
    print("Trainable parameters, full size (8 layers, width 384):")
    print(f"  {'variant':10s} {'parameters':>12s} {'vs none':>10s}")
    for cond, n in counts.items():
        print(f"  {cond:10s} {n:>12,d} {n - counts['none']:>+10,d}   ({n / 1e6:.2f}M)")


if __name__ == "__main__":
    main()
