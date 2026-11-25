"""Print the trainable parameter count of each full-size model (CPU, no data):
the 4 conditioning variants, the 2 baselines and the Phase 6 adaptation regimes.

    .venv/bin/python scripts/count_params.py
"""

import torch

from vla.models.conditioning import VARIANTS
from vla.models.policy import build_model, count_params, encoder_params
from vla.utils.config import load_config


def main() -> None:
    counts = {}
    for cond in VARIANTS:
        torch.manual_seed(0)
        counts[cond] = count_params(build_model(load_config([f"cond={cond}"])))
    print("Trainable parameters, full size (8 layers, width 384):")
    print(f"  {'variant':10s} {'parameters':>12s} {'vs none':>10s}")
    for cond, n in counts.items():
        print(f"  {cond:10s} {n:>12,d} {n - counts['none']:>+10,d}   ({n / 1e6:.2f}M)")

    print("\nBaselines (one robot each; the count is the same for both robots):")
    for name in ("bc_mlp", "resnet18"):
        cfg = load_config([f"baseline={name}", "robot=pusht", "bc.pretrained=false"])  # same shapes, no download
        n = count_params(build_model(cfg))
        print(f"  {name:10s} {n:>12,d}   ({n / 1e6:.2f}M)  images {cfg.bc.image_size}x{cfg.bc.image_size}, "
              f"lr {cfg.optim.lr:g}, batch {cfg.data.batch_size}")

    print("\nAdaptation regimes on sep_heads (loads DINOv2-small from the Hugging Face cache):")
    print(f"  {'adapt':15s} {'DINOv2':>12s} {'rest':>12s} {'total':>12s}  learning rates (DINOv2 / rest), batch")
    for adapt in ("frozen_matched", "lora16", "full_ft"):
        torch.manual_seed(0)
        cfg = load_config(["cond=sep_heads", f"adapt={adapt}"])
        model = build_model(cfg)
        enc, total = encoder_params(model), count_params(model)
        enc_lr = f"{cfg.encoder.lr:g}" if cfg.encoder.lr is not None else "frozen"
        print(f"  {adapt:15s} {enc:>12,d} {total - enc:>12,d} {total:>12,d}  {enc_lr} / {cfg.optim.lr:g}, "
              f"batch {cfg.data.batch_size}")


if __name__ == "__main__":
    main()
