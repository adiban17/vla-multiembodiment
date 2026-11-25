"""The two per-robot baselines (docs/spec.md "Baselines"). Each trains on ONE robot.

    BC-MLP      no pretrained vision at all: frames t-1 and t, resized to 64x64, flattened with
                the state into a 3-layer MLP (512 wide). Trained from scratch. ~12.9M parameters.
                Isolates: what any pretrained vision buys.
    ResNet-18   ImageNet-pretrained ResNet-18, fine-tuned end to end. Each frame (224x224) ->
                one 512-number feature; [feature t-1, feature t, state] -> the same MLP head.
                ~12.0M parameters. Isolates: a pretrained CNN vs frozen DINOv2 + shared transformer.

Same inputs, outputs and loss as the main model, so only the model differs:
    batch["pixels"]  (B, 2, H, W, 3) uint8 frames t-1 and t, straight from images.npy or the env
    batch["state"]   (B, 8) normalized state (PushT: 2 real + 6 zeros)
    -> (B, 16, 4) actions in normalized units (PushT: only the first 2 dims matter; the loss
       masks the rest, exactly as for the main model)

Image preprocessing runs INSIDE the model, with the one preprocess_image() (size 64 or 224).
Training and evaluation hand the model the same uint8 frames, so they cannot drift apart, and
on Kaggle the resizing happens on the GPU instead of in the 2 data-loader workers.
"""

import torch
from torch import nn

from vla.models.encoders import preprocess_image
from vla.models.policy import chunk_loss

CHUNK = 16
ACTION_DIM = 4
STATE_DIM = 8
N_FRAMES = 2


def mlp_head(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    """3 Linear layers with ReLU between them: in -> hidden -> hidden -> out."""
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, out_dim))


class _Baseline(nn.Module):
    """Shared parts: preprocessing, the MLP head and the loss."""

    def __init__(self, image_size: int):
        super().__init__()
        self.image_size = image_size

    def frames(self, pixels: torch.Tensor) -> torch.Tensor:
        """(B, F, H, W, 3) uint8 -> (B*F, 3, S, S) ImageNet-normalized float32. Always fp32:
        resizing is done outside fp16 autocast so it matches the CPU/eval path exactly."""
        b, f = pixels.shape[:2]
        with torch.autocast(device_type=pixels.device.type, enabled=False):
            return preprocess_image(pixels.reshape(b * f, *pixels.shape[2:]), size=self.image_size)

    def loss(self, batch: dict) -> dict:
        return chunk_loss(self, batch)


class BCMLP(_Baseline):
    def __init__(self, image_size: int = 64, hidden: int = 512, n_frames: int = N_FRAMES,
                 state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM, chunk: int = CHUNK):
        super().__init__(image_size)
        self.chunk, self.action_dim = chunk, action_dim
        in_dim = n_frames * 3 * image_size * image_size + state_dim
        self.head = mlp_head(in_dim, hidden, chunk * action_dim)

    def forward(self, batch: dict) -> torch.Tensor:
        pixels = batch["pixels"]
        b = pixels.shape[0]
        x = self.frames(pixels).reshape(b, -1)  # (B, 2*3*S*S): both frames, flattened
        x = torch.cat([x, batch["state"].float()], dim=1)
        return self.head(x).reshape(b, self.chunk, self.action_dim)


class ResNet18BC(_Baseline):
    FEATURE_DIM = 512

    def __init__(self, image_size: int = 224, hidden: int = 512, pretrained: bool = True, n_frames: int = N_FRAMES,
                 state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM, chunk: int = CHUNK):
        super().__init__(image_size)
        from torchvision.models import ResNet18_Weights, resnet18

        self.chunk, self.action_dim, self.n_frames = chunk, action_dim, n_frames
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone.fc = nn.Identity()  # keep the 512-number pooled feature, drop the 1000 classes
        self.pretrained = pretrained
        self.head = mlp_head(n_frames * self.FEATURE_DIM + state_dim, hidden, chunk * action_dim)

    def forward(self, batch: dict) -> torch.Tensor:
        pixels = batch["pixels"]
        b = pixels.shape[0]
        feats = self.backbone(self.frames(pixels))           # (B*2, 512): both frames in one pass
        x = torch.cat([feats.reshape(b, -1).float(), batch["state"].float()], dim=1)
        return self.head(x).reshape(b, self.chunk, self.action_dim)


def build_baseline(cfg) -> _Baseline:
    """The baseline named by cfg.baseline, sized by cfg.bc."""
    bc = cfg.bc
    if cfg.baseline == "bc_mlp":
        return BCMLP(image_size=int(bc.image_size), hidden=int(bc.hidden))
    if cfg.baseline == "resnet18":
        return ResNet18BC(image_size=int(bc.image_size), hidden=int(bc.hidden), pretrained=bool(bc.pretrained))
    raise ValueError(f"unknown baseline {cfg.baseline!r}")
