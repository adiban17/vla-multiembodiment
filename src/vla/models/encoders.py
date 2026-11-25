"""The two frozen pretrained encoders, and THE one image preprocessing function.

    preprocess_image()  uint8 camera frames -> 224x224 ImageNet-normalized float tensors.
                        Used by feature caching, training and evaluation. Never write a second copy.
                        The baselines use it too, with size=64 (BC-MLP) or 224 (ResNet-18).
    DinoV2Encoder       DINOv2-small: one image -> CLS + 16x16 patch vectors (384 each);
                        we average-pool the patches to 8x8, giving 65 x 384 per frame.
    MiniLMEncoder       all-MiniLM-L6-v2: one sentence -> one 384 vector (mean of token
                        vectors, then scaled to length 1, the same recipe as sentence-transformers).
    encode_frames()     batches of uint8 frames -> DINOv2 features as numpy. The one path used by
                        the feature cache and by live evaluation, so the two always match.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DINO_NAME = "facebook/dinov2-small"
MINILM_NAME = "sentence-transformers/all-MiniLM-L6-v2"
POOLED_GRID = 8  # 16x16 patches -> 8x8
FEATURE_DIM = 384


def preprocess_image(images, size: int = IMAGE_SIZE) -> torch.Tensor:
    """uint8 images -> float32 (N, 3, size, size), ready for DINOv2 (size 224) or a baseline.

    Accepts numpy or torch, one image (H, W, 3) or a batch (N, H, W, 3) or (N, 3, H, W).
    Steps: scale to [0, 1], bicubic resize to size x size, clamp (bicubic can overshoot), then
    subtract the ImageNet mean and divide by the std. Runs on the input tensor's device.

    Antialiasing is used only when shrinking (e.g. 128 -> 64 for BC-MLP), where it stops fine
    detail from turning into noise. Enlarging (96 or 128 -> 224, the DINOv2 path) never uses it,
    so the feature cache's preprocessing is exactly what it always was.
    """
    x = images if isinstance(images, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(images))
    if x.dtype != torch.uint8:
        raise TypeError(f"preprocess_image expects uint8 images (0..255), got {x.dtype}")
    if x.ndim == 3:
        x = x[None]
    if x.ndim != 4:
        raise ValueError(f"expected (H,W,3), (N,H,W,3) or (N,3,H,W), got shape {tuple(x.shape)}")
    if x.shape[-1] == 3:
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
    x = x.float() / 255.0
    shrink = size < min(x.shape[-2:])
    x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False, antialias=shrink).clamp_(0.0, 1.0)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def pool_patches(patches: torch.Tensor, grid: int = POOLED_GRID) -> torch.Tensor:
    """(N, P, D) patch vectors on a square grid (P = 16*16) -> (N, grid*grid, D) by averaging."""
    n, p, d = patches.shape
    side = int(round(p ** 0.5))
    if side * side != p:
        raise ValueError(f"{p} patches do not form a square grid")
    x = patches.transpose(1, 2).reshape(n, d, side, side)
    x = F.adaptive_avg_pool2d(x, grid)
    return x.reshape(n, d, grid * grid).transpose(1, 2)


class DinoV2Encoder(nn.Module):
    """DINOv2-small. forward(pixel_values (N,3,224,224)) -> (N, 65, 384): CLS + 8x8 pooled patches.

    Frozen by default: no gradients, and it stays in eval mode even if .train() is called
    (Phase 6 un-freezes it for full fine-tuning and LoRA).
    """

    def __init__(self, name: str = DINO_NAME, frozen: bool = True):
        super().__init__()
        from transformers import AutoModel

        self.name = name
        self.model = AutoModel.from_pretrained(name).float()
        self.revision = getattr(self.model.config, "_commit_hash", None)  # exact weights used
        self.frozen = frozen
        if frozen:
            self.model.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        tokens = self.model(pixel_values=pixel_values).last_hidden_state  # (N, 1 + 256, 384)
        return torch.cat([tokens[:, :1], pool_patches(tokens[:, 1:])], dim=1)

    def info(self) -> dict:
        return {"vision_model": self.name, "vision_revision": self.revision}


class MiniLMEncoder(nn.Module):
    """all-MiniLM-L6-v2. forward(list of sentences) -> (N, 384) unit-length vectors. Always frozen."""

    def __init__(self, name: str = MINILM_NAME):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).float()
        self.revision = getattr(self.model.config, "_commit_hash", None)
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, sentences: list[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        batch = self.tokenizer(list(sentences), padding=True, truncation=True, return_tensors="pt").to(device)
        tokens = self.model(**batch).last_hidden_state  # (N, L, 384)
        mask = batch["attention_mask"].unsqueeze(-1).to(tokens.dtype)
        mean = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # ignore padding tokens
        return F.normalize(mean, dim=-1)

    def encode(self, sentences: list[str]) -> np.ndarray:
        with torch.inference_mode():
            return self(sentences).float().cpu().numpy()

    def info(self) -> dict:
        return {"language_model": self.name, "language_revision": self.revision}


def encode_frames(encoder: DinoV2Encoder, frames, device: str, batch_size: int = 64) -> np.ndarray:
    """uint8 frames (N, H, W, 3) -> float32 numpy (N, 65, 384), in fp32, batch by batch."""
    frames = np.asarray(frames)
    out = np.empty((len(frames), POOLED_GRID * POOLED_GRID + 1, FEATURE_DIM), dtype=np.float32)
    with torch.inference_mode():
        for i in range(0, len(frames), batch_size):
            x = torch.from_numpy(np.ascontiguousarray(frames[i:i + batch_size])).to(device)
            out[i:i + len(x)] = encoder(preprocess_image(x)).float().cpu().numpy()
    return out
