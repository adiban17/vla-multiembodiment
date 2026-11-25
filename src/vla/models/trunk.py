"""The transformer trunk shared by both robots (spec "Model sizes").

8 layers, width 384, 6 attention heads, feed-forward width 1536, pre-LayerNorm, GELU, dropout 0.1,
and NO causal mask: every token can look at every other token (the 16 action queries can read the
image, and the image tokens can read the instruction).

Pre-LayerNorm means each sub-layer normalizes its input and adds its output back:
    x = x + Attention(LN1(x))
    x = x + FFN(LN2(x))

Optional FiLM (variant B) changes both LayerNorms of every layer, per sample:
    LN(x)  ->  (1 + gamma) * LN(x) + beta
gamma and beta come from the robot ID (vla.models.conditioning.FiLMGenerator). When they are
zero, the layer is exactly the unconditioned layer.
"""

import torch
import torch.nn.functional as F
from torch import nn


def film(h: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """h: (B, L, D), gamma/beta: (B, D), one pair per sample, shared by all its tokens."""
    return h * (1 + gamma[:, None, :]) + beta[:, None, :]


class Block(nn.Module):
    """One pre-LayerNorm transformer layer."""

    def __init__(self, width: int, heads: int, ffn: int, dropout: float):
        super().__init__()
        if width % heads:
            raise ValueError(f"width {width} is not divisible by {heads} heads")
        self.heads = heads
        self.ln1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.ln2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, ffn), nn.GELU(), nn.Linear(ffn, width))
        self.drop = nn.Dropout(dropout)

    def attention(self, h: torch.Tensor) -> torch.Tensor:
        b, length, width = h.shape
        qkv = self.qkv(h).view(b, length, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B, heads, L, head_dim)
        out = F.scaled_dot_product_attention(q, k, v)  # no mask: all tokens see all tokens
        return self.proj(out.transpose(1, 2).reshape(b, length, width))

    def forward(self, x: torch.Tensor, film_params: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, L, D). film_params: (B, 4, D) = gamma1, beta1, gamma2, beta2, or None."""
        h = self.ln1(x)
        if film_params is not None:
            h = film(h, film_params[:, 0], film_params[:, 1])
        x = x + self.drop(self.attention(h))
        h = self.ln2(x)
        if film_params is not None:
            h = film(h, film_params[:, 2], film_params[:, 3])
        return x + self.drop(self.ffn(h))


class Trunk(nn.Module):
    """A stack of `depth` Blocks plus a final LayerNorm (never FiLM-modulated)."""

    def __init__(self, depth: int = 8, width: int = 384, heads: int = 6, ffn: int = 1536, dropout: float = 0.1):
        super().__init__()
        self.depth, self.width = depth, width
        self.blocks = nn.ModuleList(Block(width, heads, ffn, dropout) for _ in range(depth))
        self.ln_out = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, film: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, L, D). film: (B, depth, 4, D) or None."""
        for i, block in enumerate(self.blocks):
            x = block(x, None if film is None else film[:, i])
        return self.ln_out(x)
