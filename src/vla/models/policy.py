"""The full VLA policy (frozen-encoder path) and its loss.

One sample, step by step (spec "One sample, step by step"; B = batch size):

    vision  (B, 2, 65, 384)  cached DINOv2 features of frames t-1, t -> flatten to 130 tokens -> Linear
    lang    (B, 384)         MiniLM sentence vector                  -> Linear -> 1 token
    state   (B, 8)           normalized state                        -> Linear -> 1 token
    [token]                  variant A only: the robot's learned token        -> 1 token
    queries (16, 384)        learned "what to do at step k" slots             -> 16 tokens
    => (B, 149 or 148, 384) -> trunk (FiLM-modulated in variant B) -> last 16 outputs
    => head MLP (two heads in variant C) -> (B, 16, 4) actions, in normalized units

Position and type embeddings tell the trunk what each token is: every vision token gets a
learned position (which of the 65 slots: CLS or one of the 8x8 patches) plus a type for its
frame (t-1 or t); language, state and queries get a type each. The 16 queries and the robot
token are learned vectors, so they are already distinct.

Loss: L1 over real action dims and real time steps only (spec formula), computed as a mean per
robot, then the two robot means are averaged. A plain mean over all valid numbers would count
FetchPush twice as much as PushT (4 action dims vs 2); averaging per robot weights them equally.
"""

import torch
from torch import nn

from vla.models.conditioning import (
    N_ROBOTS,
    VARIANTS,
    ActionHead,
    EmbodimentToken,
    FiLMGenerator,
    SeparateHeads,
)
from vla.models.trunk import Trunk

ROBOT_NAMES = ("pusht", "fetch")  # index = embodiment_id (vla.data.dataset.EMBODIMENT_ID)


# --- loss ---------------------------------------------------------------------


def l1_sums(pred, target, dim_mask, is_pad):
    """Per-sample sum of |pred - target| over valid elements, and the number of valid elements.

    pred, target (B, 16, 4); dim_mask (B, 4) True for real action dims; is_pad (B, 16) True for
    steps past the episode end. Invalid elements are dropped with torch.where (not multiplied
    by 0), so even a NaN there cannot leak into the loss.
    """
    valid = dim_mask[:, None, :] & ~is_pad[:, :, None]
    err = torch.where(valid, (pred.float() - target.float()).abs(), 0.0)
    return err.sum(dim=(1, 2)), valid.sum(dim=(1, 2)).float()


def plain_masked_l1(pred, target, dim_mask, is_pad):
    """The spec formula: one mean over every valid element of the batch (kept for comparison)."""
    sums, counts = l1_sums(pred, target, dim_mask, is_pad)
    return sums.sum() / counts.sum().clamp(min=1)


def balanced_l1(pred, target, dim_mask, is_pad, embodiment_id, n_robots: int = N_ROBOTS):
    """Equal-weight loss: masked L1 mean for each robot, then the average over the robots present.

    Returns (loss, sums (n_robots,), counts (n_robots,)). sums / counts are each robot's mean; the
    raw sums and counts let validation add up many batches exactly.
    """
    sums, counts = l1_sums(pred, target, dim_mask, is_pad)
    robot_sums = torch.zeros(n_robots, device=sums.device).index_add(0, embodiment_id, sums)
    robot_counts = torch.zeros(n_robots, device=sums.device).index_add(0, embodiment_id, counts)
    present = (robot_counts > 0).float()
    means = robot_sums / robot_counts.clamp(min=1)
    loss = (means * present).sum() / present.sum().clamp(min=1)
    return loss, robot_sums, robot_counts


# --- model --------------------------------------------------------------------


class VLAPolicy(nn.Module):
    def __init__(self, cond: str, width: int = 384, depth: int = 8, heads: int = 6, ffn: int = 1536,
                 dropout: float = 0.1, n_queries: int = 16, film_dim: int = 64, vision_dim: int = 384,
                 lang_dim: int = 384, state_dim: int = 8, action_dim: int = 4, n_vision_tokens: int = 65,
                 n_frames: int = 2):
        super().__init__()
        if cond not in VARIANTS:
            raise ValueError(f"unknown conditioning {cond!r}; choose from {VARIANTS}")
        self.cond = cond
        self.n_queries, self.n_frames, self.n_vision_tokens = n_queries, n_frames, n_vision_tokens

        self.vision_proj = nn.Linear(vision_dim, width)
        self.lang_proj = nn.Linear(lang_dim, width)
        self.state_proj = nn.Linear(state_dim, width)
        self.queries = nn.Parameter(torch.zeros(n_queries, width))
        self.vision_pos = nn.Parameter(torch.zeros(n_vision_tokens, width))
        # token types: 0 language, 1 state, 2..2+n_frames-1 vision frames, then queries
        self.type_lang, self.type_state, self.type_query = 0, 1, 2 + n_frames
        self.type_embed = nn.Embedding(3 + n_frames, width)

        self.trunk = Trunk(depth, width, heads, ffn, dropout)
        self.token = EmbodimentToken(width) if cond == "token" else None
        self.film = FiLMGenerator(depth, width, film_dim) if cond == "film" else None
        self.head = SeparateHeads(width) if cond == "sep_heads" else ActionHead(width, action_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        """Small random weights (std 0.02, as in GPT/ViT), zero biases; FiLM output stays zero."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.normal_(self.queries, std=0.02)
        nn.init.normal_(self.vision_pos, std=0.02)
        if self.film is not None:
            self.film.zero_init()

    def embed(self, batch: dict) -> torch.Tensor:
        """Build the token sequence (B, L, width): [robot token], language, state, vision, queries."""
        vision = batch["vision"]                                   # (B, F, 65, 384)
        b, n_frames, n_tok, _ = vision.shape
        v = self.vision_proj(vision)                               # (B, F, 65, D)
        frame_types = self.type_embed.weight[2:2 + n_frames]       # (F, D)
        v = v + self.vision_pos[None, None] + frame_types[None, :, None]
        v = v.reshape(b, n_frames * n_tok, -1)

        lang = self.lang_proj(batch["lang"])[:, None] + self.type_embed.weight[self.type_lang]
        state = self.state_proj(batch["state"])[:, None] + self.type_embed.weight[self.type_state]
        queries = (self.queries + self.type_embed.weight[self.type_query]).expand(b, -1, -1)
        parts = [lang, state, v, queries]
        if self.token is not None:
            parts.insert(0, self.token(batch["embodiment_id"]).expand(b, -1, -1))
        return torch.cat([p.to(v.dtype) for p in parts], dim=1)  # same dtype under fp16 autocast

    def forward(self, batch: dict) -> torch.Tensor:
        """-> (B, 16, 4) predicted actions in normalized units (PushT: only the first 2 matter)."""
        robot = batch["embodiment_id"]
        x = self.embed(batch)
        film = self.film(robot) if self.film is not None else None
        h = self.trunk(x, film)[:, -self.n_queries:]
        return self.head(h, robot)

    def loss(self, batch: dict) -> dict:
        """{"loss": balanced L1, "loss_pusht", "loss_fetch", "sums", "counts", "pred"}."""
        pred = self(batch)
        loss, sums, counts = balanced_l1(pred, batch["actions"], batch["action_dim_mask"],
                                         batch["action_is_pad"], batch["embodiment_id"])
        out = {"loss": loss, "sums": sums, "counts": counts, "pred": pred}
        for i, name in enumerate(ROBOT_NAMES):
            out[f"loss_{name}"] = sums[i] / counts[i].clamp(min=1)
        return out


def build_policy(cfg) -> VLAPolicy:
    """The policy described by a config from vla.utils.config.load_config."""
    m = cfg.model
    return VLAPolicy(cfg.cond, width=m.width, depth=m.depth, heads=m.heads, ffn=m.ffn, dropout=m.dropout,
                     n_queries=m.n_queries, film_dim=m.film_dim)


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
