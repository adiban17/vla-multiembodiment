"""How the policy learns which robot it drives (spec "Conditioning variants").

    none       not given: the trunk must guess the robot from the image and the instruction
    token      (A) one learned 384-number vector per robot, added to the token sequence   (+768)
    film       (B) robot ID -> 64-number embedding -> gamma and beta for both LayerNorms of
               all 8 layers, starting at zero                                           (~+0.8M)
    sep_heads  (C) one shared trunk, a different output MLP per robot (2 or 4 outputs)   (~+0.15M)

Everything else (trunk, inputs, loss) is identical across variants, so a difference in success
can be pinned on where the robot's identity enters.
"""

import torch
import torch.nn.functional as F
from torch import nn

VARIANTS = ("none", "token", "film", "sep_heads")
N_ROBOTS = 2
ROBOT_ACTION_DIMS = (2, 4)  # PushT (id 0), FetchPush (id 1)


class EmbodimentToken(nn.Module):
    """Variant A: robot ID -> one extra token (B, 1, D)."""

    def __init__(self, width: int, n_robots: int = N_ROBOTS):
        super().__init__()
        self.embed = nn.Embedding(n_robots, width)

    def forward(self, embodiment_id: torch.Tensor) -> torch.Tensor:
        return self.embed(embodiment_id)[:, None, :]


class FiLMGenerator(nn.Module):
    """Variant B: robot ID -> (B, depth, 4, D): gamma1, beta1, gamma2, beta2 for every layer.

    The output layer starts at exactly zero (weight and bias), so at the start of training
    gamma = beta = 0 and the trunk behaves like the unconditioned model. The zero weight still
    gets gradients (they depend on the random robot embedding), so FiLM learns from step 1.
    """

    def __init__(self, depth: int, width: int, dim: int = 64, n_robots: int = N_ROBOTS):
        super().__init__()
        self.depth, self.width = depth, width
        self.embed = nn.Embedding(n_robots, dim)
        self.proj = nn.Linear(dim, depth * 4 * width)
        self.zero_init()

    def zero_init(self) -> None:
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, embodiment_id: torch.Tensor) -> torch.Tensor:
        out = self.proj(self.embed(embodiment_id))
        return out.view(-1, self.depth, 4, self.width)


class ActionHead(nn.Module):
    """2-layer MLP applied to each of the 16 query outputs: width -> width -> out_dim."""

    def __init__(self, width: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, out_dim))

    def forward(self, h: torch.Tensor, embodiment_id: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(h)


class SeparateHeads(nn.Module):
    """Variant C: one ActionHead per robot. Output is always (B, 16, 4); PushT's 2 outputs are
    padded with zeros (the loss ignores those dims anyway). Both heads run on every sample and
    torch.where keeps the right one, so the unused head gets zero gradient from that sample."""

    def __init__(self, width: int, action_dims=ROBOT_ACTION_DIMS):
        super().__init__()
        self.out_dim = max(action_dims)
        self.heads = nn.ModuleList(ActionHead(width, d) for d in action_dims)

    def forward(self, h: torch.Tensor, embodiment_id: torch.Tensor) -> torch.Tensor:
        out = None
        for robot, head in enumerate(self.heads):
            y = head(h)
            y = F.pad(y, (0, self.out_dim - y.shape[-1]))
            out = y if out is None else torch.where((embodiment_id == robot)[:, None, None], y, out)
        return out
