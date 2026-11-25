"""One integer seeds Python, NumPy, PyTorch, CUDA and the data-loader workers (spec "Seeds").

GPU kernels are not bit-exact, so reruns on a GPU match closely rather than exactly.
"""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> torch.Generator:
    """Seed every random number generator. Returns a torch.Generator for DataLoaders."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA
    return torch.Generator().manual_seed(seed)


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker its own Python/NumPy seed, derived from torch's worker seed
    (which comes from the DataLoader's seeded generator), so workers differ but runs repeat."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
