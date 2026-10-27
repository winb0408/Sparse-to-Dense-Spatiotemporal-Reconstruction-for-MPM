"""Deterministic seeding.

Reproducibility here means: same seed + same config + same data => same
numbers. CUDA kernels that are not bitwise deterministic are allowed to differ
in the last bits, so results are reported across seeds rather than trusting a
single run to be identical.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["set_seed", "seed_worker"]


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker initialiser: derive a distinct but reproducible seed."""
    base = torch.initial_seed() % 2 ** 32
    np.random.seed(base)
    random.seed(base)
    os.environ.setdefault("PYTHONHASHSEED", str(base))
