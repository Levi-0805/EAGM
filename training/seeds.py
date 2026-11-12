"""Deterministic seeding for paired EAGM experiments."""
from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> dict:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA if present)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic CPU matmul where available
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    return dict(
        seed=seed,
        numpy=seed,
        torch=seed,
        cuda=seed if torch.cuda.is_available() else None,
        deterministic_algorithms=True,
    )


def fork_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g
