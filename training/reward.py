"""Unified diagnostic reward shared by imagined rollouts and physical eval.

Matches vendor/legacy_sim/experiment.py::reward / physical:

  R = -mean((q - q*)^2) - 0.0005 * mean(qd^2) - 0.05 * mean((a - a_prev)^2)
  J = sum_k 0.99^k R_{t+k}
"""
from __future__ import annotations

import numpy as np
import torch

DISCOUNT = 0.99
W_VEL = 0.0005
W_DA = 0.05


def reward_torch(x, a, ref, prev):
    """x,a,ref,prev: [... , dim]. Returns [...]."""
    return (
        -((x[..., :14] - ref) ** 2).mean(-1)
        - W_VEL * (x[..., 14:] ** 2).mean(-1)
        - W_DA * ((a - prev) ** 2).mean(-1)
    )


def reward_numpy(x, a, ref, prev) -> float:
    x = np.asarray(x, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    prev = np.asarray(prev, dtype=np.float64)
    return float(
        -np.mean((x[:14] - ref) ** 2)
        - W_VEL * np.mean(x[14:] ** 2)
        - W_DA * np.mean((a - prev) ** 2)
    )


def discounted_sum_torch(rewards, discount: float = DISCOUNT):
    total = rewards[0] * 0
    for k, r in enumerate(rewards):
        total = total + (discount ** k) * r
    return total
