"""Finite-difference policy gradients with stability masks at a given θ."""
from __future__ import annotations

import numpy as np

from training.rollout import physical_return


def sim_grad_masked(
    env,
    start,
    theta,
    *,
    eps: float = 0.05,
    scales=(0.5, 1.0, 2.0),
    horizon: int = 8,
):
    """Central-difference ∇_θ J_sim at the provided θ (not hard-coded zero).

    Returns (g, stable_mask, control_steps_used).
    """
    theta = np.asarray(theta, dtype=np.float64).reshape(28)
    out, stable = [], []
    steps = 0
    for i in range(28):
        e = np.zeros(28)
        e[i] = 1.0
        vals = []
        for sc in scales:
            ep = eps * sc
            plus, _ = physical_return(env, theta + ep * e, start, horizon=horizon)
            minus, _ = physical_return(env, theta - ep * e, start, horizon=horizon)
            vals.append((plus - minus) / (2 * ep))
            steps += 2 * horizon
        v = vals[1]
        out.append(v)
        spread = max(vals) - min(vals)
        stable.append(spread <= max(1e-9, 0.1 * abs(v)))
    return np.asarray(out, dtype=np.float64), np.asarray(stable, dtype=np.float64), steps
