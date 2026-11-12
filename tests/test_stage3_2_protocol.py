"""Stage 3.2 P0 tests: matched-θ gradients, limits, seeding."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "legacy_sim"))

from controllers.partial_unknown import ControllerBConfig, PartialUnknownArmEnv
from training.dataset import collect_trajectories, fit_stats, flatten_split, make_split
from training.gradients import sim_grad_masked
from training.seeds import seed_everything


def test_sim_grad_accepts_nonzero_theta():
    env = PartialUnknownArmEnv(ControllerBConfig(gain=1.05, extra_lag=1, smooth_beta=0.2, rate_limit=0.04))
    eps = collect_trajectories(env, n_episodes=1, horizon=30, seed=0, hist_len=4)
    start = eps[0]["rows"][10]["start"]
    th = np.linspace(-0.2, 0.2, 28)
    g0, m0, s0 = sim_grad_masked(env, start, np.zeros(28), horizon=4)
    g1, m1, s1 = sim_grad_masked(env, start, th, horizon=4)
    assert g0.shape == (28,)
    assert s0 > 0 and s1 > 0
    # Gradients at different θ should generally differ
    assert np.linalg.norm(g0 - g1) > 1e-8


def test_fit_stats_uses_env_limits():
    env = PartialUnknownArmEnv(ControllerBConfig())
    eps = collect_trajectories(env, n_episodes=2, horizon=20, seed=1, hist_len=4)
    rows = flatten_split(eps, [0, 1])
    stats = fit_stats(rows, env_lower=env.lower, env_upper=env.upper)
    assert np.allclose(stats["lower"], env.lower)
    assert np.allclose(stats["upper"], env.upper)
    assert not np.allclose(stats["lower"], stats["data_lower_margin"])


def test_seed_everything_reproducible_torch():
    seed_everything(123)
    a = torch.randn(10)
    seed_everything(123)
    b = torch.randn(10)
    assert torch.allclose(a, b)


if __name__ == "__main__":
    test_sim_grad_accepts_nonzero_theta()
    test_fit_stats_uses_env_limits()
    test_seed_everything_reproducible_torch()
    print("stage3_2_p0_ok")
