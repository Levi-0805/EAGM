"""Unit tests for Stage 6A evaluation protocol fixes."""
from __future__ import annotations

import numpy as np
import torch

from training.eval_metrics import (
    fail_rate_vs_theta0,
    gradient_quality,
    paired_delta_returns,
    stage5e_budget_breakdown,
    theta_param_sat_frac,
)
from training.gradients import diagnose_grad_pair
from training.policy_opt import eval_physical_paired_vs_theta0, policy_optimize
from training.rollout import ACTION_LIMIT, actor, actor_raw_torch, physical_return_detailed


def test_fail_rate_uses_paired_delta_not_sign_of_return():
    # All returns negative (typical cost-like), but all improve vs θ0
    r0 = [-0.01, -0.02, -0.015]
    rf = [-0.005, -0.01, -0.008]
    fr = fail_rate_vs_theta0(rf, r0, eps=0.0)
    assert fr["fail_rate"] == 0.0
    assert fr["n_improved"] == 3
    # Old buggy rule np.mean(rf > 0) would also be 0 here; use a case that diverges:
    r0b = [0.01, 0.02, -0.01]
    rfb = [0.005, 0.015, 0.02]  # first two worsen (ΔR<0), third improves
    frb = fail_rate_vs_theta0(rfb, r0b, eps=0.0)
    assert frb["n_fail"] == 2
    assert abs(frb["fail_rate"] - 2 / 3) < 1e-9
    # Bug: mean(rf > 0) = 1.0, not equal to fail rate
    assert float(np.mean(np.asarray(rfb) > 0)) != frb["fail_rate"]


def test_paired_delta_length_mismatch_raises():
    try:
        paired_delta_returns([1.0, 2.0], [1.0])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_budget_breakdown_separates_train_val_iid():
    b = stage5e_budget_breakdown(
        n_train_eps=24, n_val_eps=6, n_iid_hold_eps=6, horizon=200,
        dir_label_train_steps=16128, ood_fd_diagnostic_steps=32256,
    )
    assert b["dynamics_train_steps"] == 4800
    assert b["dynamics_val_steps"] == 1200
    assert b["dynamics_iid_holdout_steps"] == 1200
    assert b["dynamics_collect_total_steps"] == 7200
    assert b["train_interaction_budget"] == 4800 + 16128
    assert b["ood_fd_diagnostic_steps"] == 32256
    # diagnostics not in train budget
    assert b["train_interaction_budget"] != b["dynamics_collect_total_steps"] + 16128


def test_gradient_quality_full_vs_stable():
    g_true = np.array([1.0, 2.0, 0.0, -1.0])
    g_hat = np.array([1.0, 2.0, 100.0, -1.0])  # unstable dim 2 is wrong
    mask = np.array([1.0, 1.0, 0.0, 1.0])
    q = gradient_quality(g_hat, g_true, mask)
    assert q["n_stable_dims"] == 3
    assert q["cosine_stable"] > q["cosine_full"]
    q2 = diagnose_grad_pair(g_hat, g_true, mask)
    assert q2["cosine_full"] == q["cosine_full"]


def test_theta_param_sat_separate_from_action_clip():
    th = np.ones(28)
    assert theta_param_sat_frac(th) == 1.0
    # Actor raw can exceed limits while θ is interior
    theta = torch.zeros(28)
    x = torch.zeros(28)
    c = torch.zeros(14)
    ref = torch.ones(14) * 10.0
    raw = actor_raw_torch(theta, x, c, ref)
    clipped = actor(theta, x, c, ref)
    assert float(raw.abs().max()) > ACTION_LIMIT
    assert float(clipped.abs().max()) <= ACTION_LIMIT + 1e-6


def test_policy_optimize_exposes_checkpoints_and_theta_sat():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("lower", torch.full((14,), -2.0))
            self.register_buffer("upper", torch.full((14,), 2.0))
            self.register_buffer("alpha", torch.tensor(0.5))
            self.net = torch.nn.Linear(28 + 14, 28)

        def forward(self, x, c_obs, a):
            y = x + 0.0 * self.net(torch.cat([x, a], -1))
            return y, torch.zeros(x.shape[0], 14)

    model = Tiny()
    start = dict(
        x=np.zeros(28),
        c_obs=np.zeros(29),
        prev=np.zeros(14),
        refs=np.zeros((16, 14)),
        x_hist=np.zeros((8, 28)),
        a_hist=np.zeros((8, 14)),
    )
    out = policy_optimize(model, [start], use_history=False, updates=8, lr=0.03, seed=0)
    assert "checkpoints" in out
    assert "theta0" in out["checkpoints"] and "theta_final" in out["checkpoints"]
    assert "theta_param_sat_frac" in out["curve"][0]
    assert "final_theta_param_sat_frac" in out


def test_privileged_exec_state_train_only_api():
    from controllers.partial_unknown import ControllerBConfig, PartialUnknownArmEnv

    env = PartialUnknownArmEnv(ControllerBConfig(gain=1.05, extra_lag=1, rate_limit=0.04, smooth_beta=0.2))
    env.reset(seed=0, delay=1)
    env.step(np.zeros(14))
    lab = env.privileged_labels()
    assert "exec_state" in lab
    es = lab["exec_state"]
    assert "vector" in es and es["vector"].ndim == 1
    assert es["gain"] == 1.05
    # known_config_vector remains oracle-only and separate
    assert env.known_config_vector().shape == (4,)
