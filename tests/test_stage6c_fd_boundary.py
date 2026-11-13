"""Unit tests: bound-aware FD, projection loss, Dir label budget accounting."""
from __future__ import annotations

import numpy as np
import torch

from training.gradients import (
    BOUND_EPS,
    sim_grad_masked,
    sim_grad_masked_detailed,
)
from training.policy_opt import outward_pressure_frac, policy_optimize, rho_proj


class _StubEnv:
    """Quadratic return in θ so FD has a known analytic gradient at interior pts."""

    def __init__(self):
        self.lower = np.full(14, -2.0)
        self.upper = np.full(14, 2.0)

    def reset(self, *a, **k):
        return None

    def step(self, a):
        return None


# Monkeypatch physical_return used inside gradients
def _install_quad_return(monkey_theta_star=None):
    import training.gradients as gmod

    star = np.zeros(28) if monkey_theta_star is None else np.asarray(monkey_theta_star)

    def fake_physical_return(env, theta, start, horizon=8):
        th = np.asarray(theta, dtype=np.float64).reshape(28)
        # J = -||θ - star||^2  →  ∇J = -2(θ - star)
        val = -float(np.sum((th - star) ** 2))
        return val, {}

    gmod.physical_return = fake_physical_return
    return fake_physical_return


def test_feasible_fd_interior_matches_central():
    _install_quad_return()
    env = _StubEnv()
    start = {}
    theta = np.zeros(28)
    g0, m0, s0 = sim_grad_masked(env, start, theta, feasible=False, horizon=1, scales=(1.0, 1.0, 1.0), eps=0.05)
    g1, m1, s1, meta = sim_grad_masked_detailed(
        env, start, theta, feasible=True, horizon=1, scales=(1.0, 1.0, 1.0), eps=0.05
    )
    assert meta["n_boundary"] == 0
    assert meta["n_central"] == 28
    assert np.allclose(g0, g1, atol=1e-6)
    # Analytic ∇J = 0 at θ=star=0
    assert np.allclose(g1, 0.0, atol=1e-6)


def test_feasible_fd_at_upper_bound_uses_backward():
    _install_quad_return(monkey_theta_star=np.zeros(28))
    env = _StubEnv()
    theta = np.ones(28)  # all at +1
    g, mask, steps, meta = sim_grad_masked_detailed(
        env, {}, theta, feasible=True, horizon=1, scales=(0.5, 1.0, 2.0), eps=0.05
    )
    assert meta["n_boundary"] == 28
    assert meta["n_onesided"] == 28
    assert all(m == "backward" for m in meta["fd_mode"])
    # At θ=1, star=0: ∇J = -2(1-0) = -2; one-sided should be near -2
    assert np.allclose(g, -2.0, atol=0.05)
    # Stability mask is independent of boundary mask
    assert mask.shape == (28,)


def test_feasible_fd_at_lower_bound_uses_forward():
    _install_quad_return(monkey_theta_star=np.zeros(28))
    env = _StubEnv()
    theta = -np.ones(28)
    g, mask, steps, meta = sim_grad_masked_detailed(
        env, {}, theta, feasible=True, horizon=1, scales=(0.5, 1.0, 2.0), eps=0.05
    )
    assert all(m == "forward" for m in meta["fd_mode"])
    # ∇J = -2(θ-0) = -2(-1) = +2
    assert np.allclose(g, 2.0, atol=0.05)


def test_legacy_central_may_leave_box_but_feasible_stays_inside():
    """Document that feasible=False can query |θ|>1; feasible clips probes."""
    calls = []

    import training.gradients as gmod

    def tracking_return(env, theta, start, horizon=8):
        th = np.asarray(theta, dtype=np.float64)
        calls.append(th.copy())
        return float(-np.sum(th ** 2)), {}

    gmod.physical_return = tracking_return
    theta = np.ones(28)
    sim_grad_masked(_StubEnv(), {}, theta, feasible=False, horizon=1, scales=(1.0,), eps=0.05)
    outside = any(np.any(np.abs(c) > 1.0 + 1e-9) for c in calls)
    assert outside, "legacy FD should probe outside the box near the bound"

    calls.clear()
    sim_grad_masked(_StubEnv(), {}, theta, feasible=True, horizon=1, scales=(1.0,), eps=0.05)
    assert all(np.all(np.abs(c) <= 1.0 + 1e-9) for c in calls)


def test_rho_proj_and_outward_pressure():
    raw = np.array([0.5, 0.0, -0.2])
    proj = np.array([0.1, 0.0, -0.2])  # first coord clipped
    r = rho_proj(raw, proj)
    assert r > 0
    expected = float(np.linalg.norm(raw - proj)) / (float(np.linalg.norm(raw)) + 1e-12)
    assert abs(r - expected) < 1e-12

    th = np.array([1.0, -1.0, 0.0])
    # ascent grads: at +1 want more (+), at -1 want more (-), interior 0
    g = np.array([0.3, -0.4, 0.1])
    assert outward_pressure_frac(th, g) == 1.0
    g2 = np.array([-0.3, 0.4, 0.1])  # both push inward
    assert outward_pressure_frac(th, g2) == 0.0


def test_policy_optimize_records_rho_proj():
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
    assert "rho_proj" in out["curve"][0]
    assert "outward_pressure_frac" in out["curve"][0]
    assert "mean_rho_proj" in out
    assert "first_hit_update" in out
    assert len(out["first_hit_update"]) == 28


def test_dir_label_budget_accounting_equal_n_labels():
    """Matched Path vs Random must use same label count → same FD step budget class."""
    from training.stage6c_common import count_fd_control_steps, match_label_budget

    # Fake labels with recorded steps
    random_labs = [dict(control_steps=28 * 2 * 8 * 3) for _ in range(12)]
    path_labs = [dict(control_steps=28 * 2 * 8 * 3) for _ in range(12)]
    br = count_fd_control_steps(random_labs)
    bp = count_fd_control_steps(path_labs)
    assert br == bp
    trimmed = match_label_budget(path_labs + path_labs[:3], target_n=len(random_labs))
    assert len(trimmed) == 12
