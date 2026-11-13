"""Unit tests for Stage 7B Reach environment."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "legacy_sim"))

from controllers.partial_unknown import ControllerBConfig, PartialUnknownArmEnv
from tasks.reach import (
    ReachConfig,
    collect_reach_episodes,
    ee_from_q_mujoco,
    ee_positions,
    fit_fk_surrogate,
    reach_reward_numpy,
    sample_ee_goals,
)
from training.rollout import (
    imagined_eagm,
    physical_return_detailed,
    set_reach_fk_surrogate,
)
from models.eagm import EAGMBase
from training.dataset import fit_stats


def _env():
    return PartialUnknownArmEnv(
        ControllerBConfig(
            episode_randomize=False, gain=1.0, extra_lag=1, rate_limit=0.05, smooth_beta=0.2
        )
    )


def test_ee_sites_exist_and_move():
    env = _env()
    env.reset(seed=0, delay=1)
    l0, r0 = ee_positions(env)
    assert l0.shape == (3,) and r0.shape == (3,)
    env.step(np.ones(14) * 0.02)
    l1, r1 = ee_positions(env)
    assert np.linalg.norm(l1 - l0) + np.linalg.norm(r1 - r0) > 1e-6


def test_fk_only_restores_state():
    env = _env()
    x0 = env.reset(seed=1, delay=1)
    l0, r0 = ee_positions(env)
    q = x0[:14] + 0.05
    ee_from_q_mujoco(env, q)
    # after FK helper, live state restored
    assert np.allclose(env.observe()[:14], x0[:14], atol=1e-8)
    l1, r1 = ee_positions(env)
    assert np.allclose(l0, l1) and np.allclose(r0, r1)


def test_fk_surrogate_rmse_acceptable():
    env = _env()
    env.reset(seed=0, delay=1)
    fk = fit_fk_surrogate(env, n_samples=800, seed=0, steps=400)
    assert fk.fit_rmse < 0.05  # meters; loose gate for unit test sample size


def test_reach_collect_and_matched_reward():
    env = _env()
    cfg = ReachConfig(goal_mode="fixed", traj_shape="hold", init_joint_jitter=0.01)
    eps = collect_reach_episodes(env, cfg, n_episodes=2, horizon=32, seed=0)
    assert len(eps) == 2
    row = eps[0]["rows"][0]
    assert row["start"]["task_id"] == "reach"
    assert row["start"]["ee_goals"].shape == (6,)
    assert row["start"]["refs"].shape[1] == 14

    fk = fit_fk_surrogate(env, n_samples=600, seed=1, steps=300)
    set_reach_fk_surrogate(fk)
    # tiny untrained model for imagination smoke
    stats = fit_stats(
        [
            dict(
                x=r["x"], y=r["y"], a=r["a"], c=r["c"], c_obs=r["c_obs"],
                u_exec=r["u_exec"],
                x_hist=r["start"]["x_hist"], a_hist=r["start"]["a_hist"],
                c_obs_hist=r["start"]["c_obs_hist"],
            )
            for r in eps[0]["rows"]
        ],
        env_lower=env.lower,
        env_upper=env.upper,
    )
    model = EAGMBase(stats)
    start = row["start"]
    J = imagined_eagm(model, torch.zeros(28, requires_grad=True), start, horizon=4)
    assert torch.isfinite(J)
    g = torch.autograd.grad(J, torch.zeros(28, requires_grad=True), allow_unused=True)
    # re-bind theta properly
    th = torch.zeros(28, requires_grad=True)
    J2 = imagined_eagm(model, th, start, horizon=4)
    (J2).backward()
    assert th.grad is not None and torch.isfinite(th.grad).all()

    det = physical_return_detailed(env, np.zeros(28), start, horizon=4)
    assert np.isfinite(det["return_"])
    assert "true_mujoco_ee_return" in det


def test_reach_reward_decreases_with_ee_error():
    x = np.zeros(28)
    a = np.zeros(14)
    prev = np.zeros(14)
    goals = np.zeros(6)
    r_close = reach_reward_numpy(np.zeros(3), np.zeros(3), goals, x, a, prev)
    r_far = reach_reward_numpy(np.ones(3), np.ones(3), goals, x, a, prev)
    assert r_close > r_far


def test_sample_goals_modes():
    env = _env()
    env.reset(seed=0, delay=1)
    rng = np.random.default_rng(0)
    g_fix = sample_ee_goals(env, ReachConfig(goal_mode="fixed"), rng)
    g_rand = sample_ee_goals(env, ReachConfig(goal_mode="random", goal_noise=0.05), rng)
    assert g_fix.shape == (6,) and g_rand.shape == (6,)
    assert not np.allclose(g_fix, g_rand)
