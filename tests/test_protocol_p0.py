"""P0 protocol tests: command timing, replay, reward parity, trajectory splits."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "legacy_sim"))

from controllers.partial_unknown import ControllerBConfig, PartialUnknownArmEnv
from training.dataset import collect_trajectories, make_split, flatten_split
from training.reward import reward_numpy, reward_torch


def test_u_exec_after_step_matches_ctrl():
    for lag in (0, 1, 2):
        for gain in (0.95, 1.05):
            cfg = ControllerBConfig(
                gain=gain, extra_lag=lag, rate_limit=0.04, smooth_beta=0.2
            )
            env = PartialUnknownArmEnv(cfg)
            env.reset(seed=0, delay=2, jitter=0.0)
            for i in range(5):
                a = np.zeros(14)
                a[i % 14] = 0.02
                x = env.observe().copy()
                _ = env.control_state().copy()
                y = env.step(a)
                u = env.last_true_command.copy()
                assert np.allclose(u, env.data.ctrl[env.actuator_ids], atol=1e-12)
                assert y.shape == (28,)
                assert x.shape == (28,)


def test_snapshot_replay_command_and_next_state():
    cfg = ControllerBConfig(gain=1.05, extra_lag=1, rate_limit=0.045, smooth_beta=0.25)
    env = PartialUnknownArmEnv(cfg)
    env.reset(seed=3, delay=1, jitter=0.01)
    for _ in range(4):
        env.step(np.clip(np.random.randn(14) * 0.01, -0.03, 0.03))
    snap = env.snapshot()
    a = np.clip(np.ones(14) * 0.01, -0.03, 0.03)
    y1 = env.step(a)
    u1 = env.last_true_command.copy()

    env2 = PartialUnknownArmEnv(cfg)
    env2.reset(seed=99, delay=0, jitter=0.0)
    env2.restore(snap)
    y2 = env2.step(a)
    u2 = env2.last_true_command.copy()
    assert np.allclose(u1, u2, atol=1e-10), (u1 - u2).max()
    assert np.allclose(y1, y2, atol=1e-8), np.max(np.abs(y1 - y2))


def test_collect_timing_u_changes_with_action():
    cfg = ControllerBConfig(gain=1.05, extra_lag=0, rate_limit=0.05, smooth_beta=0.0)
    env = PartialUnknownArmEnv(cfg)
    eps = collect_trajectories(env, n_episodes=1, horizon=20, seed=1, hist_len=4)
    rows = eps[0]["rows"]
    # After timing fix, first-step u_exec should respond to a (not stale pre-step cmd)
    assert "u_exec" in rows[0]
    # Same snapshot + action replay matches stored u_exec / y
    r = rows[5]
    env.restore(r["start"]["snapshot"])
    y = env.step(r["a"])
    assert np.allclose(env.last_true_command, r["u_exec"], atol=1e-10)
    assert np.allclose(y, r["y"], atol=1e-8)


def test_reward_torch_numpy_parity():
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.normal(size=28)
        a = rng.normal(size=14) * 0.01
        ref = rng.normal(size=14)
        prev = rng.normal(size=14) * 0.01
        nt = float(reward_torch(
            torch.as_tensor(x), torch.as_tensor(a), torch.as_tensor(ref), torch.as_tensor(prev)
        ).item())
        nn = reward_numpy(x, a, ref, prev)
        assert abs(nt - nn) < 1e-12


def test_trajectory_split_no_episode_overlap():
    split = make_split(24, seed=0, n_train=12, n_val=6, n_test=6)
    tr, va, te = set(split.train_eps), set(split.val_eps), set(split.test_eps)
    assert not (tr & va)
    assert not (tr & te)
    assert not (va & te)
    assert len(tr | va | te) == 24


def test_known_config_vector_is_leak_if_fed_to_model():
    """Document: known_config_vector exposes B params — deployable models must not use it."""
    env = PartialUnknownArmEnv(ControllerBConfig(gain=1.05, extra_lag=2))
    env.reset(seed=0)
    v = env.known_config_vector()
    assert v.shape == (4,)
    assert abs(v[0] - 1.05) < 1e-12
    assert int(v[1]) == 2


if __name__ == "__main__":
    test_u_exec_after_step_matches_ctrl()
    test_snapshot_replay_command_and_next_state()
    test_collect_timing_u_changes_with_action()
    test_reward_torch_numpy_parity()
    test_trajectory_split_no_episode_overlap()
    test_known_config_vector_is_leak_if_fed_to_model()
    print("all_p0_ok")
