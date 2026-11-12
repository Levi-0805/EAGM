"""Imagined rollouts with unified reward (matches physical eval)."""
from __future__ import annotations

import numpy as np
import torch

from training.reward import DISCOUNT, reward_numpy, reward_torch

DTYPE = torch.float32


def actor(theta, x, c_or_obs_target14, ref):
    # c_or_obs_target14: first 14 dims are joint targets (c[:14] or c_obs[:14])
    return torch.clamp(
        0.55 * (ref - c_or_obs_target14)
        + 0.01 * theta[:14]
        + 0.08 * theta[14:] * (ref - x[..., :14]),
        -0.03,
        0.03,
    )


def c_obs_step(c_obs, a, lower, upper, alpha=0.5):
    target = torch.clamp(c_obs[..., :14] + a, lower, upper)
    filt = alpha * target + (1.0 - alpha) * c_obs[..., 14:28]
    delay = c_obs[..., 28:29]
    return torch.cat([target, filt, delay], -1)


def c_full_step(c, a, lo, hi):
    # Same as experiment.c_step
    target = torch.maximum(torch.minimum(c[..., :14] + a, hi), lo)
    filt = 0.5 * target + 0.5 * c[..., 14:28]
    return torch.cat([target, filt, c[..., 42:56], filt, c[..., 56:57]], -1)


def history_shift(obs_hist, act_hist, x_new, a_new):
    # obs_hist/act_hist: [T, D] or [B,T,D]
    if obs_hist.dim() == 2:
        oh = torch.cat([obs_hist[1:], x_new.unsqueeze(0)], 0)
        ah = torch.cat([act_hist[1:], a_new.unsqueeze(0)], 0)
        return oh, ah
    oh = torch.cat([obs_hist[:, 1:], x_new.unsqueeze(1)], 1)
    ah = torch.cat([act_hist[:, 1:], a_new.unsqueeze(1)], 1)
    return oh, ah


def imagined_eagm(model, theta, start, horizon=8, use_history: bool = False):
    x = torch.as_tensor(start["x"], dtype=DTYPE)
    c_obs = torch.as_tensor(start["c_obs"], dtype=DTYPE)
    prev = torch.as_tensor(start["prev"], dtype=DTYPE)
    refs = torch.as_tensor(start["refs"][:horizon], dtype=DTYPE)
    if use_history:
        obs_hist = torch.as_tensor(start["x_hist"], dtype=DTYPE)
        act_hist = torch.as_tensor(start["a_hist"], dtype=DTYPE)
    total = theta.sum() * 0
    for k in range(horizon):
        a = actor(theta, x, c_obs[:14], refs[k])
        if use_history:
            newx, _, _ = model(
                x.unsqueeze(0), c_obs.unsqueeze(0), a.unsqueeze(0),
                obs_hist.unsqueeze(0), act_hist.unsqueeze(0),
            )
            newx = newx.squeeze(0)
            obs_hist, act_hist = history_shift(obs_hist, act_hist, newx, a)
        else:
            newx, _ = model(x.unsqueeze(0), c_obs.unsqueeze(0), a.unsqueeze(0))
            newx = newx.squeeze(0)
        total = total + (DISCOUNT ** k) * reward_torch(newx, a, refs[k], prev)
        c_obs = c_obs_step(c_obs, a, model.lower, model.upper, float(model.alpha))
        x = newx
        prev = a
    return total


def imagined_mc(model, theta, start, horizon=8):
    x = torch.as_tensor(start["x"], dtype=DTYPE)
    c = torch.as_tensor(start["c"], dtype=DTYPE)
    h = torch.as_tensor(start["h"], dtype=DTYPE)
    prev = torch.as_tensor(start["prev"], dtype=DTYPE)
    refs = torch.as_tensor(start["refs"][:horizon], dtype=DTYPE)
    total = theta.sum() * 0
    for k in range(horizon):
        a = actor(theta, x, c[:14], refs[k])
        newx = model(x, c, a, h)
        total = total + (DISCOUNT ** k) * reward_torch(newx, a, refs[k], prev)
        # history_next like experiment
        h = torch.cat([h[1:], torch.cat([x, a], -1).unsqueeze(0)], 0)
        x = newx
        c = c_full_step(c, a, model.lower, model.upper)
        prev = a
    return total


def physical_return(env, theta, start, horizon=16):
    """Same reward definition as imagined_*; uses NumPy on real env."""
    env.restore(start["snapshot"])
    total = 0.0
    prev = np.asarray(start["prev"], dtype=np.float64).copy()
    sat = 0
    for k in range(horizon):
        x = env.observe()
        c = env.control_state()
        ref = start["refs"][k]
        raw = 0.55 * (ref - c[:14]) + 0.01 * theta[:14] + 0.08 * theta[14:] * (ref - x[:14])
        sat += np.count_nonzero(np.abs(raw) >= 0.03)
        a = np.clip(raw, -0.03, 0.03)
        env.step(a)
        newx = env.observe()
        total += (DISCOUNT ** k) * reward_numpy(newx, a, ref, prev)
        prev = a
    return float(total), sat / (14 * horizon)
