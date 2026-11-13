"""Imagined rollouts with unified reward (matches physical eval)."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from training.reward import DISCOUNT, reward_numpy, reward_torch

DTYPE = torch.float32
ACTION_LIMIT = 0.03

# Optional Reach FK surrogate (torch); set via set_reach_fk_surrogate().
_REACH_FK = None


def set_reach_fk_surrogate(model: Optional[torch.nn.Module]) -> None:
    """Register frozen FK MLP for Reach imagined/physical matched rewards."""
    global _REACH_FK
    _REACH_FK = model


def get_reach_fk_surrogate():
    return _REACH_FK


def _step_reward(task_id, newx, a, ref, prev, *, ee_goals=None):
    """Dispatch joint-tracking vs Reach reward (matched imagined/physical)."""
    if task_id == "reach":
        from tasks.reach import reach_reward_numpy, reach_reward_torch

        if _REACH_FK is None:
            raise RuntimeError("Reach rollout requires set_reach_fk_surrogate(...)")
        if torch.is_tensor(newx):
            ee6 = _REACH_FK(newx[..., :14])
            g = torch.as_tensor(ee_goals, dtype=newx.dtype, device=newx.device)
            return reach_reward_torch(ee6, g, newx, a, prev)
        # Matched FD / policy metrics use the same surrogate as imagination
        ee6 = _REACH_FK(torch.as_tensor(newx[:14], dtype=DTYPE)).detach().cpu().numpy()
        return reach_reward_numpy(ee6[:3], ee6[3:], ee_goals, newx, a, prev)
    if torch.is_tensor(newx):
        return reward_torch(newx, a, ref, prev)
    return reward_numpy(newx, a, ref, prev)


def actor_raw_torch(theta, x, c_or_obs_target14, ref):
    """Unclipped actor output (for saturation diagnostics)."""
    return (
        0.55 * (ref - c_or_obs_target14)
        + 0.01 * theta[:14]
        + 0.08 * theta[14:] * (ref - x[..., :14])
    )


def actor(theta, x, c_or_obs_target14, ref):
    # c_or_obs_target14: first 14 dims are joint targets (c[:14] or c_obs[:14])
    return torch.clamp(actor_raw_torch(theta, x, c_or_obs_target14, ref), -ACTION_LIMIT, ACTION_LIMIT)


def actor_raw_numpy(theta, x, c_target14, ref):
    return (
        0.55 * (ref - c_target14)
        + 0.01 * theta[:14]
        + 0.08 * theta[14:] * (ref - x[:14])
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


def history_shift(obs_hist, act_hist, x_new, a_new, c_obs_hist=None, c_obs_new=None):
    # obs_hist/act_hist: [T, D] or [B,T,D]
    if obs_hist.dim() == 2:
        oh = torch.cat([obs_hist[1:], x_new.unsqueeze(0)], 0)
        ah = torch.cat([act_hist[1:], a_new.unsqueeze(0)], 0)
        if c_obs_hist is None:
            return oh, ah
        ch = torch.cat([c_obs_hist[1:], c_obs_new.unsqueeze(0)], 0)
        return oh, ah, ch
    oh = torch.cat([obs_hist[:, 1:], x_new.unsqueeze(1)], 1)
    ah = torch.cat([act_hist[:, 1:], a_new.unsqueeze(1)], 1)
    if c_obs_hist is None:
        return oh, ah
    ch = torch.cat([c_obs_hist[:, 1:], c_obs_new.unsqueeze(1)], 1)
    return oh, ah, ch


def imagined_eagm(model, theta, start, horizon=8, use_history: bool = False):
    x = torch.as_tensor(start["x"], dtype=DTYPE)
    c_obs = torch.as_tensor(start["c_obs"], dtype=DTYPE)
    prev = torch.as_tensor(start["prev"], dtype=DTYPE)
    refs = torch.as_tensor(start["refs"][:horizon], dtype=DTYPE)
    uses_c = bool(getattr(model, "uses_c_obs_hist", False))
    if use_history:
        obs_hist = torch.as_tensor(start["x_hist"], dtype=DTYPE)
        act_hist = torch.as_tensor(start["a_hist"], dtype=DTYPE)
        if uses_c:
            if "c_obs_hist" not in start:
                raise KeyError("HistC rollout requires start['c_obs_hist']")
            c_obs_hist = torch.as_tensor(start["c_obs_hist"], dtype=DTYPE)
    total = theta.sum() * 0
    for k in range(horizon):
        a = actor(theta, x, c_obs[:14], refs[k])
        if use_history:
            if uses_c:
                newx, _, _ = model(
                    x.unsqueeze(0), c_obs.unsqueeze(0), a.unsqueeze(0),
                    obs_hist.unsqueeze(0), act_hist.unsqueeze(0), c_obs_hist.unsqueeze(0),
                )
            else:
                newx, _, _ = model(
                    x.unsqueeze(0), c_obs.unsqueeze(0), a.unsqueeze(0),
                    obs_hist.unsqueeze(0), act_hist.unsqueeze(0),
                )
            newx = newx.squeeze(0)
            c_obs_next = c_obs_step(c_obs, a, model.lower, model.upper, float(model.alpha))
            if uses_c:
                obs_hist, act_hist, c_obs_hist = history_shift(
                    obs_hist, act_hist, newx, a, c_obs_hist, c_obs_next
                )
            else:
                obs_hist, act_hist = history_shift(obs_hist, act_hist, newx, a)
            c_obs = c_obs_next
        else:
            newx, _ = model(x.unsqueeze(0), c_obs.unsqueeze(0), a.unsqueeze(0))
            newx = newx.squeeze(0)
            c_obs = c_obs_step(c_obs, a, model.lower, model.upper, float(model.alpha))
        total = total + (DISCOUNT ** k) * _step_reward(
            start.get("task_id"), newx, a, refs[k], prev, ee_goals=start.get("ee_goals")
        )
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
        total = total + (DISCOUNT ** k) * _step_reward(
            start.get("task_id"), newx, a, refs[k], prev, ee_goals=start.get("ee_goals")
        )
        # history_next like experiment
        h = torch.cat([h[1:], torch.cat([x, a], -1).unsqueeze(0)], 0)
        x = newx
        c = c_full_step(c, a, model.lower, model.upper)
        prev = a
    return total


def physical_return(env, theta, start, horizon=16):
    """Same reward definition as imagined_*; uses NumPy on real env.

    Returns (total_return, action_clip_frac) for backward compatibility.
    Prefer physical_return_detailed for Stage-6 saturation diagnostics.
    """
    detail = physical_return_detailed(env, theta, start, horizon=horizon)
    return detail["return_"], detail["action_clip_frac"]


def physical_return_detailed(env, theta, start, horizon=16) -> dict[str, Any]:
    """Physical rollout with separate raw-action vs clipped-action saturation."""
    env.restore(start["snapshot"])
    theta = np.asarray(theta, dtype=np.float64).reshape(28)
    total = 0.0
    prev = np.asarray(start["prev"], dtype=np.float64).copy()
    n_dims = 14 * horizon
    raw_exceed = 0
    clipped = 0
    per_joint_clip = np.zeros(14, dtype=np.int64)
    for k in range(horizon):
        x = env.observe()
        c = env.control_state()
        ref = start["refs"][k]
        raw = actor_raw_numpy(theta, x, c[:14], ref)
        a = np.clip(raw, -ACTION_LIMIT, ACTION_LIMIT)
        raw_exceed += int(np.count_nonzero(np.abs(raw) >= ACTION_LIMIT))
        was_clip = np.abs(raw) > ACTION_LIMIT + 1e-12
        clipped += int(np.count_nonzero(was_clip))
        per_joint_clip += was_clip.astype(np.int64)
        env.step(a)
        newx = env.observe()
        total += (DISCOUNT ** k) * _step_reward(
            start.get("task_id"), newx, a, ref, prev, ee_goals=start.get("ee_goals")
        )
        prev = a
    out = dict(
        return_=float(total),
        action_raw_exceed_frac=raw_exceed / n_dims,
        action_clip_frac=clipped / n_dims,
        # alias kept for callers that used sat_frac as action clip
        sat_frac=clipped / n_dims,
        per_joint_clip_frac=(per_joint_clip / horizon).tolist(),
        theta_param_sat_frac=float(np.mean(np.abs(theta) >= 1.0 - 1e-8)),
        task_id=start.get("task_id"),
    )
    # Secondary true-MuJoCo EE return for Reach diagnostics (not used in Dir FD)
    if start.get("task_id") == "reach" and start.get("ee_goals") is not None:
        from tasks.reach import ee_positions, reach_reward_numpy

        env.restore(start["snapshot"])
        prev2 = np.asarray(start["prev"], dtype=np.float64).copy()
        true_total = 0.0
        for k in range(horizon):
            x = env.observe()
            c = env.control_state()
            ref = start["refs"][k]
            raw = actor_raw_numpy(theta, x, c[:14], ref)
            a = np.clip(raw, -ACTION_LIMIT, ACTION_LIMIT)
            env.step(a)
            newx = env.observe()
            left, right = ee_positions(env)
            true_total += (DISCOUNT ** k) * reach_reward_numpy(
                left, right, start["ee_goals"], newx, a, prev2
            )
            prev2 = a
        out["true_mujoco_ee_return"] = float(true_total)
    return out


def _physical_return_reach_objective(env, theta, start, horizon, objective):
    from tasks.reach import reach_reward_true_mujoco, reach_reward_surrogate
    if start.get("task_id") != "reach":
        raise ValueError("Explicit Reach objective requires a Reach start")
    env.restore(start["snapshot"])
    prev = np.asarray(start["prev"]).copy()
    total = 0.0
    for k in range(horizon):
        x, c = env.observe(), env.control_state()
        a = np.clip(actor_raw_numpy(theta, x, c[:14], start["refs"][k]), -ACTION_LIMIT, ACTION_LIMIT)
        env.step(a)
        x = env.observe()
        if objective == "true_ee":
            r = reach_reward_true_mujoco(env, x, a, prev, start["ee_goals"])
        else:
            r = reach_reward_surrogate(_REACH_FK, x, a, prev, start["ee_goals"])
        total += DISCOUNT ** k * r
        prev = a
    return float(total), {"control_steps": horizon, "objective": objective}


def physical_return_true_ee(env, theta, start, horizon=16):
    return _physical_return_reach_objective(env, theta, start, horizon, "true_ee")


def physical_return_surrogate(env, theta, start, horizon=16):
    return _physical_return_reach_objective(env, theta, start, horizon, "surrogate")
