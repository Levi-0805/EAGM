"""Reach task: bimanual end-effector goal reaching on OpenArm + Controller B.

Design constraints (Stage 7B):
- Same robot asset / control interface as Stage 5–6 tracking.
- EAGM architecture unchanged (predicts x∈R^28; actor still joint-ref based).
- Task change = reward + goal/ref generation, not new world-model modules.
- Imagined and physical returns use the **same** Reach reward definition.
  EE pose for torch rollouts uses a frozen FK surrogate fitted to MuJoCo;
  physical diagnostics also log true MuJoCo site positions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import mujoco
import numpy as np
import torch
from torch import nn

from training.reward import DISCOUNT, W_DA, W_VEL

TASK_ID = "reach"
SITE_LEFT = "left_ee_control_point"
SITE_RIGHT = "right_ee_control_point"

# Reach reward weights (EE meters)
W_EE = 1.0
W_EE_ORI = 0.0  # position-only for Stage 7B


@dataclass(frozen=True)
class ReachConfig:
    """Goal / init randomization for Reach episodes."""

    goal_mode: str = "random"  # fixed | random
    fixed_goal_left: tuple[float, float, float] = (0.25, 0.25, -0.25)
    fixed_goal_right: tuple[float, float, float] = (0.25, -0.25, -0.25)
    # Sample EE goals in axis-aligned boxes around nominal home EE
    goal_noise: float = 0.06
    init_joint_jitter: float = 0.02
    # Joint-space refs: interpolate from start q toward IK-ish target
    ref_blend_horizon: int = 80
    # Trajectory shape for joint refs (still drives the actor)
    traj_shape: str = "hold"  # hold | sine_perturb
    sine_amp: float = 0.03
    controller_b_randomize: bool = True


def site_id(model: mujoco.MjModel, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise KeyError(name)
    return int(sid)


def ee_positions(env) -> tuple[np.ndarray, np.ndarray]:
    """True MuJoCo EE control-point positions (left, right), shape (3,)."""
    lid = site_id(env.model, SITE_LEFT)
    rid = site_id(env.model, SITE_RIGHT)
    return env.data.site_xpos[lid].copy(), env.data.site_xpos[rid].copy()


def ee_from_q_mujoco(env, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """FK-only: set qpos, mj_fwdPosition, read EE sites (does not step physics)."""
    q = np.asarray(q, dtype=np.float64).reshape(14)
    # Preserve full state then restore — callers often use this on a live env.
    integ = np.zeros(env._state_size, dtype=np.float64)
    mujoco.mj_getState(env.model, env.data, integ, env._spec)
    try:
        env.data.qpos[:] = q
        mujoco.mj_fwdPosition(env.model, env.data)
        return ee_positions(env)
    finally:
        mujoco.mj_setState(env.model, env.data, integ, env._spec)
        mujoco.mj_forward(env.model, env.data)


def sample_ee_goals(env, cfg: ReachConfig, rng: np.random.Generator) -> np.ndarray:
    """Return goals as (6,) = [left_xyz, right_xyz]."""
    if cfg.goal_mode == "fixed":
        return np.concatenate(
            [np.asarray(cfg.fixed_goal_left), np.asarray(cfg.fixed_goal_right)]
        ).astype(np.float64)
    left0, right0 = ee_positions(env)
    left = left0 + rng.uniform(-cfg.goal_noise, cfg.goal_noise, size=3)
    right = right0 + rng.uniform(-cfg.goal_noise, cfg.goal_noise, size=3)
    return np.concatenate([left, right]).astype(np.float64)


def _dls_ik_once(env, q0: np.ndarray, goals: np.ndarray, iters: int = 25) -> np.ndarray:
    """Few DLS IK iterations → joint pose near EE goals (not real-time per frame)."""
    q = np.asarray(q0, dtype=np.float64).copy()
    goals = np.asarray(goals, dtype=np.float64).reshape(6)
    eps = 1e-4
    damp = 1e-3
    for _ in range(iters):
        left, right = ee_from_q_mujoco(env, q)
        err = np.concatenate([goals[:3] - left, goals[3:] - right])
        if float(np.linalg.norm(err)) < 1e-3:
            break
        J = np.zeros((6, 14), dtype=np.float64)
        for i in range(14):
            dq = np.zeros(14)
            dq[i] = eps
            lp, rp = ee_from_q_mujoco(env, q + dq)
            lm, rm = ee_from_q_mujoco(env, q - dq)
            J[:, i] = np.concatenate([lp - lm, rp - rm]) / (2 * eps)
        dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), err)
        q = np.clip(q + 0.4 * dq, env.lower, env.upper)
    return q


def joint_refs_toward_goal(
    q0: np.ndarray,
    goals: np.ndarray,
    env,
    horizon: int,
    cfg: ReachConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build (horizon+16, 14) joint refs for the existing actor.

    Solve IK once toward EE goals, then linearly blend q0→q* and hold
    (optional sine perturbation for traj_shape ablation).
    """
    q0 = np.asarray(q0, dtype=np.float64).copy()
    q_star = _dls_ik_once(env, q0, goals, iters=25)
    refs = np.zeros((horizon + 16, 14), dtype=np.float64)
    n_blend = max(1, min(cfg.ref_blend_horizon, horizon))
    for t in range(horizon + 16):
        alpha = min(1.0, t / float(n_blend))
        refs[t] = (1.0 - alpha) * q0 + alpha * q_star
        if cfg.traj_shape == "sine_perturb":
            refs[t] = refs[t] + cfg.sine_amp * np.sin(
                2 * np.pi * (0.1 + 0.02 * (np.arange(14) % 5)) * t * float(env.control_dt)
            ) * rng.choice([-1.0, 1.0], size=14)
    return refs


def reach_reward_numpy(
    ee_left: np.ndarray,
    ee_right: np.ndarray,
    goals: np.ndarray,
    x: np.ndarray,
    a: np.ndarray,
    prev: np.ndarray,
) -> float:
    goals = np.asarray(goals, dtype=np.float64).reshape(6)
    x = np.asarray(x, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    prev = np.asarray(prev, dtype=np.float64)
    ee_err = np.concatenate(
        [np.asarray(ee_left) - goals[:3], np.asarray(ee_right) - goals[3:]]
    )
    return float(
        -W_EE * np.mean(ee_err ** 2)
        - W_VEL * np.mean(x[14:] ** 2)
        - W_DA * np.mean((a - prev) ** 2)
    )


def reach_reward_torch(ee6: torch.Tensor, goals6: torch.Tensor, x, a, prev):
    """ee6, goals6: [..., 6]."""
    return (
        -W_EE * ((ee6 - goals6) ** 2).mean(-1)
        - W_VEL * (x[..., 14:] ** 2).mean(-1)
        - W_DA * ((a - prev) ** 2).mean(-1)
    )


class FKSurrogate(nn.Module):
    """Frozen MLP q(14) → ee(6). Fitted to MuJoCo FK; used for imagined Reach reward."""

    def __init__(self, width: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(14, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 6),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.net(q)


def fit_fk_surrogate(env, n_samples: int = 4000, seed: int = 0, steps: int = 800) -> FKSurrogate:
    rng = np.random.default_rng(seed)
    qs, ees = [], []
    lo, hi = np.asarray(env.lower), np.asarray(env.upper)
    for _ in range(n_samples):
        q = rng.uniform(lo, hi)
        left, right = ee_from_q_mujoco(env, q)
        qs.append(q)
        ees.append(np.concatenate([left, right]))
    X = torch.as_tensor(np.stack(qs), dtype=torch.float32)
    Y = torch.as_tensor(np.stack(ees), dtype=torch.float32)
    model = FKSurrogate()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        pred = model(X)
        loss = ((pred - Y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # hold residual for diagnostics
    with torch.no_grad():
        rmse = float(torch.sqrt(((model(X) - Y) ** 2).mean()))
    model.fit_rmse = rmse  # type: ignore[attr-defined]
    return model


def collect_reach_episodes(
    env,
    cfg: ReachConfig,
    n_episodes: int = 24,
    horizon: int = 200,
    seed: int = 0,
    hist_len: int = 8,
) -> list[dict[str, Any]]:
    """Collect dynamics transitions under Reach goals / joint refs."""
    rng = np.random.default_rng(seed)
    episodes = []
    for ep in range(n_episodes):
        delay = int(ep % 3)
        x = env.reset(seed=seed * 1000 + ep, delay=delay, jitter=cfg.init_joint_jitter)
        goals = sample_ee_goals(env, cfg, rng)
        refs = joint_refs_toward_goal(x[:14], goals, env, horizon, cfg, rng)
        prev_a = np.zeros(14, dtype=np.float64)
        c0 = env.observed_control_state().copy()
        x_buf = [x.copy()] * hist_len
        a_buf = [np.zeros(14, dtype=np.float64)] * hist_len
        c_obs_buf = [c0.copy()] * hist_len
        h_stack = np.zeros((8, 42), dtype=np.float64)
        h_stack[:, :28] = x
        rows = []
        for t_i in range(horizon):
            c_full = env.control_state().copy()
            c_obs = env.observed_control_state().copy()
            snap = env.snapshot()
            left, right = ee_positions(env)
            a = np.clip(
                0.55 * (refs[t_i] - c_full[:14]) + rng.normal(0, 0.003, 14),
                -0.03,
                0.03,
            )
            start = dict(
                task_id=TASK_ID,
                x=x.copy(),
                c=c_full,
                c_obs=c_obs,
                h=h_stack.copy(),
                prev=prev_a.copy(),
                refs=refs[t_i : t_i + 16].copy(),
                ee_goals=goals.copy(),
                ee_left=left.copy(),
                ee_right=right.copy(),
                snapshot=snap,
                delay=delay,
                episode_id=ep,
                frame=t_i,
                config_id=f"reach_{cfg.goal_mode}_{cfg.traj_shape}",
                x_hist=np.stack(x_buf[-hist_len:], 0).copy(),
                a_hist=np.stack(a_buf[-hist_len:], 0).copy(),
                c_obs_hist=np.stack(c_obs_buf[-hist_len:], 0).copy(),
                reach_cfg=asdict(cfg),
            )
            x_next = env.step(a)
            u_exec = env.last_true_command.copy()
            left_n, right_n = ee_positions(env)
            y = x_next.copy()
            rows.append(
                dict(
                    start=start,
                    a=a.copy(),
                    u_exec=u_exec,
                    y=y,
                    x=x.copy(),
                    c=c_full,
                    c_obs=c_obs,
                    delay=delay,
                    episode_id=ep,
                    frame=t_i,
                    config_id=start["config_id"],
                    x_hist=start["x_hist"],
                    a_hist=start["a_hist"],
                    c_obs_hist=start["c_obs_hist"],
                    ee_goals=goals.copy(),
                    ee_next=np.concatenate([left_n, right_n]),
                    reward=reach_reward_numpy(left_n, right_n, goals, x_next, a, prev_a),
                )
            )
            # history update (match dataset.collect_trajectories semantics)
            x_buf.append(x_next.copy())
            a_buf.append(a.copy())
            c_obs_buf.append(env.observed_control_state().copy())
            h_stack = np.roll(h_stack, -1, axis=0)
            h_stack[-1] = np.concatenate([x, a])
            x = x_next
            prev_a = a
        episodes.append(dict(episode_id=ep, rows=rows, ee_goals=goals.copy(), config=asdict(cfg)))
    return episodes


def flatten_reach(episodes: list[dict]) -> list[dict]:
    return [r for ep in episodes for r in ep["rows"]]


def reach_reward_surrogate(fk, x, a, prev, goals):
    """Differentiable imagined objective; FK parameters must be frozen."""
    if torch.is_tensor(x):
        return reach_reward_torch(fk(x[..., :14]), torch.as_tensor(goals, dtype=x.dtype, device=x.device), x, a, prev)
    ee = fk(torch.as_tensor(x[:14], dtype=torch.float32)).detach().cpu().numpy()
    return reach_reward_numpy(ee[:3], ee[3:], goals, x, a, prev)


def reach_reward_true_mujoco(env, x, a, prev, goals):
    """Physical objective evaluated at current simulated site positions."""
    # mj_step's derived site cache can precede the final integration update.
    # Refresh FK in separate data, never mutate live warm-start/integration state.
    data = getattr(env, "_true_ee_fk_data", None)
    if data is None:
        data = mujoco.MjData(env.model)
        env._true_ee_fk_data = data
    data.qpos[:] = env.data.qpos
    data.mocap_pos[:] = env.data.mocap_pos
    data.mocap_quat[:] = env.data.mocap_quat
    mujoco.mj_fwdPosition(env.model, data)
    left = data.site_xpos[site_id(env.model, SITE_LEFT)].copy()
    right = data.site_xpos[site_id(env.model, SITE_RIGHT)].copy()
    return reach_reward_numpy(left, right, goals, x, a, prev)
