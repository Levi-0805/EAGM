"""Trajectory-level collection and splits for Controller B Stage 3.1."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


HIST_LEN_DEFAULT = 8


@dataclass
class SplitSpec:
    train_eps: list[int]
    val_eps: list[int]
    test_eps: list[int]


def make_split(n_episodes: int, seed: int = 0, n_train=12, n_val=6, n_test=6) -> SplitSpec:
    if n_train + n_val + n_test > n_episodes:
        raise ValueError("not enough episodes for requested split")
    rng = np.random.default_rng(seed + 9173)
    ids = rng.permutation(n_episodes).tolist()
    return SplitSpec(
        train_eps=sorted(ids[:n_train]),
        val_eps=sorted(ids[n_train : n_train + n_val]),
        test_eps=sorted(ids[n_train + n_val : n_train + n_val + n_test]),
    )


def collect_trajectories(
    env,
    n_episodes: int = 24,
    horizon: int = 200,
    seed: int = 0,
    hist_len: int = HIST_LEN_DEFAULT,
    config_id: str = "B_fixed",
) -> list[dict[str, Any]]:
    """Collect (x,c,c_obs,a,u_exec,y) with correct command timing.

    Semantics per transition t:
      snapshot/observe BEFORE step; u_exec = last_true_command AFTER step(a_t).
    """
    rng = np.random.default_rng(seed)
    episodes = []
    for ep in range(n_episodes):
        delay = int(ep % 3)  # balanced delays across episodes
        x = env.reset(seed=seed * 1000 + ep, delay=delay, jitter=0.01)
        center = x[:14].copy()
        tt = np.arange(horizon + 16) * float(env.control_dt)
        refs = center[None, :] + 0.08 * np.sin(
            2 * np.pi * (0.1 + 0.02 * (np.arange(14) % 5))[None, :] * tt[:, None]
        )
        prev_a = np.zeros(14, dtype=np.float64)
        # Rolling history buffers (pre-step state / previous actions / c_obs)
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
            # Train-only privileged labels for Stage 6B.2 probes (never model inputs)
            priv = None
            if hasattr(env, "privileged_exec_state"):
                priv = env.privileged_exec_state()
            ctrl_before = env.data.ctrl[env.actuator_ids].copy()
            a = np.clip(
                0.55 * (refs[t_i] - c_full[:14]) + rng.normal(0, 0.003, 14),
                -0.03,
                0.03,
            )
            # History aligned as (x_{t-k+1:t}, a_{t-k:t-1}, c_obs_{t-k+1:t})
            x_hist = np.stack(x_buf[-hist_len:], axis=0)
            a_hist = np.stack(a_buf[-hist_len:], axis=0)
            c_obs_hist = np.stack(c_obs_buf[-hist_len:], axis=0)

            start = dict(
                x=x.copy(),
                c=c_full,
                c_obs=c_obs,
                h=h_stack.copy(),
                prev=prev_a.copy(),
                refs=refs[t_i : t_i + 16].copy(),
                snapshot=snap,
                delay=delay,
                episode_id=ep,
                frame=t_i,
                config_id=config_id,
                x_hist=x_hist.copy(),
                a_hist=a_hist.copy(),
                c_obs_hist=c_obs_hist.copy(),
            )
            x_next = env.step(a)
            u_exec = env.last_true_command.copy()
            ctrl_after = env.data.ctrl[env.actuator_ids].copy()
            if not np.allclose(u_exec, ctrl_after, atol=1e-12):
                raise RuntimeError("u_exec != data.ctrl after step")
            row = dict(
                x=x.copy(),
                y=x_next.copy(),
                a=a.copy(),
                c=c_full,
                c_obs=c_obs,
                u_exec=u_exec,
                delay=delay,
                episode_id=ep,
                frame=t_i,
                config_id=config_id,
                start=start,
                x_hist=x_hist,
                a_hist=a_hist,
                c_obs_hist=c_obs_hist,
                ctrl_before=ctrl_before,
            )
            if priv is not None:
                row["exec_state_vector"] = priv["vector"].copy()
                row["exec_state_scalars"] = np.asarray(
                    [priv["gain"], priv["extra_lag"], priv["rate_limit"], priv["smooth_beta"]],
                    dtype=np.float64,
                )
                start["exec_state_scalars"] = row["exec_state_scalars"].copy()
            rows.append(row)
            h_stack = np.concatenate([h_stack[1:], np.concatenate([x, a])[None]], 0)
            x_buf.append(x_next.copy())
            a_buf.append(a.copy())
            c_obs_buf.append(c_obs.copy())
            prev_a = a.copy()
            x = x_next
        episodes.append(
            dict(
                episode_id=ep,
                delay=delay,
                config_id=config_id,
                rows=rows,
                n_steps=len(rows),
            )
        )
    return episodes


def flatten_split(episodes: list[dict], ep_ids: list[int]) -> list[dict]:
    out = []
    idset = set(ep_ids)
    for ep in episodes:
        if ep["episode_id"] in idset:
            out.extend(ep["rows"])
    return out


def fit_stats(rows: list[dict], env_lower=None, env_upper=None) -> dict:
    """Normalization from train rows; projection limits from the real controller.

    Data min/max ±0.05 must NOT replace joint/actuator projection limits used in
    imagined recurrence. Pass env.lower / env.upper from ArmEnv/Controller B.
    """
    x = np.stack([r["x"] for r in rows])
    y = np.stack([r["y"] for r in rows])
    u = np.stack([r["u_exec"] for r in rows])
    delta = y - x
    data_lo = np.min(np.stack([r["c"][:14] for r in rows]), 0) - 0.05
    data_hi = np.max(np.stack([r["c"][:14] for r in rows]), 0) + 0.05
    if env_lower is None or env_upper is None:
        lo, hi = data_lo, data_hi
    else:
        lo = np.asarray(env_lower, dtype=np.float64).copy()
        hi = np.asarray(env_upper, dtype=np.float64).copy()
    return dict(
        xmean=x.mean(0),
        xstd=np.maximum(x.std(0), 0.01),
        umean=u.mean(0),
        ustd=np.maximum(u.std(0), 0.001),
        ymean=delta.mean(0),
        ystd=np.maximum(delta.std(0), 0.001),
        lower=lo,
        upper=hi,
        alpha=np.asarray(0.5),
        data_lower_margin=data_lo,
        data_upper_margin=data_hi,
    )


def pick_starts(
    rows: list[dict],
    *,
    delay: int | None = 2,
    max_per_episode: int = 2,
    max_total: int = 12,
) -> list[dict]:
    """Select policy/dir starts from distinct episodes (trajectory isolation)."""
    by_ep: dict[int, list] = {}
    for r in rows:
        if delay is not None and r["delay"] != delay:
            continue
        by_ep.setdefault(r["episode_id"], []).append(r)
    starts = []
    for ep in sorted(by_ep):
        cand = by_ep[ep]
        # Spread within episode: early and mid frames
        idxs = np.linspace(0, len(cand) - 1, num=min(max_per_episode, len(cand)), dtype=int)
        for i in idxs:
            starts.append(cand[int(i)]["start"])
            if len(starts) >= max_total:
                return starts
    return starts
