"""Policy optimization with θ traces and physical evaluation."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from training.rollout import imagined_eagm, imagined_mc, physical_return

DTYPE = torch.float32


def policy_optimize(
    model: nn.Module,
    starts: list[dict],
    *,
    use_history: bool,
    is_mc: bool = False,
    updates: int = 50,
    lr: float = 0.03,
    h_imag: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    theta = torch.zeros(28, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    curve = []
    theta0 = theta.detach().numpy().copy()
    for u in range(updates):
        if is_mc:
            Js = [imagined_mc(model, theta, s, horizon=h_imag) for s in starts]
        else:
            Js = [
                imagined_eagm(model, theta, s, horizon=h_imag, use_history=use_history)
                for s in starts
            ]
        loss = -torch.stack(Js).mean()
        opt.zero_grad()
        loss.backward()
        gnorm = float(theta.grad.detach().norm()) if theta.grad is not None else 0.0
        nn.utils.clip_grad_norm_([theta], 1.0)
        opt.step()
        with torch.no_grad():
            sat = float(((theta.abs() >= 1.0 - 1e-8).sum()) / 28)
            theta.clamp_(-1, 1)
        curve.append(
            dict(
                update=u + 1,
                imagined_return=float((-loss).detach()),
                grad_norm=gnorm,
                theta_norm=float(theta.detach().norm()),
                sat_frac=sat,
                theta=theta.detach().numpy().tolist(),
            )
        )
    theta_f = theta.detach().numpy().copy()
    return dict(theta0=theta0, theta=theta_f, curve=curve)


def eval_physical(env, theta, starts, horizon=16) -> dict[str, Any]:
    rets = []
    for i, s in enumerate(starts):
        r, sat = physical_return(env, theta, s, horizon=horizon)
        rets.append(
            dict(
                start_index=i,
                episode_id=s.get("episode_id"),
                frame=s.get("frame"),
                delay=s.get("delay"),
                config_id=s.get("config_id"),
                return_=r,
                sat_frac=sat,
            )
        )
    arr = np.asarray([x["return_"] for x in rets], dtype=np.float64)
    return dict(
        mean=float(arr.mean()),
        std=float(arr.std()),
        per_start=rets,
        theta_norm=float(np.linalg.norm(theta)),
        sat_frac=float(np.mean(np.abs(theta) >= 1.0 - 1e-8)),
    )
