#!/usr/bin/env python3
"""Minimal Stage-3 seed-0 smoke on Controller B (after Stage-1 gate).

Collects a small on-policy dataset under PartialUnknownArmEnv, trains:
  - Mc-like MLP on full c (oracle memory; unfair upper-ish ref on B)
  - Mu-oracle using privileged true u
  - EAGM-Base (c_obs + history residual)

Reports dynamics NMSE and command MAE only — not a claim of method success.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

EAGM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EAGM_ROOT))
sys.path.insert(0, str(EAGM_ROOT / "vendor" / "legacy_sim"))

from controllers.partial_unknown import ControllerBConfig, PartialUnknownArmEnv
from models.eagm import EAGMWorld


def collect(env, episodes=6, horizon=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for ep in range(episodes):
        delay = int(rng.integers(0, 3))
        x = env.reset(seed=seed * 100 + ep, delay=delay, jitter=0.01)
        hist_x, hist_a = [], []
        for t in range(horizon):
            a = rng.normal(0, 0.01, size=14).clip(-0.03, 0.03)
            c_full = env.control_state().copy()
            c_obs = env.observed_control_state().copy()
            priv = env.privileged_labels()
            x_next = env.step(a)
            rows.append(
                dict(
                    x=x.copy(),
                    y=x_next.copy(),
                    a=a.copy(),
                    c=c_full,
                    c_obs=c_obs,
                    u_true=priv["u_true"].copy(),
                    delay=delay,
                )
            )
            hist_x.append(x.copy())
            hist_a.append(a.copy())
            x = x_next
    return rows


def fit_stats(rows):
    x = np.stack([r["x"] for r in rows])
    y = np.stack([r["y"] for r in rows])
    u = np.stack([r["u_true"] for r in rows])
    delta = y - x
    return dict(
        xmean=x.mean(0),
        xstd=np.maximum(x.std(0), 0.01),
        umean=u.mean(0),
        ustd=np.maximum(u.std(0), 0.001),
        ymean=delta.mean(0),
        ystd=np.maximum(delta.std(0), 0.001),
        lower=np.min(np.stack([r["c"][:14] for r in rows]), 0) - 0.1,
        upper=np.max(np.stack([r["c"][:14] for r in rows]), 0) + 0.1,
        alpha=np.asarray(0.5),
    )


class FullCWorld(nn.Module):
    def __init__(self, stats, in_c=57, width=160):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 + 14 + in_c, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 28),
        )
        for k, v in stats.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))

    def forward(self, x, c, a):
        xx = (x - self.xmean) / self.xstd
        return x + self.net(torch.cat([xx, a, c], -1)) * self.ystd + self.ymean


class MuOracle(nn.Module):
    def __init__(self, stats, width=160):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(42, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 28),
        )
        for k, v in stats.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))

    def forward(self, x, u):
        xx = (x - self.xmean) / self.xstd
        uu = (u - self.umean) / self.ustd
        return x + self.net(torch.cat([xx, uu], -1)) * self.ystd + self.ymean


def train_loop(model, batch_fn, updates=400, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for it in range(updates):
        loss = batch_fn(model)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        opt.step()
        if (it + 1) % 50 == 0:
            losses.append(float(loss.detach()))
    return losses


def main():
    out = EAGM_ROOT / "runs" / "stage3_b_min_seed0"
    out.mkdir(parents=True, exist_ok=False)
    cfg = ControllerBConfig(gain=1.05, extra_lag=1, rate_limit=0.045, smooth_beta=0.25)
    env = PartialUnknownArmEnv(cfg)
    t0 = time.time()
    rows = collect(env, episodes=8, horizon=150, seed=0)
    n = len(rows)
    idx = np.arange(n)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    n_train = int(0.7 * n)
    train_i, test_i = idx[:n_train], idx[n_train:]
    stats = fit_stats([rows[i] for i in train_i])

    def tensors(subset):
        return {
            k: torch.as_tensor(np.stack([rows[i][k] for i in subset]), dtype=torch.float32)
            for k in ("x", "y", "a", "c", "c_obs", "u_true")
        }

    tr, te = tensors(train_i), tensors(test_i)

    mc = FullCWorld(stats)
    mu = MuOracle(stats)
    eagm = EAGMWorld(stats, history=False)  # residual off history for smoke; uses U0 only + learnable delta via zero mem path

    # For smoke without history tensors: use CommandBottleneck with mem zeros by disabling history
    # Switch to a tiny residual MLP on (c_obs,a)
    class EAGMBase(nn.Module):
        def __init__(self, stats, width=160, lam_u: float = 1.0):
            super().__init__()
            self.lam_u = lam_u
            self.delta = nn.Sequential(
                nn.Linear(29 + 14, 64), nn.SiLU(), nn.Linear(64, 14)
            )
            # Start as U0 (Δ≈0) so residual cannot immediately hurt cmd MAE.
            nn.init.zeros_(self.delta[-1].weight)
            nn.init.zeros_(self.delta[-1].bias)
            self.net = nn.Sequential(
                nn.Linear(42, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 28)
            )
            for k, v in stats.items():
                self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))

        def u0(self, c_obs, a):
            target = torch.maximum(torch.minimum(c_obs[..., :14] + a, self.upper), self.lower)
            return 0.5 * target + 0.5 * c_obs[..., 14:28]

        def forward(self, x, c_obs, a):
            u0 = self.u0(c_obs, a)
            u_hat = u0 + self.delta(torch.cat([c_obs, a], -1))
            xx = (x - self.xmean) / self.xstd
            uu = (u_hat - self.umean) / self.ustd
            y = x + self.net(torch.cat([xx, uu], -1)) * self.ystd + self.ymean
            return y, u_hat

    eagm = EAGMBase(stats, lam_u=1.0)

    def nmse(pred, y, ystd):
        return (((pred - y) / ystd) ** 2).mean().item()

    train_loop(
        mc,
        lambda m: (((m(tr["x"], tr["c"], tr["a"]) - tr["y"]) / m.ystd) ** 2).mean(),
        updates=800,
    )
    train_loop(
        mu,
        lambda m: (((m(tr["x"], tr["u_true"]) - tr["y"]) / m.ystd) ** 2).mean(),
        updates=800,
    )

    def eagm_loss(m):
        y_hat, u_hat = m(tr["x"], tr["c_obs"], tr["a"])
        dyn = (((y_hat - tr["y"]) / m.ystd) ** 2).mean()
        cmd = (((u_hat - tr["u_true"]) / m.ustd) ** 2).mean()
        return dyn + m.lam_u * cmd

    train_loop(eagm, eagm_loss, updates=800)

    with torch.no_grad():
        mc_nmse = nmse(mc(te["x"], te["c"], te["a"]), te["y"], mc.ystd)
        mu_nmse = nmse(mu(te["x"], te["u_true"]), te["y"], mu.ystd)
        y_e, u_e = eagm(te["x"], te["c_obs"], te["a"])
        eagm_nmse = nmse(y_e, te["y"], eagm.ystd)
        cmd_mae = (u_e - te["u_true"]).abs().mean().item()
        u0 = eagm.u0(te["c_obs"], te["a"])
        u0_mae = (u0 - te["u_true"]).abs().mean().item()

    payload = dict(
        controller="B",
        config=cfg.__dict__,
        n_train=len(train_i),
        n_test=len(test_i),
        updates=800,
        mc_nmse=mc_nmse,
        mu_oracle_nmse=mu_nmse,
        eagm_base_nmse=eagm_nmse,
        eagm_cmd_mae=cmd_mae,
        u0_cmd_mae=u0_mae,
        wall_seconds=time.time() - t0,
        note="Smoke only: dynamics/cmd metrics under Controller B; not policy gains.",
    )
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
