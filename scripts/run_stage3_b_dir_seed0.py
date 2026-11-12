#!/usr/bin/env python3
"""Stage-3 seed-0: Controller B with masked directional loss + policy return.

Compares Mc-full-c, Mu-oracle (true u), EAGM-Base, EAGM-Dir under the same
PartialUnknownArmEnv. Direction labels use three-scale stability masks in the
training loss (legacy pilot omitted the mask).
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

import experiment as E  # noqa: E402
from controllers.partial_unknown import ControllerBConfig, PartialUnknownArmEnv
from training.dir_loss import huber_dir_loss

H = 8
DTYPE = torch.float32


def collect(env, episodes=12, horizon=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for ep in range(episodes):
        delay = int(rng.integers(0, 3))
        x = env.reset(seed=seed * 100 + ep, delay=delay, jitter=0.01)
        center = x[:14].copy()
        tt = np.arange(horizon + 16) * env.control_dt
        refs = center[None, :] + 0.08 * np.sin(
            2 * np.pi * (0.1 + 0.02 * (np.arange(14) % 5))[None, :] * tt[:, None]
        )
        prev_a = np.zeros(14, dtype=np.float64)
        h = np.zeros((8, 42), dtype=np.float64)
        h[:, :28] = x
        for t_i in range(horizon):
            c_full = env.control_state().copy()
            c_obs = env.observed_control_state().copy()
            u_true = env.privileged_labels()["u_true"].copy()
            snap = env.snapshot()
            start = dict(
                x=x.copy(),
                c=c_full,
                c_obs=c_obs,
                h=h.copy(),
                prev=prev_a.copy(),
                refs=refs[t_i : t_i + 16].copy(),
                snapshot=snap,
                delay=delay,
                trajectory=ep,
                frame=env.frame,
                u_true=u_true,
            )
            a = np.clip(
                0.55 * (refs[t_i] - c_full[:14]) + rng.normal(0, 0.003, 14),
                -0.03,
                0.03,
            )
            x_next = env.step(a)
            rows.append(
                dict(
                    x=x.copy(),
                    y=x_next.copy(),
                    a=a.copy(),
                    c=c_full,
                    c_obs=c_obs,
                    u_true=u_true.copy(),
                    start=start,
                    delay=delay,
                )
            )
            h = np.concatenate([h[1:], np.concatenate([x, a])[None]], 0)
            prev_a = a.copy()
            x = x_next
    return rows


def fit_stats(rows):
    x = np.stack([r["x"] for r in rows])
    y = np.stack([r["y"] for r in rows])
    u = np.stack([r["u_true"] for r in rows])
    delta = y - x
    lo = np.min(np.stack([r["c"][:14] for r in rows]), 0) - 0.05
    hi = np.max(np.stack([r["c"][:14] for r in rows]), 0) + 0.05
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
    )


def sim_grad_masked(env, start, eps=0.05, scales=(0.5, 1.0, 2.0)):
    out, stable = [], []
    for i in range(28):
        e = np.zeros(28)
        e[i] = 1.0
        vals = []
        for sc in scales:
            ep = eps * sc
            plus = E.physical(env, ep * e, start, H)[0]
            minus = E.physical(env, -ep * e, start, H)[0]
            vals.append((plus - minus) / (2 * ep))
        v = vals[1]
        out.append(v)
        spread = max(vals) - min(vals)
        stable.append(spread <= max(1e-9, 0.1 * abs(v)))
    return np.asarray(out, dtype=np.float64), np.asarray(stable, dtype=np.float64)


class McWorld(nn.Module):
    def __init__(self, stats, width=160):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 + 14 + 57, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 28),
        )
        for k, v in stats.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=DTYPE))

    def forward(self, x, c, a, h=None):
        xx = (x - self.xmean) / self.xstd
        return x + self.net(torch.cat([xx, a, c], -1)) * self.ystd + self.ymean


class MuOracle(nn.Module):
    def __init__(self, stats, width=160):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(42, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 28),
        )
        for k, v in stats.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=DTYPE))

    def forward(self, x, c, a, h=None):
        # c unused; caller passes u in a's sibling via train wrappers
        raise RuntimeError("use forward_u")

    def forward_u(self, x, u):
        xx = (x - self.xmean) / self.xstd
        uu = (u - self.umean) / self.ustd
        return x + self.net(torch.cat([xx, uu], -1)) * self.ystd + self.ymean


class EAGMBase(nn.Module):
    def __init__(self, stats, width=160):
        super().__init__()
        self.delta = nn.Sequential(
            nn.Linear(29 + 14, 64), nn.SiLU(), nn.Linear(64, 14)
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        self.net = nn.Sequential(
            nn.Linear(42, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 28),
        )
        for k, v in stats.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=DTYPE))

    def u0(self, c_obs, a):
        target = torch.maximum(torch.minimum(c_obs[..., :14] + a, self.upper), self.lower)
        return 0.5 * target + 0.5 * c_obs[..., 14:28]

    def command(self, c_obs, a):
        return self.u0(c_obs, a) + self.delta(torch.cat([c_obs, a], -1))

    def forward(self, x, c_obs, a):
        u_hat = self.command(c_obs, a)
        xx = (x - self.xmean) / self.xstd
        uu = (u_hat - self.umean) / self.ustd
        y = x + self.net(torch.cat([xx, uu], -1)) * self.ystd + self.ymean
        return y, u_hat


def imagined_eagm(model, theta, start, horizon=H):
    """28-param actor on c_obs / x like experiment.policy, rolled in EAGM."""
    x = torch.as_tensor(start["x"], dtype=DTYPE)
    c_obs = torch.as_tensor(start["c_obs"], dtype=DTYPE)
    prev = torch.as_tensor(start["prev"], dtype=DTYPE)
    refs = torch.as_tensor(start["refs"], dtype=DTYPE)
    total = 0.0
    for t in range(horizon):
        err = refs[t] - c_obs[:14]
        a = torch.clamp(
            0.55 * err + 0.01 * theta[:14] + 0.08 * theta[14:] * (refs[t] - x[:14]),
            -0.03,
            0.03,
        )
        x, _ = model(x.unsqueeze(0), c_obs.unsqueeze(0), a.unsqueeze(0))
        x = x.squeeze(0)
        # advance c_obs belief with known U0 update (partial)
        target = torch.clamp(c_obs[:14] + a, model.lower, model.upper)
        filt = 0.5 * target + 0.5 * c_obs[14:28]
        delay = c_obs[28:29]
        c_obs = torch.cat([target, filt, delay])
        total = total + (-((x[:14] - refs[t]) ** 2).mean())
        prev = x
    return total


def imagined_mc(model, theta, start, horizon=H):
    return E.imagined(model, theta, start, model.lower, model.upper, horizon=horizon)


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def policy_train_eagm(model, starts, updates=50, lr=0.03):
    theta = torch.zeros(28, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(updates):
        loss = -torch.stack([imagined_eagm(model, theta, s) for s in starts]).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_([theta], 1)
        opt.step()
        with torch.no_grad():
            theta.clamp_(-1, 1)
    return theta.detach().numpy().copy()


def policy_train_mc(model, starts, updates=50, lr=0.03):
    theta = torch.zeros(28, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(updates):
        loss = -torch.stack([imagined_mc(model, theta, s) for s in starts]).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_([theta], 1)
        opt.step()
        with torch.no_grad():
            theta.clamp_(-1, 1)
    return theta.detach().numpy().copy()


def policy_return(env, theta, starts, horizon=16):
    return np.asarray([E.physical(env, theta, s, horizon=horizon)[0] for s in starts])


def main():
    out = EAGM_ROOT / "runs" / "stage3_b_dir_seed0"
    out.mkdir(parents=True, exist_ok=False)
    cfg = ControllerBConfig(gain=1.05, extra_lag=1, rate_limit=0.045, smooth_beta=0.25)
    env = PartialUnknownArmEnv(cfg)
    t0 = time.time()
    print("collect...", flush=True)
    rows = collect(env, episodes=12, horizon=200, seed=0)
    rng = np.random.default_rng(0)
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    n_train = int(0.7 * len(rows))
    train_i, test_i = idx[:n_train], idx[n_train:]
    stats = fit_stats([rows[i] for i in train_i])

    # Direction starts: first 12 delay-2 train rows' start dicts
    dir_candidates = [rows[i] for i in train_i if rows[i]["delay"] == 2][:12]
    print(f"dir labels on {len(dir_candidates)} starts...", flush=True)
    dir_g, dir_ok = [], []
    for r in dir_candidates:
        g, ok = sim_grad_masked(env, r["start"])
        dir_g.append(g)
        dir_ok.append(ok)
    s0 = float(np.median(np.abs(np.stack(dir_g)))) or 1.0
    dir_g_t = torch.as_tensor(np.stack(dir_g), dtype=DTYPE)
    dir_ok_t = torch.as_tensor(np.stack(dir_ok), dtype=DTYPE)

    def pack(subset):
        return {
            k: torch.as_tensor(np.stack([rows[i][k] for i in subset]), dtype=DTYPE)
            for k in ("x", "y", "a", "c", "c_obs", "u_true")
        }

    tr, te = pack(train_i), pack(test_i)
    starts_train = [rows[i]["start"] for i in train_i if rows[i]["delay"] == 2][:12]
    starts_test = [rows[i]["start"] for i in test_i if rows[i]["delay"] == 2][:12]

    updates = 1000
    lam_u, lam_d = 1.0, 0.1

    mc = McWorld(stats)
    mu = MuOracle(stats)
    eagm_base = EAGMBase(stats)
    eagm_dir = EAGMBase(stats)

    def train(model, kind):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        log = []
        n = tr["x"].shape[0]
        for it in range(updates):
            b = torch.randint(0, n, (128,))
            if kind == "mc":
                pred = model(tr["x"][b], tr["c"][b], tr["a"][b])
                loss = (((pred - tr["y"][b]) / model.ystd) ** 2).mean()
            elif kind == "mu":
                pred = model.forward_u(tr["x"][b], tr["u_true"][b])
                loss = (((pred - tr["y"][b]) / model.ystd) ** 2).mean()
            else:
                y_hat, u_hat = model(tr["x"][b], tr["c_obs"][b], tr["a"][b])
                dyn = (((y_hat - tr["y"][b]) / model.ystd) ** 2).mean()
                cmd = (((u_hat - tr["u_true"][b]) / model.ustd) ** 2).mean()
                loss = dyn + lam_u * cmd
                if kind == "eagm_dir" and dir_candidates and (it % 10 == 0):
                    j = (it // 10) % len(dir_candidates)
                    s = dir_candidates[j]["start"]
                    th = torch.zeros(28, requires_grad=True)
                    J = imagined_eagm(model, th, s)
                    g = torch.autograd.grad(J, th, create_graph=True)[0]
                    loss = loss + lam_d * huber_dir_loss(
                        g.unsqueeze(0),
                        dir_g_t[j : j + 1],
                        dir_ok_t[j : j + 1],
                        s0=s0,
                    )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            if (it + 1) % 200 == 0:
                log.append(dict(update=it + 1, loss=float(loss.detach())))
                print(f"  {kind} it {it+1} loss {float(loss.detach()):.5f}", flush=True)
        return log

    print("train mc/mu/eagm...", flush=True)
    logs = {
        "mc": train(mc, "mc"),
        "mu": train(mu, "mu"),
        "eagm_base": train(eagm_base, "eagm_base"),
        "eagm_dir": train(eagm_dir, "eagm_dir"),
    }

    # Eval dynamics + cmd
    with torch.no_grad():
        mc_nmse = (((mc(te["x"], te["c"], te["a"]) - te["y"]) / mc.ystd) ** 2).mean().item()
        mu_nmse = (((mu.forward_u(te["x"], te["u_true"]) - te["y"]) / mu.ystd) ** 2).mean().item()
        yb, ub = eagm_base(te["x"], te["c_obs"], te["a"])
        yd, ud = eagm_dir(te["x"], te["c_obs"], te["a"])
        base_nmse = (((yb - te["y"]) / eagm_base.ystd) ** 2).mean().item()
        dir_nmse = (((yd - te["y"]) / eagm_dir.ystd) ** 2).mean().item()
        base_cmd = (ub - te["u_true"]).abs().mean().item()
        dir_cmd = (ud - te["u_true"]).abs().mean().item()

    # Gradient cosine on test starts
    def eagm_grad(model, start):
        th = torch.zeros(28, requires_grad=True)
        J = imagined_eagm(model, th, start)
        g = torch.autograd.grad(J, th)[0].detach().numpy()
        return g

    def mc_grad(model, start):
        return E.gradient(model, start, model.lower, model.upper)

    test_g = [sim_grad_masked(env, s)[0] for s in starts_test]
    cos = {}
    for name, model, fn in [
        ("mc", mc, mc_grad),
        ("eagm_base", eagm_base, eagm_grad),
        ("eagm_dir", eagm_dir, eagm_grad),
    ]:
        cos[name] = float(np.median([cosine(fn(model, s), g) for s, g in zip(starts_test, test_g)]))

    # Policy returns
    print("policy...", flush=True)
    rets = {}
    for name, model, trainer in [
        ("mc", mc, policy_train_mc),
        ("eagm_base", eagm_base, policy_train_eagm),
        ("eagm_dir", eagm_dir, policy_train_eagm),
    ]:
        theta = trainer(model, starts_train, updates=50)
        rets[name] = float(policy_return(env, theta, starts_test, horizon=16).mean())

    payload = dict(
        controller="B",
        config=cfg.__dict__,
        updates=updates,
        lam_u=lam_u,
        lam_d=lam_d,
        n_train=len(train_i),
        n_test=len(test_i),
        n_dir=len(dir_candidates),
        n_policy_test=len(starts_test),
        dynamics_nmse=dict(mc=mc_nmse, mu_oracle=mu_nmse, eagm_base=base_nmse, eagm_dir=dir_nmse),
        cmd_mae=dict(eagm_base=base_cmd, eagm_dir=dir_cmd),
        cosine_median=cos,
        policy_return_mean=rets,
        logs=logs,
        wall_seconds=time.time() - t0,
        note="Single-seed Controller B; masked Dir loss on; exploratory.",
    )
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    torch.save({"state": eagm_dir.state_dict(), "stats": stats}, out / "eagm_dir_seed0.pt")
    print(json.dumps({k: payload[k] for k in (
        "dynamics_nmse", "cmd_mae", "cosine_median", "policy_return_mean", "wall_seconds"
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
