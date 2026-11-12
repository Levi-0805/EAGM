"""EAGM minimal falsification pilot: does full-gradient direction supervision
improve held-out gradient fidelity and model-based policy return?

This is a scratch prototype, not the paper pipeline.  It reuses the archived
pilot transitions and frozen start states, and never writes into an existing
run directory.
"""
from pathlib import Path
import json, time, copy, argparse
import numpy as np
import torch
from torch import nn

import experiment as E
from sim_env import ArmEnv

ROOT = Path(__file__).resolve().parent
PILOT = ROOT / "runs" / "pilot_20261101"
DTYPE = torch.float32
H = 8
K = 3
WIDTH = 160
UPDATES = 1000
BATCH = 128


def load_bounds():
    meta = json.loads((PILOT / "environment.json").read_text(encoding="utf-8"))
    return (np.asarray(meta["target_lower_rad"], dtype=np.float64),
            np.asarray(meta["target_upper_rad"], dtype=np.float64))


class MuWorld(nn.Module):
    def __init__(self, stats, width=WIDTH):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(42, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 28))
        for name, val in stats.items():
            self.register_buffer(name, torch.tensor(val, dtype=DTYPE))

    def command(self, c, a):
        target = torch.maximum(torch.minimum(c[..., :14] + a, self.upper), self.lower)
        filt = self.alpha * target + (1 - self.alpha) * c[..., 14:28]
        delay = c[..., 56:57]
        return torch.where(delay == 0, filt,
                           torch.where(delay == 1, c[..., 42:56], c[..., 28:42]))

    def forward(self, x, c, a, h=None):
        xx = (x - self.xmean) / self.xstd
        u = self.command(c, a)
        uu = (u - self.umean) / self.ustd
        return x + self.net(torch.cat([xx, uu], -1)) * self.ystd + self.ymean


def shaped(npz, split, name):
    arr = npz[f"{split}_{name}"]
    return arr.reshape(-1, 300, *arr.shape[1:])


def fit_stats(npz, lo, hi):
    tr = {k: shaped(npz, "train", k) for k in ["x", "c", "a", "y"]}
    x = tr["x"].reshape(-1, 28)
    c = tr["c"].reshape(-1, 57)
    a = tr["a"].reshape(-1, 14)
    delta = (tr["y"] - tr["x"]).reshape(-1, 28)
    target = np.clip(c[:, :14] + np.clip(a, -.03, .03), lo, hi)
    filt = .5 * target + .5 * c[:, 14:28]
    d = c[:, 56].astype(int)
    u = np.where(d[:, None] == 0, filt,
                 np.where(d[:, None] == 1, c[:, 42:56], c[:, 28:42]))
    return dict(xmean=x.mean(0), xstd=np.maximum(x.std(0), .01),
                umean=u.mean(0), ustd=np.maximum(u.std(0), .001),
                ymean=delta.mean(0), ystd=np.maximum(delta.std(0), .001),
                lower=lo, upper=hi, alpha=np.asarray(.5))


def train_model(npz, stats, seed, updates=UPDATES, direction_starts=None,
                direction_grads=None, lam=0.0, s0=1.0, log=None, dir_every=10,
                dir_batch=2):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr = {k: shaped(npz, "train", k) for k in ["x", "c", "a", "y", "h"]}
    va = {k: shaped(npz, "val", k) for k in ["x", "c", "a", "y", "h"]}
    m = MuWorld(stats)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    stacked = {k: torch.as_tensor(tr[k], dtype=DTYPE) for k in ["x", "c", "a", "y", "h"]}
    valid = {k: torch.as_tensor(va[k].reshape(-1, *va[k].shape[2:]), dtype=DTYPE)
             for k in ["x", "c", "a", "y", "h"]}
    n_ep = stacked["x"].shape[0]
    best, best_state, best_it = float("inf"), None, -1
    for it in range(updates):
        e = torch.tensor(rng.integers(n_ep, size=BATCH))
        t = torch.tensor(rng.integers(300 - K, size=BATCH))
        xx, hh, loss = stacked["x"][e, t], stacked["h"][e, t], 0
        for k in range(K):
            cc, aa, yy = stacked["c"][e, t + k], stacked["a"][e, t + k], stacked["y"][e, t + k]
            yp = m(xx, cc, aa, hh)
            loss = loss + (((yp - yy) / m.ystd) ** 2).mean() / K
            hh = E.history_next(hh, xx, aa)
            xx = yp
        if lam > 0 and direction_starts and (it % dir_every == 0):
            idx = (it * dir_batch) % len(direction_starts)
            gs = []
            for j in range(dir_batch):
                s = direction_starts[(idx + j) % len(direction_starts)]
                th = torch.zeros(28, requires_grad=True)
                J = E.imagined(m, th, s, m.lower, m.upper)
                g = torch.autograd.grad(J, th, create_graph=True)[0]
                gs.append(g)
            gm = torch.stack(gs)
            gt = torch.as_tensor(np.stack([direction_grads[(idx + j) % len(direction_starts)]
                                           for j in range(dir_batch)]), dtype=DTYPE)
            loss = loss + lam * nn.functional.huber_loss(gm / s0, gt / s0, delta=1.0)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 5)
        opt.step()
        if (it + 1) % 100 == 0 or it == updates - 1:
            with torch.no_grad():
                vl = (((m(valid["x"], valid["c"], valid["a"], valid["h"]) - valid["y"])
                       / m.ystd) ** 2).mean().item()
            if vl < best:
                best, best_state, best_it = vl, copy.deepcopy(m.state_dict()), it + 1
            if log is not None:
                log.append(dict(update=it + 1, train=float(loss.detach()), validation=vl))
    m.load_state_dict(best_state)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, dict(validation=best, best_update=best_it,
                   parameters=sum(p.numel() for p in m.parameters()))


def sim_grad(env, start, lo, hi, eps=.05, scales=(.5, 1., 2.)):
    """Return central-difference gradient components and a stability mask."""
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
        stable.append(spread <= max(1e-9, .1 * abs(v)))
    return np.asarray(out), np.asarray(stable)


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def policy_train(model, starts, lo, hi, updates=30, lr=.03, horizon=H):
    theta = torch.zeros(28, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(updates):
        loss = -torch.stack([E.imagined(model, theta, s, lo, hi, horizon=horizon)
                             for s in starts]).mean()
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=UPDATES)
    ap.add_argument("--lam", type=float, default=.1)
    ap.add_argument("--policy-updates", type=int, default=30)
    ap.add_argument("--out", default="runs/eagm_pilot_20261112")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    lo, hi = load_bounds()
    lo_t, hi_t = torch.as_tensor(lo, dtype=DTYPE), torch.as_tensor(hi, dtype=DTYPE)
    npz = np.load(PILOT / "transitions.npz")
    starts = torch.load(PILOT / "starts.pt", weights_only=False)
    env = ArmEnv()
    stats = fit_stats(npz, lo, hi)

    # Training probes: delay-2 starts only, primary delay of the paper.
    train_starts = [s for s in starts["train"] if s["delay"] == 2][:12]
    test_starts = [s for s in starts["test"] if s["delay"] == 2][:12]
    t0 = time.time()
    train_g, train_ok = [], []
    for s in train_starts:
        g, ok = sim_grad(env, s, lo_t, hi_t)
        train_g.append(g)
        train_ok.append(ok)
    test_g, test_ok = [], []
    for s in test_starts:
        g, ok = sim_grad(env, s, lo_t, hi_t)
        test_g.append(g)
        test_ok.append(ok)
    s0 = float(np.median(np.abs(np.stack(train_g)))) or 1.0
    print(f"labels {time.time()-t0:.1f}s s0={s0:.3g} "
          f"stable_frac={np.mean(np.stack(train_ok)):.3f}", flush=True)

    results = {}
    for name, lam in [("mu_base", 0.0), ("mu_dir", args.lam)]:
        print("training", name, flush=True)
        log = []
        m, stat = train_model(npz, stats, args.seed, updates=args.updates,
                              direction_starts=train_starts, direction_grads=train_g,
                              lam=lam, s0=s0, log=log)
        (out / f"{name}_learning.json").write_text(json.dumps(log, indent=2))
        torch.save({"state": m.state_dict(), "stats": stats, "kind": name,
                    "seed": args.seed}, out / f"{name}.pt")
        cos = [cosine(E.gradient(m, s, lo_t, hi_t), g)
               for s, g, ok in zip(test_starts, test_g, test_ok) if ok.mean() > .5]
        # Policy: same train starts, same updates, only the model differs.
        theta = policy_train(m, train_starts[:6], lo_t, hi_t,
                             updates=args.policy_updates)
        ret = policy_return(env, theta, test_starts, horizon=16)
        results[name] = dict(stat=stat, cosine_median=float(np.median(cos)),
                             cosine_min=float(np.min(cos)), cosine_n=len(cos),
                             policy_return_mean=float(ret.mean()),
                             policy_return_per_start=ret.tolist(),
                             theta=theta.tolist())
        print(name, results[name]["stat"], "cos", results[name]["cosine_median"],
              "ret", results[name]["policy_return_mean"], flush=True)
    (out / "results.json").write_text(json.dumps(
        dict(seed=args.seed, lam=args.lam, s0=s0, results=results,
             wall_seconds=time.time() - t0,
             note="Scratch falsification pilot; not a paper result."), indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
