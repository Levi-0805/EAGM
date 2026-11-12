"""Decisive 3,000-update comparison: Mu, Mu+Dir, and the archived Mc.

This is the budget used by the paper's headline gradient result.  The question
is whether direction supervision improves held-out policy return over both the
exact-command base and the complete-memory baseline.
"""
from pathlib import Path
import json, time, argparse
import numpy as np
import torch

import experiment as E
from sim_env import ArmEnv
import eagm_pilot as P
import eagm_policy_eval as PE

ROOT = Path(__file__).resolve().parent
PILOT = ROOT / "runs" / "pilot_20261101"
MC = ROOT / "runs" / "revision4_20261111" / "long3000"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=3000)
    ap.add_argument("--lam", type=float, default=.1)
    ap.add_argument("--policy-updates", type=int, default=50)
    ap.add_argument("--out", default="runs/eagm_mudir3k_20261112")
    args = ap.parse_args()
    out = ROOT / args.out; out.mkdir(parents=True, exist_ok=False)
    lo, hi = P.load_bounds()
    lo_t, hi_t = torch.as_tensor(lo, dtype=torch.float32), torch.as_tensor(hi, dtype=torch.float32)
    npz = np.load(PILOT / "transitions.npz")
    starts = torch.load(PILOT / "starts.pt", weights_only=False)
    env = ArmEnv()
    stats = P.fit_stats(npz, lo, hi)
    train_starts = [s for s in starts["train"] if s["delay"] == 2]
    test_starts = [s for s in starts["test"] if s["delay"] == 2]
    rows = []
    t0 = time.time()
    for seed in range(5):
        print(f"seed {seed}: labels", flush=True)
        train_g = [P.sim_grad(env, s, lo_t, hi_t)[0] for s in train_starts[:12]]
        test_pairs = [P.sim_grad(env, s, lo_t, hi_t) for s in test_starts]
        s0 = float(np.median(np.abs(np.stack(train_g)))) or 1.0
        models = {"mc3k": PE.load_world(MC / f"mc_seed{seed}.pt")}
        for name, lam in [("mu3k", 0.0), ("mudir3k", args.lam)]:
            m, stat = P.train_model(
                npz, stats, seed, updates=args.updates,
                direction_starts=train_starts[:12] if lam > 0 else None,
                direction_grads=train_g if lam > 0 else None,
                lam=lam, s0=s0)
            models[name] = m
            torch.save({"state": m.state_dict(), "stats": stats, "kind": name,
                        "seed": seed}, out / f"{name}_seed{seed}.pt")
            print(f"  trained {name} val {stat['validation']:.5f}", flush=True)
        for name, m in models.items():
            cos = [P.cosine(E.gradient(m, s, lo_t, hi_t), g)
                   for s, (g, ok) in zip(test_starts, test_pairs) if ok.mean() > .5]
            theta = P.policy_train(m, train_starts[:12], lo_t, hi_t,
                                   updates=args.policy_updates)
            ret = P.policy_return(env, theta, test_starts, horizon=16)
            rows.append(dict(seed=seed, model=name, cosine_median=float(np.median(cos)),
                             return_mean=float(ret.mean()), returns=ret.tolist(),
                             theta_norm=float(np.linalg.norm(theta))))
            print(f"  {name:8s} cos {rows[-1]['cosine_median']:.4f} "
                  f"ret {rows[-1]['return_mean']:.8f}", flush=True)
        (out / "partial_results.json").write_text(json.dumps(
            dict(rows=rows, wall_seconds=time.time() - t0), indent=2))
    (out / "results.json").write_text(json.dumps(
        dict(rows=rows, updates=args.updates, lam=args.lam,
             policy_updates=args.policy_updates, wall_seconds=time.time() - t0,
             note="3,000-update checkpoints; 12 delay-2 train and 12 delay-2 test starts; "
                  "test trajectories previously inspected; exploratory."), indent=2))
    print("wrote", out, "wall", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
