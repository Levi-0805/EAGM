"""Stage-2 EAGM comparison on OpenArm: Mu vs Mu+full-gradient direction loss,
with Mx and Mc as matched 1,000-update references.

All models are evaluated with the same 28-parameter actor, the same 12 training
starts, 50 Adam updates, and the same 36 held-out delay-2 test starts.
"""
from pathlib import Path
import json, time, argparse
import numpy as np
import torch

import experiment as E
from sim_env import ArmEnv
import eagm_pilot as P

ROOT = Path(__file__).resolve().parent
PILOT = ROOT / "runs" / "pilot_20261101"
REF = ROOT / "runs" / "extended_20261102" / "alpha05"


def load_world(path):
    payload = torch.load(path, weights_only=False)
    stats = {k: np.asarray(v) for k, v in payload["stats"].items()}
    m = E.World(payload["kind"], stats)
    m.load_state_dict(payload["state"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--updates", type=int, default=1000)
    ap.add_argument("--lam", type=float, default=.1)
    ap.add_argument("--policy-updates", type=int, default=50)
    ap.add_argument("--out", default="runs/eagm_stage2_20261112")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    lo, hi = P.load_bounds()
    lo_t, hi_t = torch.as_tensor(lo, dtype=torch.float32), torch.as_tensor(hi, dtype=torch.float32)
    npz = np.load(PILOT / "transitions.npz")
    starts = torch.load(PILOT / "starts.pt", weights_only=False)
    env = ArmEnv()
    stats = P.fit_stats(npz, lo, hi)
    train_starts = [s for s in starts["train"] if s["delay"] == 2]
    test_starts = [s for s in starts["test"] if s["delay"] == 2]
    print(f"train starts {len(train_starts)} test starts {len(test_starts)}", flush=True)

    all_rows = []
    t0 = time.time()
    for seed in args.seeds:
        print(f"seed {seed}: labels", flush=True)
        train_g, train_ok = [], []
        for s in train_starts[:12]:
            g, ok = P.sim_grad(env, s, lo_t, hi_t)
            train_g.append(g)
            train_ok.append(ok)
        test_g, test_ok = [], []
        for s in test_starts:
            g, ok = P.sim_grad(env, s, lo_t, hi_t)
            test_g.append(g)
            test_ok.append(ok)
        s0 = float(np.median(np.abs(np.stack(train_g)))) or 1.0

        models = {
            "mx": load_world(REF / f"mx_seed{seed}.pt"),
            "mc": load_world(REF / f"mc_seed{seed}.pt"),
        }
        log = []
        models["mu_base"], stat_base = P.train_model(
            npz, stats, seed, updates=args.updates, lam=0.0, log=log)
        models["mu_dir"], stat_dir = P.train_model(
            npz, stats, seed, updates=args.updates,
            direction_starts=train_starts[:12], direction_grads=train_g,
            lam=args.lam, s0=s0, log=log)
        (out / f"seed{seed}_learning.json").write_text(json.dumps(log, indent=2))

        for name, m in models.items():
            cos = [P.cosine(E.gradient(m, s, lo_t, hi_t), g)
                   for s, g, ok in zip(test_starts, test_g, test_ok)
                   if ok.mean() > .5]
            theta = P.policy_train(m, train_starts[:12], lo_t, hi_t,
                                   updates=args.policy_updates)
            ret = P.policy_return(env, theta, test_starts, horizon=16)
            row = dict(seed=seed, model=name,
                       cosine_median=float(np.median(cos)),
                       cosine_mean=float(np.mean(cos)),
                       cosine_n=len(cos),
                       policy_return_mean=float(ret.mean()),
                       returns=ret.tolist(),
                       theta_norm=float(np.linalg.norm(theta)))
            if name in ("mu_base", "mu_dir"):
                row["validation"] = (stat_base if name == "mu_base" else stat_dir)["validation"]
            all_rows.append(row)
            print(f"  {name:8s} cos {row['cosine_median']:.4f} "
                  f"ret {row['policy_return_mean']:.8f}", flush=True)
        (out / "partial_results.json").write_text(json.dumps(
            dict(rows=all_rows, wall_seconds=time.time() - t0), indent=2))
    (out / "results.json").write_text(json.dumps(
        dict(rows=all_rows, lam=args.lam, updates=args.updates,
             policy_updates=args.policy_updates,
             train_starts=len(train_starts), test_starts=len(test_starts),
             s0=None, wall_seconds=time.time() - t0,
             note="Held-out test trajectories were inspected in earlier revisions; "
                  "this is an exploratory comparison, not a confirmatory test."), indent=2))
    print("wrote", out, "wall", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
