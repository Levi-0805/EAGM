"""Evaluate model-based policy learning for archived 3,000-update checkpoints.

Loads Mc and Mu checkpoints trained at the same 3,000-update budget and runs the
identical 28-parameter policy optimization used in the Stage-2 prototype.
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
MC = ROOT / "runs" / "revision4_20261111" / "long3000"
MU = ROOT / "runs" / "revision4_20261111" / "mu3000" / "models"


def load_world(path):
    payload = torch.load(path, weights_only=False)
    stats = {k: np.asarray(v) for k, v in payload["stats"].items()}
    m = E.World(payload["kind"], stats)
    m.load_state_dict(payload["state"]); m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m


def load_mu(path):
    payload = torch.load(path, weights_only=False)
    stats = {k: np.asarray(v) for k, v in payload["stats"].items()}
    stats.setdefault("alpha", np.asarray(.5))
    m = P.MuWorld(stats)
    state = dict(payload["state"]); state.setdefault("alpha", torch.tensor(.5))
    m.load_state_dict(state); m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-updates", type=int, default=50)
    ap.add_argument("--out", default="runs/eagm_policy3k_20261112")
    args = ap.parse_args()
    out = ROOT / args.out; out.mkdir(parents=True, exist_ok=False)
    lo, hi = P.load_bounds()
    lo_t, hi_t = torch.as_tensor(lo, dtype=torch.float32), torch.as_tensor(hi, dtype=torch.float32)
    starts = torch.load(PILOT / "starts.pt", weights_only=False)
    env = ArmEnv()
    train_starts = [s for s in starts["train"] if s["delay"] == 2]
    test_starts = [s for s in starts["test"] if s["delay"] == 2]
    rows = []
    t0 = time.time()
    for seed in range(5):
        models = {"mc3k": load_world(MC / f"mc_seed{seed}.pt"),
                  "mu3k": load_mu(MU / f"mu_seed{seed}.pt")}
        for name, m in models.items():
            theta = P.policy_train(m, train_starts[:12], lo_t, hi_t,
                                   updates=args.policy_updates)
            ret = P.policy_return(env, theta, test_starts, horizon=16)
            rows.append(dict(seed=seed, model=name, return_mean=float(ret.mean()),
                             returns=ret.tolist(), theta_norm=float(np.linalg.norm(theta))))
            print(seed, name, float(ret.mean()), flush=True)
    (out / "results.json").write_text(json.dumps(
        dict(rows=rows, policy_updates=args.policy_updates,
             wall_seconds=time.time() - t0,
             note="Archived 3,000-update checkpoints; same policy protocol as Stage 2; "
                  "test trajectories previously inspected."), indent=2))
    print("wrote", out, "wall", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
