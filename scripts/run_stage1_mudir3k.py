#!/usr/bin/env python3
"""Stage-1 five-seed 3k: Mc (archived) vs Mu vs Mu+Dir."""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

EAGM_ROOT = Path(__file__).resolve().parents[1]
LEGACY = (EAGM_ROOT / "vendor" / "legacy_sim").resolve()
sys.path.insert(0, str(LEGACY))
sys.path.insert(0, str(EAGM_ROOT))

os.environ.setdefault(
    "OPENARM_MODEL_PATH",
    "/home/aaa/HIL-OpenArm/third_party/openarm_mujoco/v2/openarm_bimanual.xml",
)

import experiment as E  # noqa: E402
import eagm_pilot as P  # noqa: E402
import eagm_policy_eval as PE  # noqa: E402
from controllers.compatible_arm_env import CompatibleArmEnv  # noqa: E402

PILOT = LEGACY / "runs" / "pilot_20261101"
MC = LEGACY / "runs" / "revision4_20261111" / "long3000"
ARCH = (
    EAGM_ROOT
    / "vendor"
    / "EAGM_archive_20261112"
    / "runs"
    / "eagm_mudir3k_20261112"
    / "results.json"
)


def main():
    updates = 3000
    lam = 0.1
    policy_updates = 50
    out = EAGM_ROOT / "runs" / "stage1_mudir3k"
    out.mkdir(parents=True, exist_ok=False)
    lo, hi = P.load_bounds()
    lo_t = torch.as_tensor(lo, dtype=torch.float32)
    hi_t = torch.as_tensor(hi, dtype=torch.float32)
    npz = np.load(PILOT / "transitions.npz")
    starts = torch.load(PILOT / "starts.pt", weights_only=False)
    env = CompatibleArmEnv()
    stats = P.fit_stats(npz, lo, hi)
    train_starts = [s for s in starts["train"] if s["delay"] == 2]
    test_starts = [s for s in starts["test"] if s["delay"] == 2]
    archived = {}
    if ARCH.exists():
        for r in json.loads(ARCH.read_text())["rows"]:
            archived[(r["seed"], r["model"])] = r
    (out / "environment.json").write_text(
        json.dumps(
            {
                "host": platform.uname()._asdict(),
                "torch": torch.__version__,
                "updates": updates,
                "lam": lam,
            },
            indent=2,
            default=str,
        )
    )
    rows = []
    t0 = time.time()
    for seed in range(5):
        print(f"seed {seed}: labels", flush=True)
        train_g = [P.sim_grad(env, s, lo_t, hi_t)[0] for s in train_starts[:12]]
        test_pairs = [P.sim_grad(env, s, lo_t, hi_t) for s in test_starts]
        s0 = float(np.median(np.abs(np.stack(train_g)))) or 1.0
        models = {"mc3k": PE.load_world(MC / f"mc_seed{seed}.pt")}
        for name, lam_i in [("mu3k", 0.0), ("mudir3k", lam)]:
            m, stat = P.train_model(
                npz,
                stats,
                seed,
                updates=updates,
                direction_starts=train_starts[:12] if lam_i > 0 else None,
                direction_grads=train_g if lam_i > 0 else None,
                lam=lam_i,
                s0=s0,
            )
            models[name] = m
            torch.save(
                {"state": m.state_dict(), "stats": stats, "kind": name, "seed": seed},
                out / f"{name}_seed{seed}.pt",
            )
            print(f"  trained {name} val {stat['validation']:.5f}", flush=True)
        for name, m in models.items():
            cos = [
                P.cosine(E.gradient(m, s, lo_t, hi_t), g)
                for s, (g, ok) in zip(test_starts, test_pairs)
                if ok.mean() > 0.5
            ]
            theta = P.policy_train(
                m, train_starts[:12], lo_t, hi_t, updates=policy_updates
            )
            ret = P.policy_return(env, theta, test_starts, horizon=16)
            row = dict(
                seed=seed,
                model=name,
                cosine_median=float(np.median(cos)),
                return_mean=float(ret.mean()),
                returns=ret.tolist(),
                theta_norm=float(np.linalg.norm(theta)),
            )
            a = archived.get((seed, name))
            if a:
                row["archive_cosine"] = a["cosine_median"]
                row["archive_return"] = a["return_mean"]
                row["delta_cosine"] = row["cosine_median"] - a["cosine_median"]
                row["delta_return"] = row["return_mean"] - a["return_mean"]
            rows.append(row)
            print(
                f"  {name:8s} cos {row['cosine_median']:.4f} ret {row['return_mean']:.8f}",
                flush=True,
            )
        (out / "partial_results.json").write_text(
            json.dumps({"rows": rows, "wall_seconds": time.time() - t0}, indent=2)
        )
    payload = dict(
        rows=rows,
        updates=updates,
        lam=lam,
        policy_updates=policy_updates,
        wall_seconds=time.time() - t0,
        tolerances={"cosine_abs": 0.02, "return_abs": 2e-4},
    )
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    dcos = [abs(r["delta_cosine"]) for r in rows if "delta_cosine" in r]
    dret = [abs(r["delta_return"]) for r in rows if "delta_return" in r]
    print(
        "wrote",
        out,
        "wall",
        round(payload["wall_seconds"], 1),
        "max|dcos|",
        max(dcos) if dcos else None,
        "max|dret|",
        max(dret) if dret else None,
        flush=True,
    )


if __name__ == "__main__":
    main()
