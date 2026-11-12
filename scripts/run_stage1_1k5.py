#!/usr/bin/env python3
"""Stage-1 five-seed 1k reproduction (Mc/Mx archived + Mu / Mu+Dir retrain)."""
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
from controllers.compatible_arm_env import CompatibleArmEnv  # noqa: E402

PILOT = LEGACY / "runs" / "pilot_20261101"
REF = LEGACY / "runs" / "extended_20261102" / "alpha05"
ARCH = (
    EAGM_ROOT
    / "vendor"
    / "EAGM_archive_20261112"
    / "runs"
    / "eagm_stage2_1k5_20261112"
    / "results.json"
)


def load_world(path: Path):
    payload = torch.load(path, weights_only=False, map_location="cpu")
    stats = {k: np.asarray(v) for k, v in payload["stats"].items()}
    m = E.World(payload["kind"], stats)
    m.load_state_dict(payload["state"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def main():
    seeds = [0, 1, 2, 3, 4]
    updates = 1000
    lam = 0.1
    policy_updates = 50
    out = EAGM_ROOT / "runs" / "stage1_1k5"
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
    (out / "environment.json").write_text(
        json.dumps(
            {
                "host": platform.uname()._asdict(),
                "torch": torch.__version__,
                "mujoco_env": "CompatibleArmEnv",
                "updates": updates,
                "lam": lam,
                "seeds": seeds,
            },
            indent=2,
            default=str,
        )
    )

    archived = {}
    if ARCH.exists():
        for r in json.loads(ARCH.read_text())["rows"]:
            archived[(r["seed"], r["model"])] = r

    all_rows = []
    t0 = time.time()
    for seed in seeds:
        print(f"seed {seed}: labels", flush=True)
        train_g = [P.sim_grad(env, s, lo_t, hi_t)[0] for s in train_starts[:12]]
        test_pairs = [P.sim_grad(env, s, lo_t, hi_t) for s in test_starts]
        s0 = float(np.median(np.abs(np.stack(train_g)))) or 1.0
        models = {
            "mx": load_world(REF / f"mx_seed{seed}.pt"),
            "mc": load_world(REF / f"mc_seed{seed}.pt"),
        }
        log = []
        models["mu_base"], stat_base = P.train_model(
            npz, stats, seed, updates=updates, lam=0.0, log=log
        )
        models["mu_dir"], stat_dir = P.train_model(
            npz,
            stats,
            seed,
            updates=updates,
            direction_starts=train_starts[:12],
            direction_grads=train_g,
            lam=lam,
            s0=s0,
            log=log,
        )
        (out / f"seed{seed}_learning.json").write_text(json.dumps(log, indent=2))
        torch.save(
            {"state": models["mu_base"].state_dict(), "stats": stats, "kind": "mu_base", "seed": seed},
            out / f"mu_base_seed{seed}.pt",
        )
        torch.save(
            {"state": models["mu_dir"].state_dict(), "stats": stats, "kind": "mu_dir", "seed": seed},
            out / f"mu_dir_seed{seed}.pt",
        )
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
                cosine_mean=float(np.mean(cos)),
                policy_return_mean=float(ret.mean()),
                returns=ret.tolist(),
                theta_norm=float(np.linalg.norm(theta)),
            )
            if name == "mu_base":
                row["validation"] = stat_base["validation"]
            if name == "mu_dir":
                row["validation"] = stat_dir["validation"]
            a = archived.get((seed, name))
            if a:
                row["archive_cosine_median"] = a["cosine_median"]
                row["archive_return"] = a["policy_return_mean"]
                row["delta_cosine"] = row["cosine_median"] - a["cosine_median"]
                row["delta_return"] = row["policy_return_mean"] - a["policy_return_mean"]
            all_rows.append(row)
            print(
                f"  {name:8s} cos {row['cosine_median']:.4f} ret {row['policy_return_mean']:.8f}",
                flush=True,
            )
        (out / "partial_results.json").write_text(
            json.dumps({"rows": all_rows, "wall_seconds": time.time() - t0}, indent=2)
        )

    payload = dict(
        rows=all_rows,
        updates=updates,
        lam=lam,
        policy_updates=policy_updates,
        wall_seconds=time.time() - t0,
        tolerances={"cosine_abs": 0.02, "return_abs": 2e-4},
        note="Stage-1 five-seed 1k reproduction on GPU host via CompatibleArmEnv.",
    )
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    # summary max abs deltas
    dcos = [abs(r["delta_cosine"]) for r in all_rows if "delta_cosine" in r]
    dret = [abs(r["delta_return"]) for r in all_rows if "delta_return" in r]
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
