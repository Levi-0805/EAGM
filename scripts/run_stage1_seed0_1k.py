#!/usr/bin/env python3
"""Stage-1 seed-0 reproduction: Mx/Mc (archived) + retrain Mu / Mu+Dir at 1k.

Writes only under eagm_gpu/runs/. Does not modify CVPR2027 or legacy runs/.
"""
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

os.environ.setdefault(
    "OPENARM_MODEL_PATH",
    "/home/aaa/HIL-OpenArm/third_party/openarm_mujoco/v2/openarm_bimanual.xml",
)

import experiment as E  # noqa: E402
import eagm_pilot as P  # noqa: E402

sys.path.insert(0, str(EAGM_ROOT))
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
    seed = 0
    updates = 1000
    lam = 0.1
    policy_updates = 50
    out = EAGM_ROOT / "runs" / "stage1_seed0_1k"
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

    meta = {
        "host": platform.uname()._asdict(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu",
        "openarm": os.environ.get("OPENARM_MODEL_PATH"),
        "mujoco_model": str(env.model_path),
        "seed": seed,
        "updates": updates,
        "lam": lam,
    }
    (out / "environment.json").write_text(json.dumps(meta, indent=2, default=str))

    print("labels...", flush=True)
    t0 = time.time()
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
    label_wall = time.time() - t0

    models = {
        "mx": load_world(REF / f"mx_seed{seed}.pt"),
        "mc": load_world(REF / f"mc_seed{seed}.pt"),
    }
    log = []
    print("train mu_base...", flush=True)
    t1 = time.time()
    models["mu_base"], stat_base = P.train_model(
        npz, stats, seed, updates=updates, lam=0.0, log=log
    )
    train_base_wall = time.time() - t1
    print("train mu_dir...", flush=True)
    t2 = time.time()
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
    train_dir_wall = time.time() - t2
    (out / "learning.json").write_text(json.dumps(log, indent=2))
    torch.save(
        {"state": models["mu_base"].state_dict(), "stats": stats, "kind": "mu_base", "seed": seed},
        out / "mu_base_seed0.pt",
    )
    torch.save(
        {"state": models["mu_dir"].state_dict(), "stats": stats, "kind": "mu_dir", "seed": seed},
        out / "mu_dir_seed0.pt",
    )

    rows = []
    for name, m in models.items():
        cos = [
            P.cosine(E.gradient(m, s, lo_t, hi_t), g)
            for s, g, ok in zip(test_starts, test_g, test_ok)
            if ok.mean() > 0.5
        ]
        theta = P.policy_train(m, train_starts[:12], lo_t, hi_t, updates=policy_updates)
        ret = P.policy_return(env, theta, test_starts, horizon=16)
        row = dict(
            seed=seed,
            model=name,
            cosine_median=float(np.median(cos)),
            cosine_mean=float(np.mean(cos)),
            cosine_n=len(cos),
            policy_return_mean=float(ret.mean()),
            returns=ret.tolist(),
            theta_norm=float(np.linalg.norm(theta)),
        )
        if name == "mu_base":
            row["validation"] = stat_base["validation"]
        if name == "mu_dir":
            row["validation"] = stat_dir["validation"]
        rows.append(row)
        print(
            f"  {name:8s} cos {row['cosine_median']:.4f} ret {row['policy_return_mean']:.8f}",
            flush=True,
        )

    archived = None
    if ARCH.exists():
        arch_rows = [
            r for r in json.loads(ARCH.read_text())["rows"] if r["seed"] == seed
        ]
        archived = {r["model"]: r for r in arch_rows}

    deltas = {}
    if archived:
        for row in rows:
            a = archived.get(row["model"])
            if not a:
                continue
            deltas[row["model"]] = {
                "cosine_median_delta": row["cosine_median"] - a["cosine_median"],
                "return_delta": row["policy_return_mean"] - a["policy_return_mean"],
                "archive_cosine_median": a["cosine_median"],
                "archive_return": a["policy_return_mean"],
            }

    payload = dict(
        rows=rows,
        deltas_vs_archive=deltas,
        s0=s0,
        label_wall_s=label_wall,
        train_mu_base_wall_s=train_base_wall,
        train_mu_dir_wall_s=train_dir_wall,
        wall_seconds=time.time() - t0,
        tolerances={"cosine_abs": 0.02, "return_abs": 2e-4},
        note="Stage-1 seed-0 reproduction on new GPU host; training on CPU FP32.",
    )
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    print("wrote", out, "wall", round(payload["wall_seconds"], 1), flush=True)
    print("deltas", json.dumps(deltas, indent=2), flush=True)


if __name__ == "__main__":
    main()
