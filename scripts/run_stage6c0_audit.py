#!/usr/bin/env python3
"""Stage 6C.0 — protocol audit + smoke (no large training).

Checks:
  1) Feasible FD boundary modes + step accounting
  2) Projection / saturation diagnostics on a frozen 6B.1 Dir checkpoint
  3) Paired Base/Dir smoke (tiny updates) still shares init hash
  4) Path probe extraction returns 6 checkpoints
Writes runs/stage6c0_audit/ and appends notes to protocol.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

EAGM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EAGM_ROOT))
sys.path.insert(0, str(EAGM_ROOT / "vendor" / "legacy_sim"))

from controllers.partial_unknown import ControllerBConfig, PartialUnknownArmEnv
from models.eagm import EAGMBase
from training.dataset import fit_stats, flatten_split, make_split, pick_starts
from training.gradients import sim_grad_masked_detailed
from training.paired_train import clone_model, train_pair
from training.policy_opt import STAGE6C_PATH_FRACS, policy_optimize
from training.seeds import seed_everything
from training.stage6c_common import (
    PARENT_COMMIT,
    build_labels_at_thetas,
    config_hash,
    path_probe_thetas_from_pilot,
    random_probe_thetas,
    summarize_policy_curve,
)

_spec = importlib.util.spec_from_file_location("stage5e_ood", EAGM_ROOT / "scripts" / "run_stage5e_ood.py")
_stage5e = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_stage5e)
TRAIN_RANGES = _stage5e.TRAIN_RANGES
collect_randomized = _stage5e.collect_randomized
pack_rows = _stage5e.pack_rows

OUT = EAGM_ROOT / "runs" / "stage6c0_audit"
CKPT_DIR = EAGM_ROOT / "runs" / "stage6b1_confirm" / "seed10"


def param_digest(model) -> str:
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    report = dict(stage="6C.0", parent_commit=PARENT_COMMIT, checks={})

    # --- 1) Feasible FD on real env near bounds ---
    seed_everything(0)
    env = PartialUnknownArmEnv(ControllerBConfig(
        episode_randomize=True, **{k: TRAIN_RANGES[k] for k in TRAIN_RANGES},
        gain=1.0, extra_lag=1, rate_limit=0.05, smooth_beta=0.2,
    ))
    episodes = collect_randomized(env, 8, 200, 0)
    split = make_split(8, seed=0, n_train=6, n_val=1, n_test=1)
    train_rows = flatten_split(episodes, split.train_eps)
    starts = pick_starts(train_rows, delay=None, max_per_episode=1, max_total=2)
    th_hi = np.ones(28)
    g, mask, steps, meta = sim_grad_masked_detailed(
        env, starts[0], th_hi, feasible=True, horizon=4, scales=(0.5, 1.0, 2.0)
    )
    report["checks"]["feasible_fd_upper"] = dict(
        n_boundary=meta["n_boundary"],
        n_onesided=meta["n_onesided"],
        all_backward=all(m == "backward" for m in meta["fd_mode"]),
        control_steps=steps,
        g_finite=bool(np.isfinite(g).all()),
        stable_mask_sum=float(mask.sum()),
        note="stability_mask_separate_from_boundary",
    )
    assert meta["n_onesided"] == 28
    assert np.isfinite(g).all()

    # --- 2) Projection diagnostics on frozen Dir ---
    stats = fit_stats(train_rows, env_lower=env.lower, env_upper=env.upper)
    # Prefer 6B.1 ckpt if present; else train tiny Dir
    train_pol = pick_starts(train_rows, delay=None, max_per_episode=1, max_total=4)
    if (CKPT_DIR / "eagm_dir_matched.pt").exists():
        # Need stats matching seed10 data — rebuild seed10 collect briefly
        seed_everything(10)
        env10 = PartialUnknownArmEnv(ControllerBConfig(
            episode_randomize=True, **{k: TRAIN_RANGES[k] for k in TRAIN_RANGES},
            gain=1.0, extra_lag=1, rate_limit=0.05, smooth_beta=0.2,
        ))
        eps10 = collect_randomized(env10, 36, 200, 10)
        sp10 = make_split(36, seed=10, n_train=24, n_val=6, n_test=6)
        tr10 = flatten_split(eps10, sp10.train_eps)
        stats10 = fit_stats(tr10, env_lower=env10.lower, env_upper=env10.upper)
        train_pol = pick_starts(tr10, delay=None, max_per_episode=1, max_total=8)
        blob = torch.load(CKPT_DIR / "eagm_dir_matched.pt", map_location="cpu", weights_only=False)
        model = EAGMBase(stats10)
        model.load_state_dict(blob["state"])
        model.eval()
        ckpt_note = "stage6b1_seed10_eagm_dir_matched"
        env_eval = env10
    else:
        model = EAGMBase(stats)
        ckpt_note = "fresh_untrained_base"
        env_eval = env

    def phys_fn(theta, u):
        from training.policy_opt import eval_physical
        ev = eval_physical(env_eval, theta, train_pol[:4], horizon=8)
        return dict(mean_return=ev["mean"], action_clip=ev["mean_action_clip_frac"])

    pol = policy_optimize(
        model, train_pol, use_history=False, updates=50, lr=0.03, seed=10 + 50,
        checkpoint_fracs=STAGE6C_PATH_FRACS, physical_eval_fn=phys_fn,
    )
    report["checks"]["policy_diagnostics"] = dict(
        ckpt=ckpt_note,
        **summarize_policy_curve(pol),
        physical_curve=pol["physical_curve"],
        n_checkpoints=len([k for k in pol["checkpoints"] if k.startswith("u")]),
    )

    # --- 3) Paired init smoke ---
    seed_everything(99)
    base = EAGMBase(stats)
    dir_m = clone_model(base)
    assert param_digest(base) == param_digest(dir_m)
    tr = pack_rows(train_rows)
    thetas = random_probe_thetas(4, seed=99)
    labels, steps_lab = build_labels_at_thetas(env, starts[:2], thetas, feasible=True, horizon=4)
    train_pair(
        base, dir_m, tr, updates=20, use_history=False, dir_labels=labels,
        lam_u=1.0, lam_d=0.1, batch_size=32, batch_seed=123, lr=1e-3, h_imag=4,
    )
    report["checks"]["paired_smoke"] = dict(
        init_digest_match=True,
        dir_label_steps=steps_lab,
        n_labels=len(labels),
        base_digest_after=param_digest(base),
        dir_digest_after=param_digest(dir_m),
        diverged_after_train=param_digest(base) != param_digest(dir_m),
    )

    # --- 4) Path probes ---
    path_th, pilot = path_probe_thetas_from_pilot(
        model, train_pol, use_history=False, updates=50, lr=0.03, seed=42,
        fracs=STAGE6C_PATH_FRACS,
    )
    report["checks"]["path_probes"] = dict(
        n_thetas=len(path_th),
        fracs=list(STAGE6C_PATH_FRACS),
        norms=[float(np.linalg.norm(t)) for t in path_th],
        sat=[float(np.mean(np.abs(t) >= 1.0 - 1e-8)) for t in path_th],
        pilot_final_sat=pilot["final_theta_param_sat_frac"],
    )
    assert len(path_th) == len(STAGE6C_PATH_FRACS)

    report["config_hash"] = config_hash(report["checks"])
    report["wall_seconds"] = time.time() - t0
    report["status"] = "PASS"
    (OUT / "audit.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(f"Wrote {OUT / 'audit.json'}", flush=True)


if __name__ == "__main__":
    main()
