#!/usr/bin/env python3
"""Stage 7C — single-seed Reach smoke (NOT confirmatory).

Matched Mc / Base / Dir-Random / Aux on Reach.
Reduced WM updates for the gate; do not treat as Stage 7D evidence.
"""
from __future__ import annotations

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
from models.baselines import AuxCommandMLP, count_params
from models.eagm import EAGMBase, McWorld
from tasks.reach import ReachConfig, collect_reach_episodes, fit_fk_surrogate
from training.dataset import fit_stats, make_split, pick_starts
from training.eval_metrics import gradient_quality
from training.paired_train import clone_model, train_mc, train_pair
from training.policy_opt import eval_physical_paired_vs_theta0, policy_optimize
from training.rollout import imagined_eagm, imagined_mc, set_reach_fk_surrogate
from training.seeds import seed_everything
from training.stage6c_common import build_labels_at_thetas, random_probe_thetas

OUT = EAGM_ROOT / "runs" / "stage7c_reach_smoke"
SEED = 20
UPDATES = 400  # smoke budget (full lock is 3000 for 7D)
H_IMAG, H_EVAL = 8, 16
N_LABELS = 8
DTYPE = torch.float32


def pack_rows(rows):
    keys = ["x", "y", "a", "c", "c_obs", "u_exec", "x_hist", "a_hist"]
    return {k: torch.as_tensor(np.stack([r[k] for r in rows]), dtype=DTYPE) for k in keys}


def model_grad(model, start, theta, is_mc=False):
    th = torch.as_tensor(theta, dtype=DTYPE).clone().requires_grad_(True)
    J = imagined_mc(model, th, start, horizon=H_IMAG) if is_mc else imagined_eagm(
        model, th, start, horizon=H_IMAG, use_history=False
    )
    return torch.autograd.grad(J, th)[0].detach().numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    t0 = time.time()
    env = PartialUnknownArmEnv(ControllerBConfig(
        episode_randomize=True,
        gain_range=(0.92, 1.08),
        extra_lag_choices=(0, 1, 2),
        rate_limit_range=(0.03, 0.08),
        smooth_beta_range=(0.0, 0.35),
        gain=1.0, extra_lag=1, rate_limit=0.05, smooth_beta=0.2,
    ))
    cfg = ReachConfig(goal_mode="random", traj_shape="hold", controller_b_randomize=True)
    print("fitting FK surrogate...", flush=True)
    fk = fit_fk_surrogate(env, n_samples=3000, seed=SEED, steps=600)
    set_reach_fk_surrogate(fk)
    print(f"FK RMSE={fk.fit_rmse:.4f} m", flush=True)

    print("collecting Reach episodes...", flush=True)
    episodes = collect_reach_episodes(env, cfg, n_episodes=18, horizon=120, seed=SEED)
    split = make_split(18, seed=SEED, n_train=12, n_val=3, n_test=3)
    train_rows = [r for ep in episodes if ep["episode_id"] in set(split.train_eps) for r in ep["rows"]]
    test_rows = [r for ep in episodes if ep["episode_id"] in set(split.test_eps) for r in ep["rows"]]
    stats = fit_stats(train_rows, env_lower=env.lower, env_upper=env.upper)
    tr = pack_rows(train_rows)

    dir_starts = pick_starts(train_rows, delay=None, max_per_episode=1, max_total=N_LABELS)
    train_pol = pick_starts(train_rows, delay=None, max_per_episode=1, max_total=8)
    test_starts = pick_starts(test_rows, delay=None, max_per_episode=1, max_total=8)

    thetas = random_probe_thetas(N_LABELS, seed=SEED + 101)
    labels, steps = build_labels_at_thetas(env, dir_starts, thetas, feasible=True, horizon=H_IMAG)

    seed_everything(SEED + 7)
    mc = McWorld(stats)
    train_mc(mc, tr, UPDATES, batch_seed=SEED + 20)

    seed_everything(SEED + 11)
    base_init = EAGMBase(stats)
    base = clone_model(base_init)
    dir_m = clone_model(base_init)
    train_pair(
        base, dir_m, tr, updates=UPDATES, use_history=False, dir_labels=labels,
        lam_u=1.0, lam_d=0.1, batch_size=128, batch_seed=SEED + 30,
    )

    seed_everything(SEED + 11)
    aux = AuxCommandMLP(stats)
    train_pair(
        aux, None, tr, updates=UPDATES, use_history=False, dir_labels=None,
        lam_u=1.0, lam_d=0.1, batch_size=128, batch_seed=SEED + 30,
    )

    results = {}
    for name, model, is_mc in [
        ("mc", mc, True),
        ("eagm_base", base, False),
        ("eagm_dir_random", dir_m, False),
        ("aux_command_mlp", aux, False),
    ]:
        cos = []
        for lab in labels:
            gh = model_grad(model, lab["start"], lab["theta"], is_mc=is_mc)
            cos.append(gradient_quality(gh, lab["g"], lab["mask"])["cosine_full"])
        pol = policy_optimize(
            model, train_pol, use_history=False, is_mc=is_mc,
            updates=50, lr=0.03, seed=SEED + 50,
        )
        paired = eval_physical_paired_vs_theta0(
            env, pol["theta"], test_starts, horizon=H_EVAL, theta0=pol["theta0"]
        )
        results[name] = dict(
            n_params=count_params(model),
            cosine_full_median=float(np.nanmedian(cos)),
            policy_return_mean=paired["policy_return_mean"],
            delta_vs_theta0=paired["delta_vs_theta0_mean"],
            fail_rate_vs_theta0=paired["fail_rate_vs_theta0"],
            theta_param_sat_frac=paired["theta_param_sat_frac"],
            mean_action_clip_frac=paired["mean_action_clip_frac"],
        )
        # attach true EE if present on last eval — recompute one start
        from training.rollout import physical_return_detailed
        det = physical_return_detailed(env, pol["theta"], test_starts[0], horizon=H_EVAL)
        results[name]["true_mujoco_ee_return_start0"] = det.get("true_mujoco_ee_return")
        print(
            f"  {name:18s} cos {results[name]['cosine_full_median']:.3f} "
            f"ret {results[name]['policy_return_mean']:.6f} "
            f"Δ {results[name]['delta_vs_theta0']:.6f}",
            flush=True,
        )

    contrasts = dict(
        dir_minus_base=results["eagm_dir_random"]["policy_return_mean"] - results["eagm_base"]["policy_return_mean"],
        dir_minus_mc=results["eagm_dir_random"]["policy_return_mean"] - results["mc"]["policy_return_mean"],
        dir_minus_aux=results["eagm_dir_random"]["policy_return_mean"] - results["aux_command_mlp"]["policy_return_mean"],
        base_minus_aux=results["eagm_base"]["policy_return_mean"] - results["aux_command_mlp"]["policy_return_mean"],
    )
    gate = dict(
        models_trained=True,
        dir_labels_ok=steps > 0 and len(labels) == N_LABELS,
        fk_rmse=float(fk.fit_rmse),
        physical_finite=all(np.isfinite(results[k]["policy_return_mean"]) for k in results),
        no_privileged_inputs=True,
        note="SMOKE only — not confirmatory. Human must approve before Stage 7D (10 seeds, 3000 updates).",
        recommend_proceed_to_7d=False,  # human decision
    )
    payload = dict(
        stage="7C_smoke",
        seed=SEED,
        updates_smoke=UPDATES,
        locked_full_updates=3000,
        fk_rmse=float(fk.fit_rmse),
        dir_label_steps=steps,
        reach_cfg=cfg.__dict__,
        results=results,
        contrasts=contrasts,
        gate=gate,
        wall_seconds=time.time() - t0,
    )
    (OUT / "results.json").write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(dict(contrasts=contrasts, gate=gate), indent=2))
    print(f"Wrote {OUT / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
