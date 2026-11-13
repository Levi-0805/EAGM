#!/usr/bin/env python3
"""Stage 6C.1 — frozen WM policy learning-rate ablation.

Reloads Stage 6B.1 checkpoints (Mc / Base / Dir) for seeds 10–14.
Compares policy LRs 0.03 / 0.015 / 0.0075 with updates=50.
Does not retrain world models. Does not use OOD pools for LR selection.
"""
from __future__ import annotations

import csv
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
from training.dataset import fit_stats, flatten_split, make_split, pick_starts
from training.policy_opt import STAGE6C_PATH_FRACS, eval_physical_paired_vs_theta0, policy_optimize
from training.seeds import seed_everything
from training.stage6c_common import (
    PARENT_COMMIT,
    POLICY_LRS,
    POLICY_UPDATES,
    config_hash,
    load_eagm_checkpoint,
    paired_seed_stats,
    summarize_policy_curve,
)

_spec = importlib.util.spec_from_file_location("stage5e_ood", EAGM_ROOT / "scripts" / "run_stage5e_ood.py")
_stage5e = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_stage5e)
INTERP_CONFIGS = _stage5e.INTERP_CONFIGS
EXTRAP_CONFIGS = _stage5e.EXTRAP_CONFIGS
TRAIN_RANGES = _stage5e.TRAIN_RANGES
collect_fixed_configs = _stage5e.collect_fixed_configs
collect_randomized = _stage5e.collect_randomized

SRC_ROOT = EAGM_ROOT / "runs" / "stage6b1_confirm"
OUT = EAGM_ROOT / "runs" / "stage6c1_lr_ablation"
SEEDS = [10, 11, 12, 13, 14]
MODELS = {
    "mc": (False, True),
    "eagm_base": (False, False),
    "eagm_dir_matched": (False, False),
}
H_EVAL = 16


def run_seed(seed: int) -> dict:
    seed_everything(seed)
    t0 = time.time()
    src = SRC_ROOT / f"seed{seed}"
    out_dir = OUT / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_env = PartialUnknownArmEnv(ControllerBConfig(
        episode_randomize=True, **{k: TRAIN_RANGES[k] for k in TRAIN_RANGES},
        gain=1.0, extra_lag=1, rate_limit=0.05, smooth_beta=0.2,
    ))
    episodes = collect_randomized(train_env, 36, 200, seed)
    split = make_split(36, seed=seed, n_train=24, n_val=6, n_test=6)
    train_rows = flatten_split(episodes, split.train_eps)
    val_rows = flatten_split(episodes, split.val_eps)
    stats = fit_stats(train_rows, env_lower=train_env.lower, env_upper=train_env.upper)
    train_pol = pick_starts(train_rows, delay=None, max_per_episode=1, max_total=12)
    val_pol = pick_starts(val_rows, delay=None, max_per_episode=1, max_total=8)

    interp_eps = collect_fixed_configs(INTERP_CONFIGS, 200, seed_base=10_000 + seed, episodes_per_cfg=2)
    extrap_eps = collect_fixed_configs(EXTRAP_CONFIGS, 200, seed_base=20_000 + seed, episodes_per_cfg=2)
    interp_starts = pick_starts([r for ep in interp_eps for r in ep["rows"]], delay=None, max_per_episode=1, max_total=12)
    extrap_starts = pick_starts([r for ep in extrap_eps for r in ep["rows"]], delay=None, max_per_episode=1, max_total=12)

    rows_csv = []
    results = {}
    for name, (use_h, is_mc) in MODELS.items():
        ckpt = src / f"{name}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        model = load_eagm_checkpoint(name, ckpt, stats)
        results[name] = {}
        for lr in POLICY_LRS:
            def phys_fn(theta, u, _starts=val_pol):
                from training.policy_opt import eval_physical
                ev = eval_physical(train_env, theta, _starts, horizon=H_EVAL)
                return dict(mean_return=ev["mean"], action_clip=ev["mean_action_clip_frac"])

            pol = policy_optimize(
                model, train_pol, use_history=use_h, is_mc=is_mc,
                updates=POLICY_UPDATES, lr=lr, seed=seed + 50,
                checkpoint_fracs=STAGE6C_PATH_FRACS, physical_eval_fn=phys_fn,
            )
            paired_i = eval_physical_paired_vs_theta0(
                train_env, pol["theta"], interp_starts, horizon=H_EVAL, theta0=pol["theta0"]
            )
            paired_e = eval_physical_paired_vs_theta0(
                train_env, pol["theta"], extrap_starts, horizon=H_EVAL, theta0=pol["theta0"]
            )
            # val physical at final θ (for LR selection narrative — not OOD)
            from training.policy_opt import eval_physical
            val_ev = eval_physical(train_env, pol["theta"], val_pol, horizon=H_EVAL)

            entry = dict(
                lr=lr,
                val_return_mean=val_ev["mean"],
                val_action_clip=val_ev["mean_action_clip_frac"],
                interpolation=dict(
                    policy_return_mean=paired_i["policy_return_mean"],
                    delta_vs_theta0=paired_i["delta_vs_theta0_mean"],
                    fail_rate_vs_theta0=paired_i["fail_rate_vs_theta0"],
                    theta_param_sat_frac=paired_i["theta_param_sat_frac"],
                    mean_action_clip_frac=paired_i["mean_action_clip_frac"],
                ),
                extrapolation=dict(
                    policy_return_mean=paired_e["policy_return_mean"],
                    delta_vs_theta0=paired_e["delta_vs_theta0_mean"],
                    fail_rate_vs_theta0=paired_e["fail_rate_vs_theta0"],
                    theta_param_sat_frac=paired_e["theta_param_sat_frac"],
                    mean_action_clip_frac=paired_e["mean_action_clip_frac"],
                ),
                diagnostics=summarize_policy_curve(pol),
                physical_curve=pol["physical_curve"],
            )
            results[name][str(lr)] = entry
            print(
                f"  seed{seed} {name:20s} lr={lr:.4f} "
                f"val {val_ev['mean']:.6f} "
                f"interpΔ {paired_i['delta_vs_theta0_mean']:.6f} "
                f"sat {pol['final_theta_param_sat_frac']:.2f} "
                f"ρ {pol['mean_rho_proj']:.3f}",
                flush=True,
            )
            for pool, paired in [("interpolation", paired_i), ("extrapolation", paired_e)]:
                for ps in paired["per_start"]:
                    rows_csv.append(dict(
                        seed=seed, model=name, lr=lr, pool=pool,
                        episode_id=ps.get("episode_id"), config_id=ps.get("config_id"),
                        return_=ps["return_"], return_theta0=ps.get("return_theta0"),
                        delta_vs_theta0=ps.get("delta_vs_theta0"),
                    ))

    payload = dict(
        seed=seed,
        stage="6C.1",
        parent_commit=PARENT_COMMIT,
        source_ckpts=str(src),
        policy_updates=POLICY_UPDATES,
        lrs=POLICY_LRS,
        results=results,
        wall_seconds=time.time() - t0,
    )
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, default=str))
    with (out_dir / "per_start.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)
    return payload


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seed_results = []
    for seed in SEEDS:
        seed_results.append(run_seed(seed))
        (OUT / "partial.json").write_text(json.dumps(seed_results, indent=2, default=str))

    # Aggregate: for each model×lr, mean final sat / rho / val return; paired Δ vs lr=0.03
    summary = {"models": {}}
    for name in MODELS:
        summary["models"][name] = {}
        for lr in POLICY_LRS:
            vals = [s["results"][name][str(lr)]["val_return_mean"] for s in seed_results]
            sats = [s["results"][name][str(lr)]["diagnostics"]["final_theta_param_sat_frac"] for s in seed_results]
            rhos = [s["results"][name][str(lr)]["diagnostics"]["mean_rho_proj"] for s in seed_results]
            interp_d = [s["results"][name][str(lr)]["interpolation"]["delta_vs_theta0"] for s in seed_results]
            summary["models"][name][str(lr)] = dict(
                val_return=paired_seed_stats(vals),
                final_sat=paired_seed_stats(sats),
                mean_rho_proj=paired_seed_stats(rhos),
                interp_delta_vs_theta0=paired_seed_stats(interp_d),
            )
        # Δ(lr) − Δ(0.03) on interp return
        base_lr = 0.03
        for lr in POLICY_LRS:
            if lr == base_lr:
                continue
            diffs = [
                s["results"][name][str(lr)]["interpolation"]["policy_return_mean"]
                - s["results"][name][str(base_lr)]["interpolation"]["policy_return_mean"]
                for s in seed_results
            ]
            summary["models"][name][f"interp_return_lr{lr}_minus_lr{base_lr}"] = paired_seed_stats(diffs)

    payload = dict(
        stage="6C.1",
        parent_commit=PARENT_COMMIT,
        config_hash=config_hash(dict(seeds=SEEDS, lrs=POLICY_LRS, models=list(MODELS))),
        seeds=SEEDS,
        summary=summary,
        seed_results=seed_results,
        wall_seconds=time.time() - t0,
        decision_notes=(
            "If lower LR only delays saturation without improving val/interp return, "
            "do not claim LR solves saturation. High sat + low rho_proj + improving return "
            "⇒ saturation may be benign."
        ),
    )
    (OUT / "results.json").write_text(json.dumps(payload, indent=2, default=str))
    (OUT / "protocol.json").write_text(json.dumps(dict(
        parent_commit=PARENT_COMMIT,
        frozen_wm_from="runs/stage6b1_confirm",
        lrs=POLICY_LRS,
        policy_updates=POLICY_UPDATES,
        selection="val_return_only_not_ood",
    ), indent=2))
    print(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {OUT / 'results.json'} wall={payload['wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
