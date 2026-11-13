#!/usr/bin/env python3
"""Stage 6A: re-evaluate archived Stage 5E checkpoints with corrected metrics.

Does NOT overwrite Stage 5E raw runs. Writes corrected artifacts under
runs/stage6a_stage5e_corrected/. Reloads models from local .pt files and
recollects starts deterministically (same seeds as Stage 5E).
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

EAGM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EAGM_ROOT))
sys.path.insert(0, str(EAGM_ROOT / "vendor" / "legacy_sim"))

import importlib.util

from controllers.partial_unknown import ControllerBConfig, PartialUnknownArmEnv
from models.eagm import EAGMBase, EAGMHistory, McWorld
from training.dataset import fit_stats, flatten_split, make_split, pick_starts
from training.eval_metrics import stage5e_budget_breakdown
from training.policy_opt import eval_physical_paired_vs_theta0, policy_optimize
from training.seeds import seed_everything

_spec = importlib.util.spec_from_file_location(
    "stage5e_ood", EAGM_ROOT / "scripts" / "run_stage5e_ood.py"
)
_stage5e = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_stage5e)
EXTRAP_CONFIGS = _stage5e.EXTRAP_CONFIGS
INTERP_CONFIGS = _stage5e.INTERP_CONFIGS
TRAIN_RANGES = _stage5e.TRAIN_RANGES
collect_fixed_configs = _stage5e.collect_fixed_configs
collect_randomized = _stage5e.collect_randomized
pack_rows = _stage5e.pack_rows

FROZEN = json.loads((EAGM_ROOT / "configs" / "stage5_frozen.json").read_text())
H_EVAL = int(FROZEN["eval_horizon"])
HIST = int(FROZEN["hist_len"])
SEEDS = [5, 6, 7, 8, 9]
DTYPE = torch.float32


def load_model(kind: str, ckpt_path: Path, stats: dict):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if kind == "mc":
        m = McWorld(stats)
    elif kind in ("eagm_base", "eagm_dir_matched"):
        m = EAGMBase(stats)
    elif kind in ("eagm_hist", "eagm_hist_dir_matched"):
        m = EAGMHistory(stats, hist_len=HIST)
    else:
        raise ValueError(kind)
    m.load_state_dict(blob["state"])
    m.eval()
    return m


def reeval_seed(seed: int, src: Path, out: Path) -> dict:
    seed_everything(seed)
    t0 = time.time()
    train_env = PartialUnknownArmEnv(ControllerBConfig(
        episode_randomize=True, **{k: TRAIN_RANGES[k] for k in TRAIN_RANGES},
        gain=1.0, extra_lag=1, rate_limit=0.05, smooth_beta=0.2,
    ))
    episodes = collect_randomized(train_env, 36, 200, seed)
    split = make_split(36, seed=seed, n_train=24, n_val=6, n_test=6)
    train_rows = flatten_split(episodes, split.train_eps)
    stats = fit_stats(train_rows, env_lower=train_env.lower, env_upper=train_env.upper)
    train_pol = pick_starts(train_rows, delay=None, max_per_episode=1, max_total=12)

    interp_eps = collect_fixed_configs(INTERP_CONFIGS, 200, seed_base=10_000 + seed)
    extrap_eps = collect_fixed_configs(EXTRAP_CONFIGS, 200, seed_base=20_000 + seed)
    interp_rows = [r for ep in interp_eps for r in ep["rows"]]
    extrap_rows = [r for ep in extrap_eps for r in ep["rows"]]
    interp_starts = pick_starts(interp_rows, delay=None, max_per_episode=1, max_total=12)
    extrap_starts = pick_starts(extrap_rows, delay=None, max_per_episode=1, max_total=12)

    # Dynamics metrics on frozen pools
    te_i, te_e = pack_rows(interp_rows), pack_rows(extrap_rows)

    models_meta = {
        "mc": (False, True),
        "eagm_base": (False, False),
        "eagm_dir_matched": (False, False),
        "eagm_hist": (True, False),
        "eagm_hist_dir_matched": (True, False),
    }
    results = {"interpolation": {}, "extrapolation": {}}
    rows_out = []
    for name, (use_h, is_mc) in models_meta.items():
        ckpt = src / f"{name}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        m = load_model(name, ckpt, stats)
        for pool_name, te, starts in [
            ("interpolation", te_i, interp_starts),
            ("extrapolation", te_e, extrap_starts),
        ]:
            with torch.no_grad():
                if is_mc:
                    nmse = (((m(te["x"], te["c"], te["a"]) - te["y"]) / m.ystd) ** 2).mean().item()
                    cmd = None
                elif use_h:
                    y, u, _ = m(te["x"], te["c_obs"], te["a"], te["x_hist"], te["a_hist"])
                    nmse = (((y - te["y"]) / m.ystd) ** 2).mean().item()
                    cmd = (u - te["u_exec"]).abs().mean().item()
                else:
                    y, u = m(te["x"], te["c_obs"], te["a"])
                    nmse = (((y - te["y"]) / m.ystd) ** 2).mean().item()
                    cmd = (u - te["u_exec"]).abs().mean().item()
            pol = policy_optimize(
                m, train_pol, use_history=use_h, is_mc=is_mc,
                updates=int(FROZEN["policy_updates"]), lr=float(FROZEN["lr_policy"]), seed=seed + 50,
            )
            paired = eval_physical_paired_vs_theta0(
                train_env, pol["theta"], starts, horizon=H_EVAL, theta0=pol["theta0"]
            )
            results[pool_name][name] = dict(
                dynamics_nmse=nmse,
                cmd_mae=cmd,
                policy_return_mean=paired["policy_return_mean"],
                theta0_return_mean=paired["theta0_return_mean"],
                delta_vs_theta0=paired["delta_vs_theta0_mean"],
                fail_rate_vs_theta0=paired["fail_rate_vs_theta0"],
                theta_param_sat_frac=paired["theta_param_sat_frac"],
                mean_action_clip_frac=paired["mean_action_clip_frac"],
                per_config=paired["per_config"],
                final_theta_param_sat_frac=pol["final_theta_param_sat_frac"],
            )
            for ps in paired["per_start"]:
                rows_out.append(dict(
                    seed=seed, pool=pool_name, model=name,
                    episode_id=ps["episode_id"], config_id=ps.get("config_id"),
                    return_=ps["return_"], return_theta0=ps["return_theta0"],
                    delta_vs_theta0=ps["delta_vs_theta0"],
                ))
            print(
                f"  seed{seed} {pool_name[:5]} {name:22s} "
                f"ret {paired['policy_return_mean']:.6f} "
                f"ΔR {paired['delta_vs_theta0_mean']:.6g} "
                f"fail {paired['fail_rate_vs_theta0']:.2f} "
                f"θsat {paired['theta_param_sat_frac']:.2f} "
                f"aclip {paired['mean_action_clip_frac']:.2f}",
                flush=True,
            )

    # Recover FD step counts from original Stage5E JSON if present
    orig = json.loads((src / "results.json").read_text()) if (src / "results.json").exists() else {}
    steps_tr = int(orig.get("sim_budget", {}).get("dir_label_train_steps", 16128))
    steps_i = int(orig.get("sim_budget", {}).get("interp_fd_diagnostic", 16128))
    steps_e = int(orig.get("sim_budget", {}).get("extrap_fd_diagnostic", 16128))
    budget = stage5e_budget_breakdown(
        n_train_eps=24, n_val_eps=6, n_iid_hold_eps=6, horizon=200,
        dir_label_train_steps=steps_tr,
        ood_fd_diagnostic_steps=steps_i + steps_e,
    )
    budget.update(dict(interp_fd_diagnostic=steps_i, extrap_fd_diagnostic=steps_e))

    contrasts = dict(
        interp_hist_dir_minus_mc=results["interpolation"]["eagm_hist_dir_matched"]["policy_return_mean"]
        - results["interpolation"]["mc"]["policy_return_mean"],
        interp_dir_minus_base=results["interpolation"]["eagm_dir_matched"]["policy_return_mean"]
        - results["interpolation"]["eagm_base"]["policy_return_mean"],
        extrap_hist_dir_minus_mc=results["extrapolation"]["eagm_hist_dir_matched"]["policy_return_mean"]
        - results["extrapolation"]["mc"]["policy_return_mean"],
        extrap_dir_minus_base=results["extrapolation"]["eagm_dir_matched"]["policy_return_mean"]
        - results["extrapolation"]["eagm_base"]["policy_return_mean"],
    )
    payload = dict(
        seed=seed,
        stage="6A_corrected_stage5e",
        source_stage5e=str(src),
        sim_budget=budget,
        results=results,
        contrasts=contrasts,
        wall_seconds=time.time() - t0,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    with (out / "per_start.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    return payload


def paired_ci(diffs):
    rng = np.random.default_rng(0)
    diffs = np.asarray(diffs, dtype=np.float64)
    boots = [diffs[rng.integers(0, len(diffs), size=len(diffs))].mean() for _ in range(20000)]
    return dict(
        mean=float(diffs.mean()),
        median=float(np.median(diffs)),
        ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        n=int(len(diffs)),
        n_positive=int((diffs > 0).sum()),
        note="n=5 seed percentile bootstrap is unstable.",
    )


def main():
    root = EAGM_ROOT / "runs" / "stage6a_stage5e_corrected"
    root.mkdir(parents=True, exist_ok=False)
    all_seed = []
    t0 = time.time()
    for seed in SEEDS:
        print(f"==== Stage6A reeval Stage5E seed {seed} ====", flush=True)
        src = EAGM_ROOT / "runs" / "stage5e_ood" / f"seed{seed}"
        sd = root / f"seed{seed}"
        all_seed.append(reeval_seed(seed, src, sd))
        (root / "partial.json").write_text(json.dumps(all_seed, indent=2, default=str))

    summary = dict(
        stage="6A",
        purpose="Corrected Stage5E metrics (paired fail_rate, budget split, saturation)",
        seeds=SEEDS,
        paired={
            k: paired_ci([s["contrasts"][k] for s in all_seed])
            for k in all_seed[0]["contrasts"]
        },
        fail_rates={
            pool: {
                name: {
                    "mean_fail_rate": float(np.mean([
                        s["results"][pool][name]["fail_rate_vs_theta0"] for s in all_seed
                    ])),
                    "mean_theta_param_sat": float(np.mean([
                        s["results"][pool][name]["theta_param_sat_frac"] for s in all_seed
                    ])),
                    "mean_action_clip": float(np.mean([
                        s["results"][pool][name]["mean_action_clip_frac"] for s in all_seed
                    ])),
                }
                for name in all_seed[0]["results"][pool]
            }
            for pool in ("interpolation", "extrapolation")
        },
        budget_example=all_seed[0]["sim_budget"],
        seed_results=all_seed,
        wall_seconds=time.time() - t0,
    )
    (root / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary["paired"], indent=2), flush=True)
    print(json.dumps(summary["fail_rates"], indent=2), flush=True)
    print("wrote", root, "wall", round(summary["wall_seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
