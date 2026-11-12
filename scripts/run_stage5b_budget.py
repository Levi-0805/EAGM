#!/usr/bin/env python3
"""Stage 5B: same simulation-budget comparison.

A. EAGM-Base, original dynamics budget (reuse Stage 5A metrics when present)
B. EAGM-Base + extra dynamics trajectories equal to Dir train FD steps
C. EAGM-Dir-Matched (original dynamics + direction labels)

Same test set per seed as Stage 5A (deterministic recollect with seed).
Test FD diagnostics listed separately; not in train budget.
"""
from __future__ import annotations

import csv
import json
import math
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
from training.dataset import (
    collect_trajectories,
    fit_stats,
    flatten_split,
    make_split,
    pick_starts,
)
from training.gradients import sim_grad_masked
from training.paired_train import clone_model, train_pair
from training.policy_opt import eval_physical, policy_optimize
from training.rollout import imagined_eagm
from training.seeds import seed_everything

FROZEN = json.loads((EAGM_ROOT / "configs" / "stage5_frozen.json").read_text())
UPDATES = int(FROZEN["world_model_updates"])
H_IMAG = int(FROZEN["imagined_horizon"])
H_EVAL = int(FROZEN["eval_horizon"])
HIST = int(FROZEN["hist_len"])
SEEDS = [5, 6, 7, 8, 9]
DTYPE = torch.float32
# 12 starts * 28 dims * 3 scales * 2 * H_IMAG
DIR_TRAIN_STEPS_EXPECTED = 12 * 28 * 3 * 2 * H_IMAG  # 16128
EXTRA_EPISODES = int(math.ceil(DIR_TRAIN_STEPS_EXPECTED / 200))


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def pack_rows(rows):
    keys = ["x", "y", "a", "c", "c_obs", "u_exec", "x_hist", "a_hist"]
    return {k: torch.as_tensor(np.stack([r[k] for r in rows]), dtype=DTYPE) for k in keys}


def build_matched_labels(env, starts, thetas):
    labels, steps = [], 0
    for i, s in enumerate(starts):
        th = np.asarray(thetas[i % len(thetas)], dtype=np.float64)
        g, mask, st = sim_grad_masked(env, s, th, horizon=H_IMAG)
        steps += st
        s0 = float(np.median(np.abs(g[mask > 0.5]))) if (mask > 0.5).any() else float(np.median(np.abs(g))) or 1.0
        labels.append(dict(start=s, theta=th.copy(), g=g, mask=mask, s0=s0 or 1.0))
    return labels, steps


def model_grad(model, start, theta):
    th = torch.as_tensor(theta, dtype=DTYPE).clone().requires_grad_(True)
    J = imagined_eagm(model, th, start, horizon=H_IMAG, use_history=False)
    return torch.autograd.grad(J, th)[0].detach().numpy()


def eval_model(model, te, labels_te, train_pol, env, test_starts, seed):
    with torch.no_grad():
        y, u = model(te["x"], te["c_obs"], te["a"])
        nmse = (((y - te["y"]) / model.ystd) ** 2).mean().item()
        cmd = (u - te["u_exec"]).abs().mean().item()
    cos = [cosine(model_grad(model, lab["start"], lab["theta"]), lab["g"]) for lab in labels_te]
    pol = policy_optimize(
        model, train_pol, use_history=False, is_mc=False,
        updates=int(FROZEN["policy_updates"]), lr=float(FROZEN["lr_policy"]), seed=seed + 50,
    )
    th0 = eval_physical(env, np.zeros(28), test_starts, horizon=H_EVAL)
    phys = eval_physical(env, pol["theta"], test_starts, horizon=H_EVAL)
    return dict(
        dynamics_nmse=nmse,
        cmd_mae=cmd,
        cosine_median=float(np.median(cos)),
        policy_return_mean=phys["mean"],
        delta_vs_theta0=phys["mean"] - th0["mean"],
        theta_l2_delta=float(np.linalg.norm(pol["theta"] - pol["theta0"])),
        per_start=phys["per_start"],
        theta0_return=th0["mean"],
    )


def run_seed(seed: int, out_dir: Path) -> dict:
    seed_everything(seed)
    cfg = ControllerBConfig(**{k: FROZEN["controller_b"][k] for k in (
        "gain", "extra_lag", "rate_limit", "smooth_beta", "episode_randomize"
    )})
    env = PartialUnknownArmEnv(cfg)
    t0 = time.time()

    episodes = collect_trajectories(
        env, n_episodes=24, horizon=200, seed=seed, hist_len=HIST, config_id="B_fixed_105"
    )
    split = make_split(24, seed=seed, n_train=12, n_val=6, n_test=6)
    train_rows = flatten_split(episodes, split.train_eps)
    test_rows = flatten_split(episodes, split.test_eps)

    # Extra dynamics for budget-matched Base (train-only, separate seed offset)
    extra_eps = collect_trajectories(
        env, n_episodes=EXTRA_EPISODES, horizon=200, seed=seed + 1000,
        hist_len=HIST, config_id="B_fixed_105_extra_dyn",
    )
    extra_steps = sum(ep["n_steps"] for ep in extra_eps)
    train_rows_extra = list(train_rows)
    for ep in extra_eps:
        ep["episode_id"] = 10_000 + ep["episode_id"]
        for r in ep["rows"]:
            r["episode_id"] = ep["episode_id"]
            r["start"]["episode_id"] = ep["episode_id"]
        train_rows_extra.extend(ep["rows"])

    stats = fit_stats(train_rows, env_lower=env.lower, env_upper=env.upper)
    stats_extra = fit_stats(train_rows_extra, env_lower=env.lower, env_upper=env.upper)

    dir_starts = pick_starts(train_rows, delay=None, max_per_episode=1, max_total=12)
    test_starts = pick_starts(test_rows, delay=None, max_per_episode=2, max_total=12)
    rng_th = np.random.default_rng(seed + 101)
    thetas = [np.zeros(28)] + [rng_th.normal(0, 0.3, 28).clip(-1, 1) for _ in range(11)]
    labels_tr, steps_tr = build_matched_labels(env, dir_starts, thetas)
    labels_te, steps_te = build_matched_labels(env, test_starts, thetas)

    tr = pack_rows(train_rows)
    tr_extra = pack_rows(train_rows_extra)
    te = pack_rows(test_rows)
    train_pol = pick_starts(train_rows, delay=None, max_per_episode=1, max_total=12)

    # A: Base original
    seed_everything(seed + 11)
    base = EAGMBase(stats)
    dir_m = clone_model(base)
    train_pair(
        base, dir_m, tr, updates=UPDATES, use_history=False, dir_labels=labels_tr,
        lam_u=float(FROZEN["lam_u"]), lam_d=float(FROZEN["lam_d"]),
        batch_size=int(FROZEN["batch_size"]), batch_seed=seed + 30, lr=float(FROZEN["lr_model"]),
    )

    # B: Base with extra dynamics (no dir); same init seed family
    seed_everything(seed + 11)
    base_extra = EAGMBase(stats_extra)
    train_pair(
        base_extra, None, tr_extra, updates=UPDATES, use_history=False, dir_labels=None,
        lam_u=float(FROZEN["lam_u"]), lam_d=float(FROZEN["lam_d"]),
        batch_size=int(FROZEN["batch_size"]), batch_seed=seed + 30, lr=float(FROZEN["lr_model"]),
    )

    methods = {
        "A_base_original": base,
        "B_base_budget_matched": base_extra,
        "C_dir_matched": dir_m,
    }
    results = {}
    rows_out = []
    for name, m in methods.items():
        t_m = time.time()
        ev = eval_model(m, te, labels_te, train_pol, env, test_starts, seed)
        wall_m = time.time() - t_m
        results[name] = {k: v for k, v in ev.items() if k != "per_start"}
        results[name]["eval_wall_seconds"] = wall_m
        for ps in ev["per_start"]:
            rows_out.append(dict(seed=seed, method=name, episode_id=ps["episode_id"],
                                 frame=ps["frame"], return_=ps["return_"]))
        torch.save({"state": m.state_dict(), "kind": name, "seed": seed}, out_dir / f"{name}.pt")
        print(f"  seed{seed} {name:24s} cos {results[name]['cosine_median']:.3f} "
              f"ret {results[name]['policy_return_mean']:.6f}", flush=True)

    budget = dict(
        dynamics_primary_steps=24 * 200,
        dynamics_extra_steps=extra_steps,
        extra_episodes=EXTRA_EPISODES,
        dir_label_train_steps=steps_tr,
        dir_label_test_diagnostic_steps=steps_te,
        A_train_budget=24 * 200,
        B_train_budget=24 * 200 + extra_steps,
        C_train_budget=24 * 200 + steps_tr,
        budget_match_note=(
            f"B extra dynamics steps={extra_steps} vs C dir train steps={steps_tr}; "
            f"target was {DIR_TRAIN_STEPS_EXPECTED}."
        ),
    )
    payload = dict(
        seed=seed,
        stage="5B",
        sim_budget=budget,
        results=results,
        contrasts=dict(
            C_minus_A=results["C_dir_matched"]["policy_return_mean"] - results["A_base_original"]["policy_return_mean"],
            C_minus_B=results["C_dir_matched"]["policy_return_mean"] - results["B_base_budget_matched"]["policy_return_mean"],
            B_minus_A=results["B_base_budget_matched"]["policy_return_mean"] - results["A_base_original"]["policy_return_mean"],
        ),
        wall_seconds=time.time() - t0,
    )
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    with (out_dir / "per_start.csv").open("w", newline="") as f:
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
    root = EAGM_ROOT / "runs" / "stage5b_budget"
    root.mkdir(parents=True, exist_ok=False)
    all_seed = []
    t0 = time.time()
    for seed in SEEDS:
        print(f"==== Stage5B budget seed {seed} ====", flush=True)
        sd = root / f"seed{seed}"
        sd.mkdir(exist_ok=False)
        all_seed.append(run_seed(seed, sd))
        (root / "partial.json").write_text(json.dumps(all_seed, indent=2, default=str))

    c_minus_a = [s["contrasts"]["C_minus_A"] for s in all_seed]
    c_minus_b = [s["contrasts"]["C_minus_B"] for s in all_seed]
    b_minus_a = [s["contrasts"]["B_minus_A"] for s in all_seed]
    summary = dict(
        stage="5B",
        seeds=SEEDS,
        extra_episodes=EXTRA_EPISODES,
        dir_train_steps_expected=DIR_TRAIN_STEPS_EXPECTED,
        paired=dict(
            C_minus_A=paired_ci(c_minus_a),
            C_minus_B=paired_ci(c_minus_b),
            B_minus_A=paired_ci(b_minus_a),
        ),
        per_seed=[{"seed": s["seed"], **s["contrasts"],
                   "A": s["results"]["A_base_original"]["policy_return_mean"],
                   "B": s["results"]["B_base_budget_matched"]["policy_return_mean"],
                   "C": s["results"]["C_dir_matched"]["policy_return_mean"],
                   "budget": s["sim_budget"]} for s in all_seed],
        seed_results=all_seed,
        wall_seconds=time.time() - t0,
        interpretation_rule=(
            "If C > A but C <= B, Dir's gain may be explained by extra sim interactions. "
            "Only if C > B under matched train budget does Dir show extra data-utilization value."
        ),
    )
    (root / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary["paired"], indent=2), flush=True)
    print("wrote", root, "wall", round(summary["wall_seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
