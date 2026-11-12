#!/usr/bin/env python3
"""Stage 4: fixed Controller B, five paired seeds (frozen Stage 3.2 protocol).

Models: Mc, EAGM-Base, EAGM-Dir-Matched, EAGM-Hist, EAGM-Hist+Dir-Matched.
Does not overwrite stage3_1 or stage3_2 seed0 directories.
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

from controllers.partial_unknown import ControllerBConfig, PartialUnknownArmEnv
from models.eagm import EAGMBase, EAGMHistory, McWorld
from training.dataset import (
    collect_trajectories,
    fit_stats,
    flatten_split,
    make_split,
    pick_starts,
)
from training.gradients import sim_grad_masked
from training.paired_train import clone_model, train_mc, train_pair
from training.policy_opt import eval_physical, policy_optimize
from training.rollout import imagined_eagm, imagined_mc
from training.seeds import seed_everything

UPDATES = 3000
H_IMAG = 8
H_EVAL = 16
HIST = 8
SEEDS = [0, 1, 2, 3, 4]
DTYPE = torch.float32


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


def model_grad(model, start, theta, use_history, is_mc=False):
    th = torch.as_tensor(theta, dtype=DTYPE).clone().requires_grad_(True)
    J = imagined_mc(model, th, start, horizon=H_IMAG) if is_mc else imagined_eagm(
        model, th, start, horizon=H_IMAG, use_history=use_history
    )
    return torch.autograd.grad(J, th)[0].detach().numpy()


def run_seed(seed: int, out_dir: Path) -> dict:
    seed_everything(seed)
    cfg = ControllerBConfig(gain=1.05, extra_lag=1, rate_limit=0.045, smooth_beta=0.25)
    env = PartialUnknownArmEnv(cfg)
    t0 = time.time()
    episodes = collect_trajectories(
        env, n_episodes=24, horizon=200, seed=seed, hist_len=HIST, config_id="B_fixed_105"
    )
    split = make_split(24, seed=seed, n_train=12, n_val=6, n_test=6)
    train_rows = flatten_split(episodes, split.train_eps)
    test_rows = flatten_split(episodes, split.test_eps)
    stats = fit_stats(train_rows, env_lower=env.lower, env_upper=env.upper)
    dir_starts = pick_starts(train_rows, delay=None, max_per_episode=1, max_total=12)
    test_starts = pick_starts(test_rows, delay=None, max_per_episode=2, max_total=12)
    rng_th = np.random.default_rng(seed + 101)
    thetas = [np.zeros(28)] + [rng_th.normal(0, 0.3, 28).clip(-1, 1) for _ in range(11)]
    labels_tr, steps_tr = build_matched_labels(env, dir_starts, thetas)
    labels_te, steps_te = build_matched_labels(env, test_starts, thetas)
    tr, te = pack_rows(train_rows), pack_rows(test_rows)

    seed_everything(seed + 7)
    mc = McWorld(stats)
    train_mc(mc, tr, UPDATES, batch_seed=seed + 20)

    seed_everything(seed + 11)
    base = EAGMBase(stats)
    dir_m = clone_model(base)
    train_pair(
        base, dir_m, tr, updates=UPDATES, use_history=False, dir_labels=labels_tr, batch_seed=seed + 30
    )

    seed_everything(seed + 13)
    hist = EAGMHistory(stats, hist_len=HIST)
    hist_dir = clone_model(hist)
    train_pair(
        hist, hist_dir, tr, updates=UPDATES, use_history=True, dir_labels=labels_tr, batch_seed=seed + 40
    )

    models = {
        "mc": (mc, False, True),
        "eagm_base": (base, False, False),
        "eagm_dir_matched": (dir_m, False, False),
        "eagm_hist": (hist, True, False),
        "eagm_hist_dir_matched": (hist_dir, True, False),
    }
    train_pol = pick_starts(train_rows, delay=None, max_per_episode=1, max_total=12)
    th0 = eval_physical(env, np.zeros(28), test_starts, horizon=H_EVAL)
    rows_out = []
    summary = {}
    for name, (m, use_h, is_mc) in models.items():
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
        cos = []
        for lab in labels_te:
            gh = model_grad(m, lab["start"], lab["theta"], use_h, is_mc=is_mc)
            cos.append(cosine(gh, lab["g"]))
        pol = policy_optimize(m, train_pol, use_history=use_h, is_mc=is_mc, updates=50, seed=seed + 50)
        phys = eval_physical(env, pol["theta"], test_starts, horizon=H_EVAL)
        summary[name] = dict(
            dynamics_nmse=nmse,
            cmd_mae=cmd,
            cosine_median=float(np.median(cos)),
            policy_return_mean=phys["mean"],
            delta_vs_theta0=phys["mean"] - th0["mean"],
            theta_l2_delta=float(np.linalg.norm(pol["theta"] - pol["theta0"])),
        )
        for ps in phys["per_start"]:
            rows_out.append(
                dict(
                    seed=seed,
                    model=name,
                    episode_id=ps["episode_id"],
                    frame=ps["frame"],
                    delay=ps["delay"],
                    return_=ps["return_"],
                )
            )
        torch.save({"state": m.state_dict(), "stats": stats, "kind": name, "seed": seed}, out_dir / f"{name}.pt")
        print(
            f"  seed{seed} {name:22s} cos {summary[name]['cosine_median']:.3f} "
            f"ret {summary[name]['policy_return_mean']:.6f}",
            flush=True,
        )
    payload = dict(
        seed=seed,
        split=dict(train=split.train_eps, val=split.val_eps, test=split.test_eps),
        dir_steps=steps_tr + steps_te,
        theta0_return=th0["mean"],
        results=summary,
        wall_seconds=time.time() - t0,
    )
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    with (out_dir / "per_start.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    return payload


def paired_ci(diffs, n_boot=20000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=np.float64)
    boots = [diffs[rng.integers(0, len(diffs), size=len(diffs))].mean() for _ in range(n_boot)]
    return dict(
        mean=float(diffs.mean()),
        median=float(np.median(diffs)),
        ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        n=len(diffs),
        n_positive=int((diffs > 0).sum()),
    )


def main():
    root = EAGM_ROOT / "runs" / "stage4_fixedB_five_seed"
    root.mkdir(parents=True, exist_ok=False)
    all_seed = []
    t0 = time.time()
    for seed in SEEDS:
        print(f"==== seed {seed} ====", flush=True)
        sd = root / f"seed{seed}"
        sd.mkdir(exist_ok=False)
        all_seed.append(run_seed(seed, sd))
        (root / "partial.json").write_text(json.dumps(all_seed, indent=2, default=str))

    # Seed-level paired diffs vs Mc
    names = ["eagm_base", "eagm_dir_matched", "eagm_hist", "eagm_hist_dir_matched"]
    stats = {}
    for name in names:
        dret = [s["results"][name]["policy_return_mean"] - s["results"]["mc"]["policy_return_mean"] for s in all_seed]
        dcos = [s["results"][name]["cosine_median"] - s["results"]["mc"]["cosine_median"] for s in all_seed]
        stats[name] = dict(return_vs_mc=paired_ci(dret), cosine_vs_mc=paired_ci(dcos))
    # Also Dir vs Base
    stats["eagm_dir_matched_vs_base"] = paired_ci(
        [
            s["results"]["eagm_dir_matched"]["policy_return_mean"] - s["results"]["eagm_base"]["policy_return_mean"]
            for s in all_seed
        ]
    )
    stats["eagm_hist_dir_vs_hist"] = paired_ci(
        [
            s["results"]["eagm_hist_dir_matched"]["policy_return_mean"]
            - s["results"]["eagm_hist"]["policy_return_mean"]
            for s in all_seed
        ]
    )

    # Trajectory-level: pool per-start returns nested in seeds (report means only + caution)
    per_start = []
    for seed in SEEDS:
        with (root / f"seed{seed}" / "per_start.csv").open() as f:
            per_start.extend(list(csv.DictReader(f)))

    summary = dict(
        seeds=SEEDS,
        updates=UPDATES,
        seed_results=all_seed,
        paired_seed_stats=stats,
        n_per_start_rows=len(per_start),
        wall_seconds=time.time() - t0,
        note="Seed-level paired bootstrap (n=5). Do not treat pooled starts as independent.",
    )
    (root / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(stats, indent=2), flush=True)
    print("wrote", root, "wall", round(summary["wall_seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
