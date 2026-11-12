#!/usr/bin/env python3
"""Stage 5C: fair structured baselines vs EAGM-Base / Dir-Matched.

Baseline 1: ObservableMemoryMLP — same observables, no command bottleneck.
Baseline 2: AuxCommandMLP — aux u_exec supervision, no U0+Δ bottleneck.

Uses confirmatory seeds [5..9], frozen hypers, same collection protocol as 5A.
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
from models.baselines import AuxCommandMLP, ObservableMemoryMLP, count_params
from models.eagm import EAGMBase
from training.dataset import (
    collect_trajectories,
    fit_stats,
    flatten_split,
    make_split,
    pick_starts,
)
from training.gradients import sim_grad_masked
from training.paired_train import clone_model, train_obs_mlp, train_pair
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


def eval_model(model, te, labels_te, train_pol, env, test_starts, seed, has_cmd=True):
    with torch.no_grad():
        out = model(te["x"], te["c_obs"], te["a"])
        y, u = out[0], out[1]
        nmse = (((y - te["y"]) / model.ystd) ** 2).mean().item()
        cmd = (u - te["u_exec"]).abs().mean().item() if has_cmd else None
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
        n_params=count_params(model),
        per_start=phys["per_start"],
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
    stats = fit_stats(train_rows, env_lower=env.lower, env_upper=env.upper)
    dir_starts = pick_starts(train_rows, delay=None, max_per_episode=1, max_total=12)
    test_starts = pick_starts(test_rows, delay=None, max_per_episode=2, max_total=12)
    rng_th = np.random.default_rng(seed + 101)
    thetas = [np.zeros(28)] + [rng_th.normal(0, 0.3, 28).clip(-1, 1) for _ in range(11)]
    labels_tr, steps_tr = build_matched_labels(env, dir_starts, thetas)
    labels_te, steps_te = build_matched_labels(env, test_starts, thetas)
    tr, te = pack_rows(train_rows), pack_rows(test_rows)
    train_pol = pick_starts(train_rows, delay=None, max_per_episode=1, max_total=12)

    # EAGM Base / Dir (reference, same protocol as 5A)
    seed_everything(seed + 11)
    base = EAGMBase(stats)
    dir_m = clone_model(base)
    t_train = time.time()
    train_pair(
        base, dir_m, tr, updates=UPDATES, use_history=False, dir_labels=labels_tr,
        lam_u=float(FROZEN["lam_u"]), lam_d=float(FROZEN["lam_d"]),
        batch_size=int(FROZEN["batch_size"]), batch_seed=seed + 30, lr=float(FROZEN["lr_model"]),
    )
    wall_eagm = time.time() - t_train

    # Obs MLP
    seed_everything(seed + 11)
    obs = ObservableMemoryMLP(stats)
    t_train = time.time()
    train_obs_mlp(obs, tr, UPDATES, batch_seed=seed + 30, lr=float(FROZEN["lr_model"]))
    wall_obs = time.time() - t_train

    # Aux command MLP (uses train_pair Base path: dynamics + lam_u cmd)
    seed_everything(seed + 11)
    aux = AuxCommandMLP(stats)
    t_train = time.time()
    train_pair(
        aux, None, tr, updates=UPDATES, use_history=False, dir_labels=None,
        lam_u=float(FROZEN["lam_u"]), lam_d=float(FROZEN["lam_d"]),
        batch_size=int(FROZEN["batch_size"]), batch_seed=seed + 30, lr=float(FROZEN["lr_model"]),
    )
    wall_aux = time.time() - t_train

    models = {
        "eagm_base": (base, True),
        "eagm_dir_matched": (dir_m, True),
        "obs_memory_mlp": (obs, False),  # placeholder u; cmd_mae not meaningful
        "aux_command_mlp": (aux, True),
    }
    results = {}
    rows_out = []
    for name, (m, has_cmd) in models.items():
        ev = eval_model(m, te, labels_te, train_pol, env, test_starts, seed, has_cmd=has_cmd)
        results[name] = {k: v for k, v in ev.items() if k != "per_start"}
        for ps in ev["per_start"]:
            rows_out.append(dict(seed=seed, model=name, episode_id=ps["episode_id"],
                                 frame=ps["frame"], return_=ps["return_"]))
        torch.save({"state": m.state_dict(), "kind": name, "seed": seed, "n_params": count_params(m)},
                   out_dir / f"{name}.pt")
        print(f"  seed{seed} {name:20s} params {results[name]['n_params']} "
              f"cos {results[name]['cosine_median']:.3f} ret {results[name]['policy_return_mean']:.6f}",
              flush=True)

    results["eagm_base"]["train_wall_seconds"] = wall_eagm / 2.0  # approx share with dir
    results["eagm_dir_matched"]["train_wall_seconds"] = wall_eagm / 2.0
    results["obs_memory_mlp"]["train_wall_seconds"] = wall_obs
    results["aux_command_mlp"]["train_wall_seconds"] = wall_aux

    payload = dict(
        seed=seed,
        stage="5C",
        dir_steps_train=steps_tr,
        dir_steps_test_diagnostic=steps_te,
        results=results,
        contrasts=dict(
            base_minus_obs=results["eagm_base"]["policy_return_mean"] - results["obs_memory_mlp"]["policy_return_mean"],
            base_minus_aux=results["eagm_base"]["policy_return_mean"] - results["aux_command_mlp"]["policy_return_mean"],
            dir_minus_aux=results["eagm_dir_matched"]["policy_return_mean"] - results["aux_command_mlp"]["policy_return_mean"],
            dir_minus_base=results["eagm_dir_matched"]["policy_return_mean"] - results["eagm_base"]["policy_return_mean"],
            aux_minus_obs=results["aux_command_mlp"]["policy_return_mean"] - results["obs_memory_mlp"]["policy_return_mean"],
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
    root = EAGM_ROOT / "runs" / "stage5c_baselines"
    root.mkdir(parents=True, exist_ok=False)
    all_seed = []
    t0 = time.time()
    for seed in SEEDS:
        print(f"==== Stage5C baselines seed {seed} ====", flush=True)
        sd = root / f"seed{seed}"
        sd.mkdir(exist_ok=False)
        all_seed.append(run_seed(seed, sd))
        (root / "partial.json").write_text(json.dumps(all_seed, indent=2, default=str))

    keys = ["base_minus_obs", "base_minus_aux", "dir_minus_aux", "dir_minus_base", "aux_minus_obs"]
    paired = {k: paired_ci([s["contrasts"][k] for s in all_seed]) for k in keys}
    summary = dict(
        stage="5C",
        seeds=SEEDS,
        paired=paired,
        per_seed=[{"seed": s["seed"], **s["contrasts"],
                   "params": {k: s["results"][k]["n_params"] for k in s["results"]},
                   "returns": {k: s["results"][k]["policy_return_mean"] for k in s["results"]},
                   } for s in all_seed],
        seed_results=all_seed,
        wall_seconds=time.time() - t0,
        goal=(
            "Separate (1) structured U0+Δ bottleneck, (2) extra u_exec supervision, "
            "(3) gradient-consistency Dir loss."
        ),
    )
    (root / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(paired, indent=2), flush=True)
    print("wrote", root, "wall", round(summary["wall_seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
