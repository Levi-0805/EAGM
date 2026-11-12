#!/usr/bin/env python3
"""Stage 5D: gradient quality along the actual policy-optimization path.

For Base and Dir-Matched on confirmatory seeds, at θ∈{0, mid, final}:
  - model ∇θ J vs independent sim FD ∇θ J
  - balanced directional accuracy / directional error
  - true return change for the actual update direction

Answers whether Dir improves gradients where policy opt actually walks.
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
from training.rollout import imagined_eagm, physical_return
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


def balanced_dir_acc(g_hat, g_true, mask=None):
    g_hat = np.asarray(g_hat)
    g_true = np.asarray(g_true)
    if mask is None:
        mask = np.ones_like(g_true)
    m = mask > 0.5
    if not m.any():
        return 0.0
    # sign agreement on non-near-zero true components
    nz = m & (np.abs(g_true) > 1e-12)
    if not nz.any():
        return 0.0
    return float((np.sign(g_hat[nz]) == np.sign(g_true[nz])).mean())


def dir_error(g_hat, g_true, mask=None):
    g_hat = np.asarray(g_hat)
    g_true = np.asarray(g_true)
    if mask is None:
        mask = np.ones_like(g_true)
    m = mask > 0.5
    if not m.any():
        return float("nan")
    # normalized L2 on unit vectors
    a = g_hat[m]
    b = g_true[m]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.linalg.norm(a / na - b / nb))


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


def model_grad_at(model, starts, theta):
    th = torch.as_tensor(theta, dtype=DTYPE).clone().requires_grad_(True)
    Js = [imagined_eagm(model, th, s, horizon=H_IMAG, use_history=False) for s in starts]
    J = torch.stack(Js).mean()
    return torch.autograd.grad(J, th)[0].detach().numpy()


def sim_grad_at(env, starts, theta, max_starts=4):
    """FD gradient of mean return over a few starts (costly)."""
    use = starts[:max_starts]
    acc = None
    steps = 0
    mask_acc = None
    for s in use:
        g, mask, st = sim_grad_masked(env, s, theta, horizon=H_IMAG)
        steps += st
        acc = g if acc is None else acc + g
        mask_acc = mask if mask_acc is None else np.minimum(mask_acc, mask)
    return acc / len(use), mask_acc, steps


def mean_physical(env, theta, starts, horizon):
    rets = [physical_return(env, theta, s, horizon=horizon)[0] for s in starts]
    return float(np.mean(rets))


def analyze_path(model, name, env, train_pol, test_starts, seed, out_dir):
    pol = policy_optimize(
        model, train_pol, use_history=False, is_mc=False,
        updates=int(FROZEN["policy_updates"]), lr=float(FROZEN["lr_policy"]), seed=seed + 50,
    )
    checkpoints = {
        "theta0": np.asarray(pol["theta0"], dtype=np.float64),
        "theta_mid": np.asarray(pol["curve"][24]["theta"], dtype=np.float64),
        "theta_final": np.asarray(pol["theta"], dtype=np.float64),
    }
    path_diag = {}
    fd_steps = 0
    for label, th in checkpoints.items():
        g_model = model_grad_at(model, train_pol, th)
        g_sim, mask, st = sim_grad_at(env, train_pol, th, max_starts=4)
        fd_steps += st
        sat = float((np.abs(th) >= 1.0 - 1e-8).mean())
        ret = mean_physical(env, th, test_starts, H_EVAL)
        path_diag[label] = dict(
            cosine=cosine(g_model, g_sim),
            balanced_dir_acc=balanced_dir_acc(g_model, g_sim, mask),
            directional_error=dir_error(g_model, g_sim, mask),
            model_grad_norm=float(np.linalg.norm(g_model)),
            sim_grad_norm=float(np.linalg.norm(g_sim)),
            theta_norm=float(np.linalg.norm(th)),
            sat_frac=sat,
            physical_return_on_test=ret,
            theta=th.tolist(),
        )
        print(f"    {name} {label}: cos={path_diag[label]['cosine']:.3f} "
              f"bacc={path_diag[label]['balanced_dir_acc']:.3f} ret={ret:.6f}", flush=True)

    # Actual update direction: from θ0 to θ_final, measure true return change on test
    th0 = checkpoints["theta0"]
    thf = checkpoints["theta_final"]
    delta = thf - th0
    ret0 = mean_physical(env, th0, test_starts, H_EVAL)
    retf = mean_physical(env, thf, test_starts, H_EVAL)
    # Also probe a small step along model grad at θ0
    g0 = model_grad_at(model, train_pol, th0)
    step = 0.03
    if np.linalg.norm(g0) > 0:
        th_probe = np.clip(th0 + step * g0 / (np.linalg.norm(g0) + 1e-12), -1, 1)
    else:
        th_probe = th0.copy()
    ret_probe = mean_physical(env, th_probe, test_starts, H_EVAL)

    summary = dict(
        model=name,
        path=path_diag,
        update_true_delta_return=retf - ret0,
        theta0_return=ret0,
        theta_final_return=retf,
        theta_l2_delta=float(np.linalg.norm(delta)),
        probe_along_model_grad_delta_return=ret_probe - ret0,
        fd_control_steps=fd_steps,
        curve_imagined=[dict(update=c["update"], imagined_return=c["imagined_return"],
                             grad_norm=c["grad_norm"], sat_frac=c["sat_frac"]) for c in pol["curve"]],
    )
    (out_dir / f"{name}_path.json").write_text(json.dumps(summary, indent=2))
    return summary


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
    tr = pack_rows(train_rows)
    train_pol = pick_starts(train_rows, delay=None, max_per_episode=1, max_total=12)

    seed_everything(seed + 11)
    base = EAGMBase(stats)
    dir_m = clone_model(base)
    train_pair(
        base, dir_m, tr, updates=UPDATES, use_history=False, dir_labels=labels_tr,
        lam_u=float(FROZEN["lam_u"]), lam_d=float(FROZEN["lam_d"]),
        batch_size=int(FROZEN["batch_size"]), batch_seed=seed + 30, lr=float(FROZEN["lr_model"]),
    )

    print(f"  seed{seed} path analysis Base...", flush=True)
    base_path = analyze_path(base, "eagm_base", env, train_pol, test_starts, seed, out_dir)
    print(f"  seed{seed} path analysis Dir...", flush=True)
    dir_path = analyze_path(dir_m, "eagm_dir_matched", env, train_pol, test_starts, seed, out_dir)

    # Probe-label cosine (training-style) for reference
    labels_te, steps_te = build_matched_labels(env, test_starts, thetas)

    def probe_cos(model):
        vals = []
        for lab in labels_te:
            th = torch.as_tensor(lab["theta"], dtype=DTYPE).clone().requires_grad_(True)
            J = imagined_eagm(model, th, lab["start"], horizon=H_IMAG, use_history=False)
            gh = torch.autograd.grad(J, th)[0].detach().numpy()
            vals.append(cosine(gh, lab["g"]))
        return float(np.median(vals))

    payload = dict(
        seed=seed,
        stage="5D",
        dir_label_train_steps=steps_tr,
        dir_label_test_diagnostic_steps=steps_te,
        probe_cosine_median=dict(base=probe_cos(base), dir=probe_cos(dir_m)),
        base_path=base_path,
        dir_path=dir_path,
        path_cosine_delta=dict(
            theta0=dir_path["path"]["theta0"]["cosine"] - base_path["path"]["theta0"]["cosine"],
            theta_mid=dir_path["path"]["theta_mid"]["cosine"] - base_path["path"]["theta_mid"]["cosine"],
            theta_final=dir_path["path"]["theta_final"]["cosine"] - base_path["path"]["theta_final"]["cosine"],
        ),
        update_return_delta=dict(
            base=base_path["update_true_delta_return"],
            dir=dir_path["update_true_delta_return"],
            dir_minus_base=dir_path["update_true_delta_return"] - base_path["update_true_delta_return"],
        ),
        wall_seconds=time.time() - t0,
    )
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
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
    root = EAGM_ROOT / "runs" / "stage5d_gradpath"
    root.mkdir(parents=True, exist_ok=False)
    all_seed = []
    t0 = time.time()
    for seed in SEEDS:
        print(f"==== Stage5D grad-path seed {seed} ====", flush=True)
        sd = root / f"seed{seed}"
        sd.mkdir(exist_ok=False)
        all_seed.append(run_seed(seed, sd))
        (root / "partial.json").write_text(json.dumps(all_seed, indent=2, default=str))

    summary = dict(
        stage="5D",
        seeds=SEEDS,
        paired=dict(
            path_cos_delta_theta0=paired_ci([s["path_cosine_delta"]["theta0"] for s in all_seed]),
            path_cos_delta_mid=paired_ci([s["path_cosine_delta"]["theta_mid"] for s in all_seed]),
            path_cos_delta_final=paired_ci([s["path_cosine_delta"]["theta_final"] for s in all_seed]),
            update_return_dir_minus_base=paired_ci([s["update_return_delta"]["dir_minus_base"] for s in all_seed]),
        ),
        per_seed=[{
            "seed": s["seed"],
            "path_cosine_delta": s["path_cosine_delta"],
            "update_return_delta": s["update_return_delta"],
            "probe_cosine": s["probe_cosine_median"],
            "base_path_cos": {k: s["base_path"]["path"][k]["cosine"] for k in ("theta0", "theta_mid", "theta_final")},
            "dir_path_cos": {k: s["dir_path"]["path"][k]["cosine"] for k in ("theta0", "theta_mid", "theta_final")},
        } for s in all_seed],
        seed_results=all_seed,
        wall_seconds=time.time() - t0,
        question=(
            "Does Dir improve gradients in the region visited by policy optimization, "
            "or only on fixed probe θ used for training labels?"
        ),
    )
    (root / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary["paired"], indent=2), flush=True)
    print("wrote", root, "wall", round(summary["wall_seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
