"""Shared Stage-5 helpers (frozen hyperparams from Stage 4)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

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

ROOT = Path(__file__).resolve().parents[1]
FROZEN = json.loads((ROOT / "configs" / "stage5_frozen.json").read_text())
UPDATES = int(FROZEN["world_model_updates"])
H_IMAG = int(FROZEN["imagined_horizon"])
H_EVAL = int(FROZEN["eval_horizon"])
HIST = int(FROZEN["hist_len"])
DTYPE = torch.float32
CFG = ControllerBConfig(**{k: FROZEN["controller_b"][k] for k in (
    "gain", "extra_lag", "rate_limit", "smooth_beta", "episode_randomize"
)})


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


def model_grad(model, start, theta, *, use_history=False, is_mc=False):
    th = torch.as_tensor(theta, dtype=DTYPE).clone().requires_grad_(True)
    if is_mc:
        J = imagined_mc(model, th, start, horizon=H_IMAG)
    else:
        J = imagined_eagm(model, th, start, horizon=H_IMAG, use_history=use_history)
    return torch.autograd.grad(J, th)[0].detach().numpy()


def paired_ci(diffs, n_boot=20000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=np.float64)
    boots = [diffs[rng.integers(0, len(diffs), size=len(diffs))].mean() for _ in range(n_boot)]
    return dict(
        mean=float(diffs.mean()),
        median=float(np.median(diffs)),
        ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        n=int(len(diffs)),
        n_positive=int((diffs > 0).sum()),
        note="n=5 seed percentile bootstrap is unstable; show all seeds.",
    )


def collect_budget(env, seed, n_episodes, horizon=200, config_id="B_fixed_105"):
    eps = collect_trajectories(
        env, n_episodes=n_episodes, horizon=horizon, seed=seed, hist_len=HIST, config_id=config_id
    )
    steps = sum(ep["n_steps"] for ep in eps)
    return eps, steps


def prepare_split_data(seed, *, n_episodes=24, extra_train_episodes=0):
    """Collect primary 24 eps + optional extra train-only dynamics eps for budget match."""
    seed_everything(seed)
    env = PartialUnknownArmEnv(CFG)
    episodes, steps_main = collect_budget(env, seed, n_episodes)
    split = make_split(n_episodes, seed=seed, n_train=12, n_val=6, n_test=6)
    train_rows = flatten_split(episodes, split.train_eps)
    val_rows = flatten_split(episodes, split.val_eps)
    test_rows = flatten_split(episodes, split.test_eps)
    extra_steps = 0
    if extra_train_episodes > 0:
        # Separate seed offset so extras never overlap primary episode IDs/content
        extra_eps, extra_steps = collect_budget(
            env, seed + 1000, extra_train_episodes, config_id="B_fixed_105_extra_dyn"
        )
        for ep in extra_eps:
            # Remap episode ids to avoid collision; keep as train-only
            ep["episode_id"] = 10_000 + ep["episode_id"]
            for r in ep["rows"]:
                r["episode_id"] = ep["episode_id"]
                r["start"]["episode_id"] = ep["episode_id"]
            train_rows.extend(ep["rows"])
    stats = fit_stats(train_rows, env_lower=env.lower, env_upper=env.upper)
    return dict(
        env=env,
        episodes=episodes,
        split=split,
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        stats=stats,
        steps_main=steps_main,
        steps_extra_dyn=extra_steps,
    )


def make_theta_bank(seed):
    rng = np.random.default_rng(seed + 101)
    return [np.zeros(28)] + [rng.normal(0, 0.3, 28).clip(-1, 1) for _ in range(11)]


def eval_suite(model, te, test_starts, labels_te, train_pol, env, *, use_history, is_mc, pol_seed):
    with torch.no_grad():
        if is_mc:
            nmse = (((model(te["x"], te["c"], te["a"]) - te["y"]) / model.ystd) ** 2).mean().item()
            cmd = None
        elif use_history:
            y, u, _ = model(te["x"], te["c_obs"], te["a"], te["x_hist"], te["a_hist"])
            nmse = (((y - te["y"]) / model.ystd) ** 2).mean().item()
            cmd = (u - te["u_exec"]).abs().mean().item()
        else:
            # EAGMBase returns (y,u); ObservableMLP returns y only
            out = model(te["x"], te["c_obs"], te["a"])
            if isinstance(out, tuple):
                y, u = out[0], out[1]
                cmd = (u - te["u_exec"]).abs().mean().item()
            else:
                y = out
                cmd = None
            nmse = (((y - te["y"]) / model.ystd) ** 2).mean().item()
    cos = [
        cosine(model_grad(model, lab["start"], lab["theta"], use_history=use_history, is_mc=is_mc), lab["g"])
        for lab in labels_te
    ]
    # Policy API: ObservableMLP needs a thin wrapper for imagined_eagm — handled by caller flags
    pol = policy_optimize(
        model, train_pol, use_history=use_history, is_mc=is_mc, updates=int(FROZEN["policy_updates"]), seed=pol_seed
    )
    th0 = eval_physical(env, np.zeros(28), test_starts, horizon=H_EVAL)
    phys = eval_physical(env, pol["theta"], test_starts, horizon=H_EVAL)
    return dict(
        dynamics_nmse=nmse,
        cmd_mae=cmd,
        cosine_median=float(np.median(cos)),
        policy_return_mean=phys["mean"],
        policy_return_per_start=phys["per_start"],
        delta_vs_theta0=phys["mean"] - th0["mean"],
        theta0=pol["theta0"].tolist(),
        theta_final=pol["theta"].tolist(),
        theta_l2_delta=float(np.linalg.norm(pol["theta"] - pol["theta0"])),
        policy_curve=pol["curve"],
        theta0_baseline=th0["mean"],
    )
