#!/usr/bin/env python3
"""Stage 3.2 seed-0: matched-θ Dir labels, paired Base/Dir, physical limits.

Does not overwrite runs/stage3_1_seed0/.
"""
from __future__ import annotations

import json
import platform
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

DTYPE = torch.float32
UPDATES = 3000
H_IMAG = 8
H_EVAL = 16
HIST = 8


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def balanced_dir_acc(g_hat, g_true, mask, eps=1e-12):
    m = (mask > 0.5) & (np.abs(g_true) > eps)
    if m.sum() == 0:
        return float("nan")
    agree = np.sign(g_hat[m]) == np.sign(g_true[m])
    pos = g_true[m] > 0
    neg = ~pos
    scores = []
    if pos.any():
        scores.append(agree[pos].mean())
    if neg.any():
        scores.append(agree[neg].mean())
    return float(np.mean(scores))


def pack_rows(rows, device=torch.device("cpu")):
    keys = ["x", "y", "a", "c", "c_obs", "u_exec", "x_hist", "a_hist"]
    return {
        k: torch.as_tensor(np.stack([r[k] for r in rows]), dtype=DTYPE, device=device)
        for k in keys
    }


def build_dir_labels(env, starts, thetas, mode: str):
    """mode: 'zero' -> all θ=0; 'matched' -> label at each provided θ."""
    labels = []
    total_steps = 0
    records = []
    for i, s in enumerate(starts):
        if mode == "zero":
            th = np.zeros(28)
        else:
            th = np.asarray(thetas[i % len(thetas)], dtype=np.float64)
        g, mask, steps = sim_grad_masked(env, s, th, horizon=H_IMAG)
        total_steps += steps
        s0 = float(np.median(np.abs(g[mask > 0.5]))) if (mask > 0.5).any() else float(np.median(np.abs(g))) or 1.0
        labels.append(dict(start=s, theta=th.copy(), g=g, mask=mask, s0=s0 or 1.0))
        records.append(
            dict(
                index=i,
                episode_id=s.get("episode_id"),
                frame=s.get("frame"),
                mode=mode,
                theta=th.tolist(),
                theta_norm=float(np.linalg.norm(th)),
                n_stable=int(mask.sum()),
                s0=s0,
                steps=steps,
            )
        )
    return labels, records, total_steps


def model_grad(model, start, theta, use_history):
    th = torch.as_tensor(theta, dtype=DTYPE).clone().requires_grad_(True)
    J = imagined_eagm(model, th, start, horizon=H_IMAG, use_history=use_history)
    return torch.autograd.grad(J, th)[0].detach().numpy()


def mc_grad(model, start, theta):
    th = torch.as_tensor(theta, dtype=DTYPE).clone().requires_grad_(True)
    J = imagined_mc(model, th, start, horizon=H_IMAG)
    return torch.autograd.grad(J, th)[0].detach().numpy()


def eval_grad_bank(model, starts, labels_or_thetas, use_history, is_mc=False):
    """Evaluate cosine/bal at the θ used for each start's reference label."""
    cos, bal = [], []
    for i, s in enumerate(starts):
        if isinstance(labels_or_thetas[0], dict):
            lab = labels_or_thetas[i % len(labels_or_thetas)]
            th, g, mask = lab["theta"], lab["g"], lab["mask"]
        else:
            th = labels_or_thetas[i % len(labels_or_thetas)]
            # caller must pass precomputed test labels as dicts
            raise ValueError("pass label dicts")
        if is_mc:
            gh = mc_grad(model, s, th)
        else:
            gh = model_grad(model, s, th, use_history)
        cos.append(cosine(gh, g))
        bal.append(balanced_dir_acc(gh, g, mask))
    return float(np.median(cos)), float(np.nanmedian(bal))


def dyn_cmd(model, te, use_history=False, is_mc=False):
    with torch.no_grad():
        if is_mc:
            y = model(te["x"], te["c"], te["a"])
            return (((y - te["y"]) / model.ystd) ** 2).mean().item(), None
        if use_history:
            y, u, _ = model(te["x"], te["c_obs"], te["a"], te["x_hist"], te["a_hist"])
        else:
            y, u = model(te["x"], te["c_obs"], te["a"])
        return (
            (((y - te["y"]) / model.ystd) ** 2).mean().item(),
            (u - te["u_exec"]).abs().mean().item(),
        )


def main():
    seed = 0
    out = EAGM_ROOT / "runs" / "stage3_2_fixedB" / "seed0"
    out.mkdir(parents=True, exist_ok=False)
    seed_meta = seed_everything(seed)
    t0 = time.time()

    cfg = ControllerBConfig(gain=1.05, extra_lag=1, rate_limit=0.045, smooth_beta=0.25)
    env = PartialUnknownArmEnv(cfg)
    print("collect...", flush=True)
    episodes = collect_trajectories(
        env, n_episodes=24, horizon=200, seed=seed, hist_len=HIST, config_id="B_fixed_105"
    )
    split = make_split(24, seed=seed, n_train=12, n_val=6, n_test=6)
    train_rows = flatten_split(episodes, split.train_eps)
    test_rows = flatten_split(episodes, split.test_eps)
    stats = fit_stats(train_rows, env_lower=env.lower, env_upper=env.upper)

    limit_audit = dict(
        env_lower=env.lower.tolist(),
        env_upper=env.upper.tolist(),
        data_lower_margin=stats["data_lower_margin"].tolist(),
        data_upper_margin=stats["data_upper_margin"].tolist(),
        imagined_uses="env.lower/upper (physical controller projection)",
        max_abs_diff_lo=float(np.max(np.abs(stats["lower"] - stats["data_lower_margin"]))),
        max_abs_diff_hi=float(np.max(np.abs(stats["upper"] - stats["data_upper_margin"]))),
    )

    dir_starts = pick_starts(train_rows, delay=None, max_per_episode=1, max_total=12)
    test_starts = pick_starts(test_rows, delay=None, max_per_episode=2, max_total=12)
    # Freeze non-zero θ bank for matched-theta (independent of training RNG after seed)
    rng_th = np.random.default_rng(seed + 101)
    matched_thetas = [np.zeros(28)]
    for _ in range(11):
        matched_thetas.append(rng_th.normal(0, 0.3, size=28).clip(-1, 1))

    print("dir labels zero-theta...", flush=True)
    labels_zero, rec_zero, steps_zero = build_dir_labels(env, dir_starts, matched_thetas, "zero")
    print("dir labels matched-theta...", flush=True)
    labels_match, rec_match, steps_match = build_dir_labels(
        env, dir_starts, matched_thetas, "matched"
    )
    # Independent test labels at θ=0 and at matched bank (for cosine reporting)
    print("test labels...", flush=True)
    test_lab_zero, _, steps_tz = build_dir_labels(env, test_starts, matched_thetas, "zero")
    test_lab_match, _, steps_tm = build_dir_labels(env, test_starts, matched_thetas, "matched")

    tr = pack_rows(train_rows)
    te = pack_rows(test_rows)

    # --- Train Mc ---
    print("train Mc...", flush=True)
    seed_everything(seed + 7)
    mc = McWorld(stats)
    log_mc = train_mc(mc, tr, UPDATES, batch_seed=seed + 20)

    # --- Paired Base / Dir-Zero / Dir-Matched (Markov) ---
    print("train Markov Base/Dir pairs...", flush=True)
    seed_everything(seed + 11)
    base_proto = EAGMBase(stats)
    dir_zero = clone_model(base_proto)
    # re-init second pair from same proto state
    seed_everything(seed + 11)
    base_proto2 = EAGMBase(stats)
    # Ensure identical init: load from base_proto
    base = clone_model(base_proto)
    dir_zero = clone_model(base_proto)
    dir_match = clone_model(base_proto)
    # Train Base alone first? Spec wants Base vs Dir paired with same batches.
    # Train Base+DirZero as pair1, then Base is already trained — better:
    # Pair A: base_A + dir_zero share batches
    # Pair B: base_B + dir_match share batches (same init as A)
    # Report Base from pair A (identical to what DirZero was paired with).
    seed_everything(seed + 11)
    base = EAGMBase(stats)
    dir_zero = clone_model(base)
    pair_zero = train_pair(
        base,
        dir_zero,
        tr,
        updates=UPDATES,
        use_history=False,
        dir_labels=labels_zero,
        batch_seed=seed + 30,
    )
    seed_everything(seed + 11)
    base_for_match = EAGMBase(stats)  # same init
    dir_match = clone_model(base_for_match)
    pair_match = train_pair(
        base_for_match,
        dir_match,
        tr,
        updates=UPDATES,
        use_history=False,
        dir_labels=labels_match,
        batch_seed=seed + 30,  # same minibatch stream as pair_zero's Base twin
    )
    # Use `base` from pair_zero as EAGM-Base reference (DirZero's twin).
    # base_for_match should match `base` weights if fully deterministic — verify L2.
    with torch.no_grad():
        diff = sum(
            (a - b).pow(2).sum().item()
            for a, b in zip(base.parameters(), base_for_match.parameters())
        )
    print(f"  base twin L2^2 vs match-pair base: {diff:.3e}", flush=True)

    # --- Paired History / History+Dir-Matched ---
    print("train History pairs...", flush=True)
    seed_everything(seed + 13)
    hist = EAGMHistory(stats, hist_len=HIST)
    hist_dir = clone_model(hist)
    pair_hist = train_pair(
        hist,
        hist_dir,
        tr,
        updates=UPDATES,
        use_history=True,
        dir_labels=labels_match,
        batch_seed=seed + 40,
    )

    models = {
        "mc": mc,
        "eagm_base": base,
        "eagm_dir_zero": dir_zero,
        "eagm_dir_matched": dir_match,
        "eagm_hist": hist,
        "eagm_hist_dir_matched": hist_dir,
    }

    # --- Eval ---
    results = {}
    pol_seed = seed + 50
    train_pol = pick_starts(train_rows, delay=None, max_per_episode=1, max_total=12)
    theta0_ret = eval_physical(env, np.zeros(28), test_starts, horizon=H_EVAL)

    for name, m in models.items():
        use_h = name.startswith("eagm_hist")
        is_mc = name == "mc"
        print(f"eval {name}...", flush=True)
        nmse, cmd = dyn_cmd(m, te, use_history=use_h, is_mc=is_mc)
        # Grad metrics: for Dir-Zero report vs zero labels; others vs matched test labels
        if name in ("eagm_dir_zero",):
            cos, bal = eval_grad_bank(m, test_starts, test_lab_zero, use_h, is_mc=False)
            lab_note = "test_labels_theta0"
        elif is_mc:
            cos, bal = eval_grad_bank(m, test_starts, test_lab_match, False, is_mc=True)
            lab_note = "test_labels_matched_theta"
        else:
            cos, bal = eval_grad_bank(m, test_starts, test_lab_match, use_h, is_mc=False)
            lab_note = "test_labels_matched_theta"

        pol = policy_optimize(
            m,
            train_pol,
            use_history=use_h,
            is_mc=is_mc,
            updates=50,
            seed=pol_seed,
        )
        phys = eval_physical(env, pol["theta"], test_starts, horizon=H_EVAL)
        results[name] = dict(
            dynamics_nmse=nmse,
            cmd_mae=cmd,
            cosine_median=cos,
            balanced_dir_acc=bal,
            grad_label_bank=lab_note,
            policy_return_mean=phys["mean"],
            policy_return_std=phys["std"],
            policy_return_per_start=phys["per_start"],
            delta_vs_theta0=phys["mean"] - theta0_ret["mean"],
            theta0=pol["theta0"].tolist(),
            theta_final=pol["theta"].tolist(),
            theta_l2_delta=float(np.linalg.norm(pol["theta"] - pol["theta0"])),
            theta_sat_frac=phys["sat_frac"],
            policy_curve=pol["curve"][::5],  # sparse
            checkpoint=str(out / f"{name}.pt"),
        )
        torch.save({"state": m.state_dict(), "stats": stats, "kind": name}, out / f"{name}.pt")

    # θ identity check Dir vs HistDir finals
    th_dz = np.asarray(results["eagm_dir_matched"]["theta_final"])
    th_hd = np.asarray(results["eagm_hist_dir_matched"]["theta_final"])
    th_dz0 = np.asarray(results["eagm_dir_zero"]["theta_final"])

    payload = dict(
        protocol="stage3_2",
        seed=seed,
        seed_meta=seed_meta,
        updates=UPDATES,
        controller_b=cfg.__dict__,
        split=dict(train=split.train_eps, val=split.val_eps, test=split.test_eps),
        limit_audit=limit_audit,
        base_twin_param_l2sq=diff,
        dir_label_cost=dict(
            zero_steps=steps_zero,
            matched_steps=steps_match,
            test_zero_steps=steps_tz,
            test_matched_steps=steps_tm,
            total_steps=steps_zero + steps_match + steps_tz + steps_tm,
        ),
        dir_label_records=dict(zero=rec_zero, matched=rec_match),
        pair_logs=dict(zero=pair_zero, matched=pair_match, hist=pair_hist, mc=log_mc),
        theta0_baseline_return=theta0_ret,
        results=results,
        theta_final_l2=dict(
            dir_matched_vs_dir_zero=float(np.linalg.norm(th_dz - th_dz0)),
            hist_dir_vs_dir_matched=float(np.linalg.norm(th_hd - th_dz)),
        ),
        wall_seconds=time.time() - t0,
        host=platform.uname()._asdict(),
        note="Matched-θ Dir; paired Base/Dir; physical env limits in imagined rollout.",
    )
    (out / "results.json").write_text(json.dumps(payload, indent=2, default=str))
    # concise print
    summary = {
        k: dict(
            nmse=round(v["dynamics_nmse"], 4),
            cmd=None if v["cmd_mae"] is None else round(v["cmd_mae"], 5),
            cos=round(v["cosine_median"], 3),
            ret=round(v["policy_return_mean"], 6),
            d_th0=round(v["delta_vs_theta0"], 6),
            th_l2=round(v["theta_l2_delta"], 4),
        )
        for k, v in results.items()
    }
    print(json.dumps(summary, indent=2), flush=True)
    print("wrote", out, "wall", round(payload["wall_seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
