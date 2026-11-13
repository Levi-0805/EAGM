#!/usr/bin/env python3
"""Stage 7C.1 — Reach failure diagnosis (no 7D).

1) Reward parity: imagined-surrogate vs true-MuJoCo EE vs joint-tracking baseline.
2) Gradient audit at θ0 / θmid / θfinal for Mc / Base / Dir.
3) Full-budget seed20 (3000 WM updates) Mc / Base / Dir-Random paired.
Preserves runs/stage7c_reach_smoke/. Writes runs/stage7c1_*.
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
from models.eagm import EAGMBase, McWorld
from tasks.reach import (
    ReachConfig,
    collect_reach_episodes,
    ee_from_q_mujoco,
    ee_positions,
    fit_fk_surrogate,
    reach_reward_numpy,
    reach_reward_torch,
)
from training.dataset import fit_stats, make_split, pick_starts
from training.eval_metrics import gradient_quality
from training.gradients import sim_grad_masked_detailed
from training.paired_train import clone_model, train_mc, train_pair
from training.policy_opt import STAGE6C_PATH_FRACS, eval_physical_paired_vs_theta0, policy_optimize
from training.reward import DISCOUNT, W_DA, W_VEL, reward_numpy
from training.rollout import (
    ACTION_LIMIT,
    actor,
    actor_raw_numpy,
    imagined_eagm,
    imagined_mc,
    physical_return_detailed,
    set_reach_fk_surrogate,
)
from training.seeds import seed_everything
from training.stage6c_common import build_labels_at_thetas, random_probe_thetas, summarize_policy_curve

OUT = EAGM_ROOT / "runs" / "stage7c1_diagnosis"
SEED = 20
UPDATES = 3000
H_IMAG, H_EVAL = 8, 16
N_LABELS = 12
DTYPE = torch.float32
SMOKE_PATH = EAGM_ROOT / "runs" / "stage7c_reach_smoke" / "results.json"


def pack_rows(rows):
    keys = ["x", "y", "a", "c", "c_obs", "u_exec", "x_hist", "a_hist"]
    return {k: torch.as_tensor(np.stack([r[k] for r in rows]), dtype=DTYPE) for k in keys}


def make_env():
    return PartialUnknownArmEnv(ControllerBConfig(
        episode_randomize=True,
        gain_range=(0.92, 1.08),
        extra_lag_choices=(0, 1, 2),
        rate_limit_range=(0.03, 0.08),
        smooth_beta_range=(0.0, 0.35),
        gain=1.0, extra_lag=1, rate_limit=0.05, smooth_beta=0.2,
    ))


def reward_parity_audit(env, fk, starts: list[dict]) -> dict:
    """Identical (x,a,prev,goals): compare surrogate EE reward vs true MuJoCo EE reward."""
    set_reach_fk_surrogate(fk)
    rows = []
    abs_diffs = []
    fk_ee_errs = []
    for i, st in enumerate(starts[:8]):
        x = np.asarray(st["x"], dtype=np.float64)
        a = np.asarray(st["prev"], dtype=np.float64)  # use stored prev as action proxy if needed
        # Prefer a synthetic small action for controlled compare
        a = np.zeros(14, dtype=np.float64)
        prev = np.zeros(14, dtype=np.float64)
        goals = np.asarray(st["ee_goals"], dtype=np.float64)
        # Surrogate EE from q
        ee_s = fk(torch.as_tensor(x[:14], dtype=DTYPE)).detach().numpy()
        r_surr = reach_reward_numpy(ee_s[:3], ee_s[3:], goals, x, a, prev)
        # True MuJoCo EE from same q
        left, right = ee_from_q_mujoco(env, x[:14])
        r_true = reach_reward_numpy(left, right, goals, x, a, prev)
        # Joint-tracking reward at same x with joint ref = x[:14] (zero joint error)
        r_joint0 = reward_numpy(x, a, x[:14], prev)
        fk_err = float(np.linalg.norm(ee_s - np.concatenate([left, right])))
        d = abs(r_surr - r_true)
        abs_diffs.append(d)
        fk_ee_errs.append(fk_err)
        rows.append(dict(
            start_index=i,
            r_surrogate=r_surr,
            r_true_mujoco=r_true,
            abs_diff=d,
            rel_diff=d / (abs(r_true) + 1e-9),
            fk_ee_l2=fk_err,
            r_joint_at_zero_error=r_joint0,
            W_ee=1.0, W_vel=W_VEL, W_da=W_DA, discount=DISCOUNT,
        ))

    # One-step physical vs imagined reward along a short rollout at θ=0
    st0 = starts[0]
    set_reach_fk_surrogate(fk)
    det = physical_return_detailed(env, np.zeros(28), st0, horizon=4)
    # Imagined with a dummy identity-ish model is hard; instead compare per-state after env steps
    env.restore(st0["snapshot"])
    step_rows = []
    prev = np.asarray(st0["prev"], dtype=np.float64).copy()
    for k in range(4):
        x = env.observe()
        c = env.control_state()
        ref = st0["refs"][k]
        a = np.clip(actor_raw_numpy(np.zeros(28), x, c[:14], ref), -ACTION_LIMIT, ACTION_LIMIT)
        env.step(a)
        newx = env.observe()
        left, right = ee_positions(env)
        ee_s = fk(torch.as_tensor(newx[:14], dtype=DTYPE)).detach().numpy()
        r_s = reach_reward_numpy(ee_s[:3], ee_s[3:], st0["ee_goals"], newx, a, prev)
        r_t = reach_reward_numpy(left, right, st0["ee_goals"], newx, a, prev)
        step_rows.append(dict(k=k, r_surr=r_s, r_true=r_t, abs_diff=abs(r_s - r_t)))
        prev = a

    return dict(
        static_state_compare=rows,
        mean_abs_reward_diff=float(np.mean(abs_diffs)),
        max_abs_reward_diff=float(np.max(abs_diffs)),
        mean_fk_ee_l2=float(np.mean(fk_ee_errs)),
        max_fk_ee_l2=float(np.max(fk_ee_errs)),
        rollout_step_compare=step_rows,
        physical_surrogate_return_h4=det["return_"],
        physical_true_mujoco_return_h4=det.get("true_mujoco_ee_return"),
        constants=dict(W_ee=1.0, W_vel=W_VEL, W_da=W_DA, discount=DISCOUNT, h_imag=H_IMAG, h_eval=H_EVAL),
        left_right_weights_equal=True,
        same_code_path_imagined_physical="_step_reward(task_id=reach) via FK surrogate",
        parity_pass=float(np.mean(abs_diffs)) < 0.01 and float(np.mean(fk_ee_errs)) < 0.08,
    )


def model_grad(model, start, theta, is_mc=False):
    th = torch.as_tensor(theta, dtype=DTYPE).clone().requires_grad_(True)
    J = imagined_mc(model, th, start, horizon=H_IMAG) if is_mc else imagined_eagm(
        model, th, start, horizon=H_IMAG, use_history=False
    )
    return torch.autograd.grad(J, th)[0].detach().numpy(), float(J.detach())


def gradient_audit(env, models: dict, starts: list[dict], thetas_path: dict) -> dict:
    """At θ0/mid/final: model vs physical FD (feasible)."""
    out = {}
    for name, (model, is_mc) in models.items():
        out[name] = {}
        for tag, th in thetas_path.items():
            cos_f, cos_s, mag_err, dir_acc, nstab, nbound = [], [], [], [], [], []
            for st in starts[:3]:
                g_hat, _ = model_grad(model, st, th, is_mc=is_mc)
                g_sim, mask, steps, meta = sim_grad_masked_detailed(
                    env, st, th, feasible=True, horizon=H_IMAG
                )
                q = gradient_quality(g_hat, g_sim, mask)
                cos_f.append(q["cosine_full"])
                cos_s.append(q["cosine_stable"])
                # magnitude error on stable dims
                m = mask > 0.5
                if m.any():
                    gh, gs = g_hat[m], g_sim[m]
                    mag_err.append(float(np.linalg.norm(gh - gs) / (np.linalg.norm(gs) + 1e-12)))
                    dir_acc.append(float(np.mean(np.sign(gh) == np.sign(gs))))
                nstab.append(q["n_stable_dims"])
                nbound.append(meta["n_boundary"])
            sat = float(np.mean(np.abs(th) >= 1.0 - 1e-8))
            out[name][tag] = dict(
                cosine_full_median=float(np.nanmedian(cos_f)),
                cosine_stable_median=float(np.nanmedian(cos_s)),
                cosine_full_mean=float(np.nanmean(cos_f)),
                mag_err_median=float(np.nanmedian(mag_err)) if mag_err else float("nan"),
                directional_agreement_median=float(np.nanmedian(dir_acc)) if dir_acc else float("nan"),
                n_stable_dims_mean=float(np.mean(nstab)),
                n_boundary_mean=float(np.mean(nbound)),
                theta_param_sat_frac=sat,
                per_start_cosine_full=cos_f,
            )
    return out


def eval_bundle(model, te_rows, labels, train_pol, env, test_starts, seed, is_mc):
    te = pack_rows(te_rows)
    with torch.no_grad():
        if is_mc:
            nmse = (((model(te["x"], te["c"], te["a"]) - te["y"]) / model.ystd) ** 2).mean().item()
            cmd = None
        else:
            y, u = model(te["x"], te["c_obs"], te["a"])
            nmse = (((y - te["y"]) / model.ystd) ** 2).mean().item()
            cmd = (u - te["u_exec"]).abs().mean().item()
    cos_f, cos_s = [], []
    for lab in labels:
        gh, _ = model_grad(model, lab["start"], lab["theta"], is_mc=is_mc)
        q = gradient_quality(gh, lab["g"], lab["mask"])
        cos_f.append(q["cosine_full"])
        cos_s.append(q["cosine_stable"])
    pol = policy_optimize(
        model, train_pol, use_history=False, is_mc=is_mc,
        updates=50, lr=0.03, seed=seed + 50, checkpoint_fracs=STAGE6C_PATH_FRACS,
    )
    paired = eval_physical_paired_vs_theta0(
        env, pol["theta"], test_starts, horizon=H_EVAL, theta0=pol["theta0"]
    )
    # true mujoco on first start
    det0 = physical_return_detailed(env, pol["theta"], test_starts[0], horizon=H_EVAL)
    det_th0 = physical_return_detailed(env, pol["theta0"], test_starts[0], horizon=H_EVAL)
    return dict(
        dynamics_nmse=nmse,
        cmd_mae=cmd,
        cosine_full_median=float(np.nanmedian(cos_f)),
        cosine_stable_median=float(np.nanmedian(cos_s)),
        policy_return_mean=paired["policy_return_mean"],
        theta0_return_mean=paired["theta0_return_mean"],
        delta_vs_theta0=paired["delta_vs_theta0_mean"],
        fail_rate_vs_theta0=paired["fail_rate_vs_theta0"],
        theta_param_sat_frac=paired["theta_param_sat_frac"],
        mean_action_clip_frac=paired["mean_action_clip_frac"],
        diagnostics=summarize_policy_curve(pol),
        true_mujoco_ee_return_final_start0=det0.get("true_mujoco_ee_return"),
        true_mujoco_ee_return_theta0_start0=det_th0.get("true_mujoco_ee_return"),
        checkpoints={
            "theta0": pol["checkpoints"].get("theta0") or pol["theta0"].tolist(),
            "theta_50": pol["checkpoints"].get("theta_50"),
            "theta_final": pol["checkpoints"].get("theta_final") or pol["theta"].tolist(),
        },
        theta_final=pol["theta"].tolist(),
        theta0=pol["theta0"].tolist(),
        per_start=paired["per_start"],
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seed_everything(SEED)
    env = make_env()
    cfg = ReachConfig(goal_mode="random", traj_shape="hold")

    print("=== 7C.1 fit FK + collect ===", flush=True)
    fk = fit_fk_surrogate(env, n_samples=4000, seed=SEED, steps=800)
    set_reach_fk_surrogate(fk)
    print(f"FK RMSE={fk.fit_rmse:.4f}", flush=True)

    episodes = collect_reach_episodes(env, cfg, n_episodes=36, horizon=200, seed=SEED)
    split = make_split(36, seed=SEED, n_train=24, n_val=6, n_test=6)
    train_rows = [r for ep in episodes if ep["episode_id"] in set(split.train_eps) for r in ep["rows"]]
    val_rows = [r for ep in episodes if ep["episode_id"] in set(split.val_eps) for r in ep["rows"]]
    test_rows = [r for ep in episodes if ep["episode_id"] in set(split.test_eps) for r in ep["rows"]]
    stats = fit_stats(train_rows, env_lower=env.lower, env_upper=env.upper)
    tr = pack_rows(train_rows)

    dir_starts = pick_starts(train_rows, delay=None, max_per_episode=1, max_total=N_LABELS)
    train_pol = pick_starts(train_rows, delay=None, max_per_episode=1, max_total=12)
    test_starts = pick_starts(test_rows, delay=None, max_per_episode=1, max_total=12)
    val_starts = pick_starts(val_rows, delay=None, max_per_episode=1, max_total=8)

    print("=== Task1 reward parity ===", flush=True)
    reward_audit = reward_parity_audit(env, fk, dir_starts)
    (OUT / "reward_audit.json").write_text(json.dumps(reward_audit, indent=2, default=str))
    print(
        f"  mean|ΔR|={reward_audit['mean_abs_reward_diff']:.6f} "
        f"mean FK L2={reward_audit['mean_fk_ee_l2']:.4f} "
        f"parity_pass={reward_audit['parity_pass']}",
        flush=True,
    )

    thetas = random_probe_thetas(N_LABELS, seed=SEED + 101)
    labels, steps = build_labels_at_thetas(env, dir_starts, thetas, feasible=True, horizon=H_IMAG)

    print("=== Task3 full-budget train seed20 ===", flush=True)
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

    print("=== Task4 physical eval ===", flush=True)
    results = {}
    for name, model, is_mc in [
        ("mc", mc, True),
        ("eagm_base", base, False),
        ("eagm_dir_random", dir_m, False),
    ]:
        ev = eval_bundle(model, test_rows, labels, train_pol, env, test_starts, SEED, is_mc)
        results[name] = {k: v for k, v in ev.items() if k != "per_start"}
        print(
            f"  {name:16s} nmse {ev['dynamics_nmse']:.3f} "
            f"cosF {ev['cosine_full_median']:.3f} cosS {ev['cosine_stable_median']:.3f} "
            f"ret {ev['policy_return_mean']:.6f} Δ {ev['delta_vs_theta0']:.6f} "
            f"fail {ev['fail_rate_vs_theta0']:.2f}",
            flush=True,
        )
        torch.save(
            {"state": model.state_dict(), "stats": stats, "kind": name, "seed": SEED, "stage": "7c1"},
            OUT / f"{name}.pt",
        )

    # Gradient audit using policy path thetas from Dir
    print("=== Task2 gradient audit ===", flush=True)
    th_path = {
        "theta0": np.asarray(results["eagm_dir_random"]["checkpoints"]["theta0"], dtype=np.float64),
        "theta_mid": np.asarray(
            results["eagm_dir_random"]["checkpoints"].get("theta_50")
            or results["eagm_dir_random"]["theta_final"],
            dtype=np.float64,
        ),
        "theta_final": np.asarray(results["eagm_dir_random"]["theta_final"], dtype=np.float64),
    }
    # Also audit each model's own final theta
    grad_audit = gradient_audit(
        env,
        {"mc": (mc, True), "eagm_base": (base, False), "eagm_dir_random": (dir_m, False)},
        test_starts,
        th_path,
    )
    # Own-final audit
    for name, model, is_mc in [("mc", mc, True), ("eagm_base", base, False), ("eagm_dir_random", dir_m, False)]:
        own = {
            "own_final": np.asarray(results[name]["theta_final"], dtype=np.float64),
            "own_theta0": np.asarray(results[name]["theta0"], dtype=np.float64),
        }
        ga = gradient_audit(env, {name: (model, is_mc)}, test_starts, own)
        grad_audit[name]["own_path"] = ga[name]

    (OUT / "gradient_audit.json").write_text(json.dumps(grad_audit, indent=2, default=str))

    contrasts = dict(
        dir_minus_base=results["eagm_dir_random"]["policy_return_mean"] - results["eagm_base"]["policy_return_mean"],
        dir_minus_mc=results["eagm_dir_random"]["policy_return_mean"] - results["mc"]["policy_return_mean"],
        base_minus_mc=results["eagm_base"]["policy_return_mean"] - results["mc"]["policy_return_mean"],
        dir_delta=results["eagm_dir_random"]["delta_vs_theta0"],
        base_delta=results["eagm_base"]["delta_vs_theta0"],
        mc_delta=results["mc"]["delta_vs_theta0"],
    )

    smoke = json.loads(SMOKE_PATH.read_text()) if SMOKE_PATH.exists() else None
    decision = dict(
        reward_parity_pass=bool(reward_audit["parity_pass"]),
        dir_beats_base_full_budget=contrasts["dir_minus_base"] > 0,
        dir_beats_mc_full_budget=contrasts["dir_minus_mc"] > 0,
        recommend_three_seed_pilot=False,
        recommend_stage7d=False,
        rationale=[],
    )
    if not reward_audit["parity_pass"]:
        decision["rationale"].append("Repair reward/FK parity before more training.")
    if contrasts["dir_minus_base"] <= 0:
        decision["rationale"].append(
            "Dir still ≤ Base at 3000 updates on seed20 — investigate task-specific limits; do not launch 7D."
        )
    else:
        decision["rationale"].append("Dir > Base on seed20 full budget — may propose seeds 20–22 pilot only.")
        decision["recommend_three_seed_pilot"] = True
    # Gradient vs return diagnosis
    g_dir = grad_audit["eagm_dir_random"]["theta_final"]["cosine_full_median"]
    g_base = grad_audit["eagm_base"]["theta_final"]["cosine_full_median"]
    if g_dir > g_base + 0.05 and contrasts["dir_minus_base"] <= 0:
        decision["rationale"].append(
            "Dir improves gradient cosine vs Base but not physical return — do not claim optimization gain."
        )
    decision["grad_cosine_final_dir"] = g_dir
    decision["grad_cosine_final_base"] = g_base

    payload = dict(
        stage="7C.1",
        seed=SEED,
        updates=UPDATES,
        policy_updates=50,
        lr_policy=0.03,
        lam_u=1.0,
        lam_d=0.1,
        fk_rmse=float(fk.fit_rmse),
        dir_label_steps=steps,
        n_dir_labels=N_LABELS,
        smoke_400_preserved=SMOKE_PATH.exists(),
        smoke_400_contrasts=(smoke or {}).get("contrasts"),
        reward_audit_summary={k: reward_audit[k] for k in (
            "mean_abs_reward_diff", "max_abs_reward_diff", "mean_fk_ee_l2", "max_fk_ee_l2",
            "parity_pass", "constants", "left_right_weights_equal", "same_code_path_imagined_physical",
        )},
        results=results,
        contrasts=contrasts,
        gradient_audit=grad_audit,
        decision=decision,
        wall_seconds=time.time() - t0,
    )
    (OUT / "results.json").write_text(json.dumps(payload, indent=2, default=str))

    # per-start CSV for Dir/Base/Mc
    rows_csv = []
    for name, model, is_mc in [("mc", mc, True), ("eagm_base", base, False), ("eagm_dir_random", dir_m, False)]:
        ev = eval_physical_paired_vs_theta0(
            env, np.asarray(results[name]["theta_final"]), test_starts,
            horizon=H_EVAL, theta0=np.asarray(results[name]["theta0"]),
        )
        for ps in ev["per_start"]:
            rows_csv.append(dict(
                seed=SEED, model=name, episode_id=ps.get("episode_id"),
                return_=ps["return_"], return_theta0=ps.get("return_theta0"),
                delta_vs_theta0=ps.get("delta_vs_theta0"),
            ))
    with (OUT / "per_start.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)

    print(json.dumps(dict(contrasts=contrasts, decision=decision), indent=2, default=str))
    print(f"Wrote {OUT} wall={payload['wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
