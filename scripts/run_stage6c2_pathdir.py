#!/usr/bin/env python3
"""Stage 6C.2 / 6C.3 — Path-Aware Dir vs Dir-Random (matched budget) + Mc/Base.

Seeds 15–19 (confirmatory). Same arch/init/minibatch for Dir-Random and Dir-Path.
Path labels from frozen pilot on a Base twin (train starts only); feasible FD.
Does not overwrite prior stage runs. Stop after reports — no Stage 6D.
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
from models.eagm import EAGMBase, McWorld
from training.dataset import fit_stats, flatten_split, make_split, pick_starts
from training.eval_metrics import gradient_quality, stage5e_budget_breakdown
from training.paired_train import clone_model, train_mc, train_pair
from training.policy_opt import STAGE6C_PATH_FRACS, eval_physical_paired_vs_theta0, policy_optimize
from training.rollout import imagined_eagm, imagined_mc
from training.seeds import seed_everything
from training.stage6c_common import (
    PARENT_COMMIT,
    POLICY_UPDATES,
    STAGE6C_SEEDS,
    build_labels_at_thetas,
    config_hash,
    count_fd_control_steps,
    match_label_budget,
    paired_seed_stats,
    path_probe_thetas_from_pilot,
    random_probe_thetas,
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
pack_rows = _stage5e.pack_rows

OUT = EAGM_ROOT / "runs" / "stage6c2_pathdir"
UPDATES = 3000
H_IMAG = 8
H_EVAL = 16
LAM_U = 1.0
LAM_D = 0.1
BATCH = 128
LR_M = 1e-3
LR_P = 0.03
EPISODES_PER_CFG = 2
DTYPE = torch.float32
N_DIR_LABELS = 12  # matched count for Random and Path


def model_grad(model, start, theta, use_history, is_mc=False):
    th = torch.as_tensor(theta, dtype=DTYPE).clone().requires_grad_(True)
    J = imagined_mc(model, th, start, horizon=H_IMAG) if is_mc else imagined_eagm(
        model, th, start, horizon=H_IMAG, use_history=use_history
    )
    return torch.autograd.grad(J, th)[0].detach().numpy()


def probe_grad_quality(model, labels, use_history, is_mc):
    full, stable = [], []
    for lab in labels:
        gh = model_grad(model, lab["start"], lab["theta"], use_history, is_mc)
        q = gradient_quality(gh, lab["g"], lab["mask"])
        full.append(q["cosine_full"])
        stable.append(q["cosine_stable"])
    return dict(
        cosine_full_median=float(np.nanmedian(full)),
        cosine_stable_median=float(np.nanmedian(stable)),
        cosine_full_mean=float(np.nanmean(full)),
        cosine_stable_mean=float(np.nanmean(stable)),
    )


def eval_model(model, te_rows, labels, train_pol, env_eval, test_starts, seed, use_h, is_mc):
    te = pack_rows(te_rows)
    with torch.no_grad():
        if is_mc:
            nmse = (((model(te["x"], te["c"], te["a"]) - te["y"]) / model.ystd) ** 2).mean().item()
            cmd = None
        else:
            y, u = model(te["x"], te["c_obs"], te["a"])
            nmse = (((y - te["y"]) / model.ystd) ** 2).mean().item()
            cmd = (u - te["u_exec"]).abs().mean().item()
    gq = probe_grad_quality(model, labels, use_h, is_mc) if labels else {}
    pol = policy_optimize(
        model, train_pol, use_history=use_h, is_mc=is_mc,
        updates=POLICY_UPDATES, lr=LR_P, seed=seed + 50,
        checkpoint_fracs=STAGE6C_PATH_FRACS,
    )
    paired = eval_physical_paired_vs_theta0(
        env_eval, pol["theta"], test_starts, horizon=H_EVAL, theta0=pol["theta0"]
    )
    return dict(
        dynamics_nmse=nmse,
        cmd_mae=cmd,
        **gq,
        policy_return_mean=paired["policy_return_mean"],
        theta0_return_mean=paired["theta0_return_mean"],
        delta_vs_theta0=paired["delta_vs_theta0_mean"],
        fail_rate_vs_theta0=paired["fail_rate_vs_theta0"],
        theta_param_sat_frac=paired["theta_param_sat_frac"],
        mean_action_clip_frac=paired["mean_action_clip_frac"],
        per_start=paired["per_start"],
        diagnostics=summarize_policy_curve(pol),
        theta0=pol["theta0"].tolist(),
        theta_final=pol["theta"].tolist(),
    )


def run_seed(seed: int) -> dict:
    seed_everything(seed)
    t0 = time.time()
    out_dir = OUT / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_env = PartialUnknownArmEnv(ControllerBConfig(
        episode_randomize=True, **{k: TRAIN_RANGES[k] for k in TRAIN_RANGES},
        gain=1.0, extra_lag=1, rate_limit=0.05, smooth_beta=0.2,
    ))
    n_collect = 36
    episodes = collect_randomized(train_env, n_collect, 200, seed)
    split = make_split(n_collect, seed=seed, n_train=24, n_val=6, n_test=6)
    train_rows = flatten_split(episodes, split.train_eps)
    val_rows = flatten_split(episodes, split.val_eps)
    iid_rows = flatten_split(episodes, split.test_eps)
    stats = fit_stats(train_rows, env_lower=train_env.lower, env_upper=train_env.upper)

    interp_eps = collect_fixed_configs(
        INTERP_CONFIGS, 200, seed_base=10_000 + seed, episodes_per_cfg=EPISODES_PER_CFG
    )
    extrap_eps = collect_fixed_configs(
        EXTRAP_CONFIGS, 200, seed_base=20_000 + seed, episodes_per_cfg=EPISODES_PER_CFG
    )
    interp_rows = [r for ep in interp_eps for r in ep["rows"]]
    extrap_rows = [r for ep in extrap_eps for r in ep["rows"]]

    dir_starts = pick_starts(train_rows, delay=None, max_per_episode=1, max_total=N_DIR_LABELS)
    train_pol = pick_starts(train_rows, delay=None, max_per_episode=1, max_total=12)
    interp_starts = pick_starts(interp_rows, delay=None, max_per_episode=1, max_total=12)
    extrap_starts = pick_starts(extrap_rows, delay=None, max_per_episode=1, max_total=12)
    iid_starts = pick_starts(iid_rows, delay=None, max_per_episode=1, max_total=12)

    # --- Random probe labels (feasible FD for Stage 6C) ---
    thetas_rand = random_probe_thetas(N_DIR_LABELS, seed=seed + 101)
    labels_rand, steps_rand = build_labels_at_thetas(
        train_env, dir_starts, thetas_rand, feasible=True, horizon=H_IMAG
    )
    assert len(labels_rand) == N_DIR_LABELS

    # Diagnostic labels on OOD (not used for training)
    labels_interp, steps_interp = build_labels_at_thetas(
        train_env, interp_starts, thetas_rand, feasible=True, horizon=H_IMAG
    )
    labels_extrap, steps_extrap = build_labels_at_thetas(
        train_env, extrap_starts, thetas_rand, feasible=True, horizon=H_IMAG
    )
    labels_iid, steps_iid = build_labels_at_thetas(
        train_env, iid_starts, thetas_rand, feasible=True, horizon=H_IMAG
    )

    tr = pack_rows(train_rows)

    # --- Mc ---
    seed_everything(seed + 7)
    mc = McWorld(stats)
    train_mc(mc, tr, UPDATES, batch_seed=seed + 20, batch_size=BATCH, lr=LR_M)

    # --- Shared Base init for Base / Dir-Random / Dir-Path ---
    seed_everything(seed + 11)
    base_init = EAGMBase(stats)
    base = clone_model(base_init)
    dir_random = clone_model(base_init)
    dir_path = clone_model(base_init)

    # Pilot path on a frozen copy of init (untrained) would be weak; instead:
    # train a short Base-only pilot twin on same data stream for path θ, OR
    # use Dir-Random protocol: freeze a separately trained Base for pilot.
    # Spec: "Generate a pilot policy trajectory using training starts only"
    # We train Base first (no dir), then pilot on frozen Base, then build path labels,
    # then train Dir-Path from the SAME init with path labels.
    # Dir-Random trains from same init with random labels in parallel stream.
    #
    # Order:
    #   1) Train Base alone (consumes batch stream seed+30)
    #   2) Pilot on frozen Base → path thetas → path labels (train starts)
    #   3) Retrain Dir-Random and Dir-Path from identical init with matched labels
    #      using the SAME batch_seed so minibatch streams match each other.
    # Base already trained in (1); we keep it.

    train_pair(
        base, None, tr, updates=UPDATES, use_history=False, dir_labels=None,
        lam_u=LAM_U, lam_d=LAM_D, batch_size=BATCH, batch_seed=seed + 30, lr=LR_M,
    )
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)

    path_thetas, pilot = path_probe_thetas_from_pilot(
        base, train_pol, use_history=False, updates=POLICY_UPDATES, lr=LR_P,
        seed=seed + 77, fracs=STAGE6C_PATH_FRACS,
    )
    # Map N_DIR_LABELS starts onto path thetas (cycle 6 checkpoints)
    labels_path, steps_path = build_labels_at_thetas(
        train_env, dir_starts, path_thetas, feasible=True, horizon=H_IMAG
    )
    labels_path = match_label_budget(labels_path, N_DIR_LABELS)
    # Rematch step accounting after trim
    steps_path = count_fd_control_steps(labels_path)
    steps_rand = count_fd_control_steps(labels_rand)
    assert len(labels_path) == len(labels_rand) == N_DIR_LABELS

    # Train Dir-Random and Dir-Path from identical init + identical minibatch seed
    # Train them as two separate train_pair calls with same batch_seed — each pairs
    # with a disposable base clone so the Dir twin sees the same batches.
    seed_everything(seed + 11)  # re-seed RNG around training (Adam noise unused on CPU det)
    dir_random = clone_model(base_init)
    base_r = clone_model(base_init)
    train_pair(
        base_r, dir_random, tr, updates=UPDATES, use_history=False, dir_labels=labels_rand,
        lam_u=LAM_U, lam_d=LAM_D, batch_size=BATCH, batch_seed=seed + 40, lr=LR_M,
    )

    dir_path = clone_model(base_init)
    base_p = clone_model(base_init)
    train_pair(
        base_p, dir_path, tr, updates=UPDATES, use_history=False, dir_labels=labels_path,
        lam_u=LAM_U, lam_d=LAM_D, batch_size=BATCH, batch_seed=seed + 40, lr=LR_M,
    )

    models = {
        "mc": (mc, False, True),
        "eagm_base": (base, False, False),
        "eagm_dir_random": (dir_random, False, False),
        "eagm_dir_path": (dir_path, False, False),
    }

    # Grad probes: evaluate each Dir model on BOTH label sets for diagnostics
    results = {"interpolation": {}, "extrapolation": {}, "iid_holdout": {}}
    rows_out = []
    for name, (m, use_h, is_mc) in models.items():
        for pool_name, rows, labels, starts in [
            ("interpolation", interp_rows, labels_interp, interp_starts),
            ("extrapolation", extrap_rows, labels_extrap, extrap_starts),
            ("iid_holdout", iid_rows, labels_iid, iid_starts),
        ]:
            # Unfreeze grads for policy opt / grad probe
            for p in m.parameters():
                p.requires_grad_(True)
            m.train(False)
            ev = eval_model(m, rows, labels, train_pol, train_env, starts, seed, use_h, is_mc)
            # Extra: path-label cosine for Dir models on train path set
            if name.startswith("eagm_dir") and pool_name == "interpolation":
                ev["cosine_on_path_labels"] = probe_grad_quality(m, labels_path, False, False)
                ev["cosine_on_random_labels"] = probe_grad_quality(m, labels_rand, False, False)
            results[pool_name][name] = {k: v for k, v in ev.items() if k not in ("per_start",)}
            for ps in ev["per_start"]:
                rows_out.append(dict(
                    seed=seed, pool=pool_name, model=name,
                    episode_id=ps.get("episode_id"), config_id=ps.get("config_id"),
                    return_=ps["return_"], return_theta0=ps.get("return_theta0"),
                    delta_vs_theta0=ps.get("delta_vs_theta0"),
                ))
            print(
                f"  seed{seed} {pool_name[:5]:5s} {name:18s} "
                f"nmse {ev['dynamics_nmse']:.3f} "
                f"ret {ev['policy_return_mean']:.6f} "
                f"fail {ev['fail_rate_vs_theta0']:.2f} "
                f"θsat {ev['theta_param_sat_frac']:.2f} "
                f"ρ {ev['diagnostics']['mean_rho_proj']:.3f}",
                flush=True,
            )
        torch.save(
            {"state": m.state_dict(), "stats": stats, "kind": name, "seed": seed, "stage": "6c2"},
            out_dir / f"{name}.pt",
        )

    budget = stage5e_budget_breakdown(
        n_train_eps=len(split.train_eps),
        n_val_eps=len(split.val_eps),
        n_iid_hold_eps=len(split.test_eps),
        horizon=200,
        dir_label_train_steps=steps_rand + steps_path,  # both label packs built
        ood_fd_diagnostic_steps=steps_interp + steps_extrap + steps_iid,
    )
    budget.update(dict(
        dir_random_label_steps=steps_rand,
        dir_path_label_steps=steps_path,
        dir_label_count_each=N_DIR_LABELS,
        matched_label_count=len(labels_rand) == len(labels_path),
        pilot_path_fracs=list(STAGE6C_PATH_FRACS),
        pilot_final_sat=pilot["final_theta_param_sat_frac"],
        feasible_fd=True,
        # Each Dir twin only *uses* its own label pack during training:
        per_dir_train_fd_steps=steps_rand,  # equal by construction if steps match
    ))

    def contrast(pool, a, b):
        return results[pool][a]["policy_return_mean"] - results[pool][b]["policy_return_mean"]

    payload = dict(
        seed=seed,
        stage="6C.2",
        parent_commit=PARENT_COMMIT,
        set_label="stage6c3_confirmatory_seeds_15_19",
        hypers=dict(updates=UPDATES, policy_updates=POLICY_UPDATES, h_imag=H_IMAG, h_eval=H_EVAL,
                    lam_u=LAM_U, lam_d=LAM_D, batch=BATCH, lr_m=LR_M, lr_p=LR_P),
        sim_budget=budget,
        results=results,
        contrasts=dict(
            interp_path_minus_random=contrast("interpolation", "eagm_dir_path", "eagm_dir_random"),
            interp_path_minus_base=contrast("interpolation", "eagm_dir_path", "eagm_base"),
            interp_path_minus_mc=contrast("interpolation", "eagm_dir_path", "mc"),
            interp_random_minus_base=contrast("interpolation", "eagm_dir_random", "eagm_base"),
            extrap_path_minus_random=contrast("extrapolation", "eagm_dir_path", "eagm_dir_random"),
            extrap_path_minus_base=contrast("extrapolation", "eagm_dir_path", "eagm_base"),
            extrap_path_minus_mc=contrast("extrapolation", "eagm_dir_path", "mc"),
            extrap_random_minus_base=contrast("extrapolation", "eagm_dir_random", "eagm_base"),
        ),
        wall_seconds=time.time() - t0,
    )
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, default=str))
    with (out_dir / "per_start.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    # Save path thetas for audit
    np.savez(
        out_dir / "path_thetas.npz",
        **{f"f{int(100*f)}": path_thetas[i] for i, f in enumerate(STAGE6C_PATH_FRACS)},
    )
    return payload


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seed_results = []
    for seed in STAGE6C_SEEDS:
        print(f"=== Stage 6C.2 seed {seed} ===", flush=True)
        seed_results.append(run_seed(seed))
        (OUT / "partial.json").write_text(json.dumps(
            dict(done_seeds=[s["seed"] for s in seed_results], seed_results=seed_results),
            indent=2, default=str,
        ))

    paired = {}
    for key in [
        "interp_path_minus_random", "interp_path_minus_base", "interp_path_minus_mc",
        "interp_random_minus_base",
        "extrap_path_minus_random", "extrap_path_minus_base", "extrap_path_minus_mc",
        "extrap_random_minus_base",
    ]:
        diffs = [s["contrasts"][key] for s in seed_results]
        paired[key] = paired_seed_stats(diffs)

    # Go/No-Go
    prim = paired["interp_path_minus_random"]
    keep = (
        prim["n_positive"] >= 4
        and prim["mean"] > 0
        and prim["ci95"][0] > 0
    )
    soft_keep = prim["n_positive"] >= 4 and prim["mean"] > 0
    decision = "KEEP_PATHDIR" if keep else ("KEEP_PATHDIR_SOFT" if soft_keep else "STOP_PATHDIR")

    compact_seeds = []
    for s in seed_results:
        compact_seeds.append(dict(
            seed=s["seed"],
            contrasts=s["contrasts"],
            sim_budget=s["sim_budget"],
            wall_seconds=s["wall_seconds"],
            results_summary={
                pool: {
                    m: {kk: vv for kk, vv in ev.items() if kk not in ("diagnostics", "theta0", "theta_final")}
                    for m, ev in s["results"][pool].items()
                }
                for pool in s["results"]
            },
        ))
    payload = dict(
        stage="6C.2+6C.3",
        parent_commit=PARENT_COMMIT,
        config_hash=config_hash(dict(seeds=STAGE6C_SEEDS, n_labels=N_DIR_LABELS, feasible=True)),
        seeds=STAGE6C_SEEDS,
        paired_seed_stats=paired,
        seed_results=compact_seeds,
        decision_gate=dict(
            primary="interp_path_minus_random",
            rule="mean>0 and ci95_lo>0 and n_positive>=4 → KEEP; else if n_pos>=4 & mean>0 → SOFT; else STOP",
            decision=decision,
            primary_stats=prim,
        ),
        wall_seconds=time.time() - t0,
    )
    # Full seed results are large; also dump compact
    (OUT / "results.json").write_text(json.dumps(payload, indent=2, default=str))
    (OUT / "protocol.json").write_text(json.dumps(dict(
        parent_commit=PARENT_COMMIT,
        seeds=STAGE6C_SEEDS,
        n_dir_labels=N_DIR_LABELS,
        feasible_fd=True,
        path_fracs=list(STAGE6C_PATH_FRACS),
        matched=["arch", "init", "batch_seed", "updates", "lam", "fd_eps", "label_count"],
    ), indent=2))
    print(json.dumps(dict(paired=paired, decision=decision), indent=2, default=str))
    print(f"Wrote {OUT / 'results.json'} wall={payload['wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
