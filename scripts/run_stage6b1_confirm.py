#!/usr/bin/env python3
"""Stage 6B.1: independent seeds 10–14 confirming Stage 5E under Stage 6A metrics.

Protocol: Stage 5E legacy (2 episodes/config). No model/hyperparameter changes.
Compound OOD configs are NOT part of the main conclusion this round.
Does not overwrite Stage 5 / Stage 6A artifacts. Does not launch Stage 6B.2.
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
from models.eagm import EAGMBase, EAGMHistory, McWorld
from training.dataset import fit_stats, flatten_split, make_split, pick_starts
from training.eval_metrics import gradient_quality, stage5e_budget_breakdown
from training.gradients import sim_grad_masked
from training.paired_train import clone_model, train_mc, train_pair
from training.policy_opt import eval_physical_paired_vs_theta0, policy_optimize
from training.rollout import imagined_eagm, imagined_mc
from training.seeds import seed_everything

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

FROZEN5 = json.loads((EAGM_ROOT / "configs" / "stage5_frozen.json").read_text())
FROZEN6 = json.loads((EAGM_ROOT / "configs" / "stage6_frozen.json").read_text())
# Lock hypers to Stage 5 values explicitly
UPDATES = 3000
H_IMAG = 8
H_EVAL = 16
HIST = 8
LAM_U = 1.0
LAM_D = 0.1
BATCH = 128
LR_M = 1e-3
LR_P = 0.03
SEEDS = [10, 11, 12, 13, 14]
DTYPE = torch.float32
EPISODES_PER_CFG = 2  # Stage 5E legacy for round-1 comparability


def assert_hypers_locked():
    h = FROZEN6["stage5_hypers_locked"]
    assert h["world_model_updates"] == UPDATES
    assert h["policy_updates"] == 50
    assert h["imagined_horizon"] == H_IMAG
    assert h["eval_horizon"] == H_EVAL
    assert h["hist_len"] == HIST
    assert h["lam_u"] == LAM_U and h["lam_d"] == LAM_D
    assert h["batch_size"] == BATCH
    assert h["lr_model"] == LR_M and h["lr_policy"] == LR_P
    assert FROZEN5["world_model_updates"] == UPDATES
    assert EPISODES_PER_CFG == int(FROZEN6["ood_eval"]["episodes_per_config_stage5e_legacy"])


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


def probe_grad_quality(model, labels, use_history, is_mc):
    full, stable, bacc, derr, nstab = [], [], [], [], []
    for lab in labels:
        gh = model_grad(model, lab["start"], lab["theta"], use_history, is_mc)
        q = gradient_quality(gh, lab["g"], lab["mask"])
        full.append(q["cosine_full"])
        stable.append(q["cosine_stable"])
        bacc.append(q["balanced_dir_acc"])
        derr.append(q["directional_error"])
        nstab.append(q["n_stable_dims"])
    return dict(
        cosine_full_median=float(np.nanmedian(full)),
        cosine_stable_median=float(np.nanmedian(stable)),
        balanced_dir_acc_median=float(np.nanmedian(bacc)),
        directional_error_median=float(np.nanmedian(derr)),
        n_stable_dims_mean=float(np.mean(nstab)),
        cosine_full_mean=float(np.nanmean(full)),
        cosine_stable_mean=float(np.nanmean(stable)),
    )


def eval_on_pool(model, te_rows, labels, train_pol, env_eval, test_starts, seed, use_h, is_mc):
    te = pack_rows(te_rows)
    with torch.no_grad():
        if is_mc:
            nmse = (((model(te["x"], te["c"], te["a"]) - te["y"]) / model.ystd) ** 2).mean().item()
            cmd = None
        elif use_h:
            y, u, _ = model(te["x"], te["c_obs"], te["a"], te["x_hist"], te["a_hist"])
            nmse = (((y - te["y"]) / model.ystd) ** 2).mean().item()
            cmd = (u - te["u_exec"]).abs().mean().item()
        else:
            y, u = model(te["x"], te["c_obs"], te["a"])
            nmse = (((y - te["y"]) / model.ystd) ** 2).mean().item()
            cmd = (u - te["u_exec"]).abs().mean().item()
    gq = probe_grad_quality(model, labels, use_h, is_mc)
    pol = policy_optimize(
        model, train_pol, use_history=use_h, is_mc=is_mc,
        updates=50, lr=LR_P, seed=seed + 50,
    )
    paired = eval_physical_paired_vs_theta0(
        env_eval, pol["theta"], test_starts, horizon=H_EVAL, theta0=pol["theta0"], fail_eps=0.0
    )
    return dict(
        dynamics_nmse=nmse,
        cmd_mae=cmd,
        **gq,
        # backward-compatible alias
        cosine_median=gq["cosine_full_median"],
        policy_return_mean=paired["policy_return_mean"],
        theta0_return_mean=paired["theta0_return_mean"],
        delta_vs_theta0=paired["delta_vs_theta0_mean"],
        fail_rate_vs_theta0=paired["fail_rate_vs_theta0"],
        theta_param_sat_frac=paired["theta_param_sat_frac"],
        mean_action_clip_frac=paired["mean_action_clip_frac"],
        per_config=paired["per_config"],
        per_start=paired["per_start"],
        policy_checkpoints={k: v for k, v in pol.get("checkpoints", {}).items() if k.startswith("theta")},
        final_theta_param_sat_frac=pol["final_theta_param_sat_frac"],
        theta0=pol["theta0"].tolist(),
        theta_final=pol["theta"].tolist(),
    )


def run_seed(seed: int, out_dir: Path) -> dict:
    seed_everything(seed)
    t0 = time.time()
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

    dir_starts = pick_starts(train_rows, delay=None, max_per_episode=1, max_total=12)
    train_pol = pick_starts(train_rows, delay=None, max_per_episode=1, max_total=12)
    interp_starts = pick_starts(interp_rows, delay=None, max_per_episode=1, max_total=12)
    extrap_starts = pick_starts(extrap_rows, delay=None, max_per_episode=1, max_total=12)
    # IID holdout starts for secondary reporting (not mixed into OOD tables)
    iid_starts = pick_starts(iid_rows, delay=None, max_per_episode=1, max_total=12)

    rng_th = np.random.default_rng(seed + 101)
    thetas = [np.zeros(28)] + [rng_th.normal(0, 0.3, 28).clip(-1, 1) for _ in range(11)]
    labels_tr, steps_tr = build_matched_labels(train_env, dir_starts, thetas)
    labels_interp, steps_interp = build_matched_labels(train_env, interp_starts, thetas)
    labels_extrap, steps_extrap = build_matched_labels(train_env, extrap_starts, thetas)
    labels_iid, steps_iid = build_matched_labels(train_env, iid_starts, thetas)

    tr = pack_rows(train_rows)

    seed_everything(seed + 7)
    mc = McWorld(stats)
    train_mc(mc, tr, UPDATES, batch_seed=seed + 20, batch_size=BATCH, lr=LR_M)

    seed_everything(seed + 11)
    base = EAGMBase(stats)
    dir_m = clone_model(base)
    train_pair(
        base, dir_m, tr, updates=UPDATES, use_history=False, dir_labels=labels_tr,
        lam_u=LAM_U, lam_d=LAM_D, batch_size=BATCH, batch_seed=seed + 30, lr=LR_M,
    )

    seed_everything(seed + 13)
    hist = EAGMHistory(stats, hist_len=HIST)
    hist_dir = clone_model(hist)
    train_pair(
        hist, hist_dir, tr, updates=UPDATES, use_history=True, dir_labels=labels_tr,
        lam_u=LAM_U, lam_d=LAM_D, batch_size=BATCH, batch_seed=seed + 40, lr=LR_M,
    )

    models = {
        "mc": (mc, False, True),
        "eagm_base": (base, False, False),
        "eagm_dir_matched": (dir_m, False, False),
        "eagm_hist": (hist, True, False),
        "eagm_hist_dir_matched": (hist_dir, True, False),
    }

    results = {"interpolation": {}, "extrapolation": {}, "iid_holdout": {}}
    rows_out = []
    for name, (m, use_h, is_mc) in models.items():
        for pool_name, rows, labels, starts in [
            ("interpolation", interp_rows, labels_interp, interp_starts),
            ("extrapolation", extrap_rows, labels_extrap, extrap_starts),
            ("iid_holdout", iid_rows, labels_iid, iid_starts),
        ]:
            ev = eval_on_pool(m, rows, labels, train_pol, train_env, starts, seed, use_h, is_mc)
            results[pool_name][name] = {k: v for k, v in ev.items() if k not in ("per_start",)}
            for ps in ev["per_start"]:
                rows_out.append(dict(
                    seed=seed, pool=pool_name, model=name,
                    episode_id=ps["episode_id"], config_id=ps.get("config_id"),
                    return_=ps["return_"], return_theta0=ps.get("return_theta0"),
                    delta_vs_theta0=ps.get("delta_vs_theta0"),
                ))
            print(
                f"  seed{seed} {pool_name[:5]:5s} {name:22s} "
                f"nmse {ev['dynamics_nmse']:.3f} "
                f"cosF {ev['cosine_full_median']:.3f} cosS {ev['cosine_stable_median']:.3f} "
                f"ret {ev['policy_return_mean']:.6f} "
                f"fail {ev['fail_rate_vs_theta0']:.2f} "
                f"θsat {ev['theta_param_sat_frac']:.2f}",
                flush=True,
            )
        torch.save(
            {"state": m.state_dict(), "stats": stats, "kind": name, "seed": seed, "stage": "6b1"},
            out_dir / f"{name}.pt",
        )

    budget = stage5e_budget_breakdown(
        n_train_eps=len(split.train_eps),
        n_val_eps=len(split.val_eps),
        n_iid_hold_eps=len(split.test_eps),
        horizon=200,
        dir_label_train_steps=steps_tr,
        ood_fd_diagnostic_steps=steps_interp + steps_extrap + steps_iid,
    )
    budget.update(dict(
        interp_fd_diagnostic=steps_interp,
        extrap_fd_diagnostic=steps_extrap,
        iid_fd_diagnostic=steps_iid,
        val_rows=len(val_rows),
        episodes_per_config=EPISODES_PER_CFG,
        protocol="stage5e_legacy_2eps_per_config",
    ))

    def contrast(pool, a, b):
        return results[pool][a]["policy_return_mean"] - results[pool][b]["policy_return_mean"]

    payload = dict(
        seed=seed,
        stage="6B.1",
        set_label="stage6b1_confirmatory_seeds_10_14",
        protocol="stage5e_legacy",
        episodes_per_config=EPISODES_PER_CFG,
        hypers=dict(updates=UPDATES, policy_updates=50, h_imag=H_IMAG, h_eval=H_EVAL,
                    hist=HIST, lam_u=LAM_U, lam_d=LAM_D, batch=BATCH, lr_m=LR_M, lr_p=LR_P),
        train_ranges=TRAIN_RANGES,
        split=dict(train=split.train_eps, val=split.val_eps, iid_hold=split.test_eps),
        sim_budget=budget,
        results=results,
        contrasts=dict(
            interp_dir_minus_base=contrast("interpolation", "eagm_dir_matched", "eagm_base"),
            interp_dir_minus_mc=contrast("interpolation", "eagm_dir_matched", "mc"),
            interp_hist_dir_minus_mc=contrast("interpolation", "eagm_hist_dir_matched", "mc"),
            interp_hist_dir_minus_hist=contrast("interpolation", "eagm_hist_dir_matched", "eagm_hist"),
            extrap_dir_minus_base=contrast("extrapolation", "eagm_dir_matched", "eagm_base"),
            extrap_dir_minus_mc=contrast("extrapolation", "eagm_dir_matched", "mc"),
            extrap_hist_dir_minus_mc=contrast("extrapolation", "eagm_hist_dir_matched", "mc"),
            extrap_hist_dir_minus_hist=contrast("extrapolation", "eagm_hist_dir_matched", "eagm_hist"),
        ),
        wall_seconds=time.time() - t0,
    )
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, default=str))
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
        std=float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
        median=float(np.median(diffs)),
        ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        n=int(len(diffs)),
        n_positive=int((diffs > 0).sum()),
        per_seed=diffs.tolist(),
        note="Seed-level paired bootstrap (n=5). Starts are not independent seeds.",
    )


def main():
    assert_hypers_locked()
    root = EAGM_ROOT / "runs" / "stage6b1_confirm"
    root.mkdir(parents=True, exist_ok=False)
    (root / "protocol.json").write_text(json.dumps(dict(
        stage="6B.1",
        seeds=SEEDS,
        episodes_per_config=EPISODES_PER_CFG,
        hypers=dict(updates=UPDATES, policy_updates=50, h_imag=H_IMAG, h_eval=H_EVAL,
                    hist=HIST, lam_u=LAM_U, lam_d=LAM_D, batch=BATCH, lr_m=LR_M, lr_p=LR_P),
        train_ranges=TRAIN_RANGES,
        interp=INTERP_CONFIGS,
        extrap=EXTRAP_CONFIGS,
        compound_excluded_from_main=True,
        head_commit_at_launch=None,
        note="Independent confirmation of Stage 5E; Stage 6A metrics; no structure change.",
    ), indent=2))
    # stamp git SHA if available
    try:
        import subprocess
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=EAGM_ROOT, text=True).strip()
        proto = json.loads((root / "protocol.json").read_text())
        proto["head_commit_at_launch"] = sha
        (root / "protocol.json").write_text(json.dumps(proto, indent=2))
    except Exception:
        pass

    all_seed = []
    t0 = time.time()
    for seed in SEEDS:
        print(f"==== Stage6B.1 confirmatory seed {seed} ====", flush=True)
        sd = root / f"seed{seed}"
        sd.mkdir(exist_ok=False)
        all_seed.append(run_seed(seed, sd))
        (root / "partial.json").write_text(json.dumps(all_seed, indent=2, default=str))

    keys = list(all_seed[0]["contrasts"].keys())
    paired = {k: paired_ci([s["contrasts"][k] for s in all_seed]) for k in keys}

    def pool_metric(pool, model, field):
        return [s["results"][pool][model][field] for s in all_seed]

    summary_tables = {}
    for pool in ("interpolation", "extrapolation"):
        summary_tables[pool] = {
            name: dict(
                return_mean=float(np.mean(pool_metric(pool, name, "policy_return_mean"))),
                fail_rate_mean=float(np.mean(pool_metric(pool, name, "fail_rate_vs_theta0"))),
                theta_sat_mean=float(np.mean(pool_metric(pool, name, "theta_param_sat_frac"))),
                action_clip_mean=float(np.mean(pool_metric(pool, name, "mean_action_clip_frac"))),
                nmse_mean=float(np.mean(pool_metric(pool, name, "dynamics_nmse"))),
                cos_full_mean=float(np.mean(pool_metric(pool, name, "cosine_full_median"))),
                cos_stable_mean=float(np.mean(pool_metric(pool, name, "cosine_stable_median"))),
            )
            for name in all_seed[0]["results"][pool]
        }

    out = dict(
        stage="6B.1",
        confirmatory_seeds=SEEDS,
        exploratory_stage5e_seeds=[5, 6, 7, 8, 9],
        protocol="stage5e_legacy_2eps",
        paired_seed_stats=paired,
        pool_summary=summary_tables,
        per_seed=[{
            "seed": s["seed"],
            **s["contrasts"],
            "interp": {k: dict(
                ret=s["results"]["interpolation"][k]["policy_return_mean"],
                fail=s["results"]["interpolation"][k]["fail_rate_vs_theta0"],
                θsat=s["results"]["interpolation"][k]["theta_param_sat_frac"],
                nmse=s["results"]["interpolation"][k]["dynamics_nmse"],
                cosF=s["results"]["interpolation"][k]["cosine_full_median"],
                cosS=s["results"]["interpolation"][k]["cosine_stable_median"],
            ) for k in s["results"]["interpolation"]},
            "extrap": {k: dict(
                ret=s["results"]["extrapolation"][k]["policy_return_mean"],
                fail=s["results"]["extrapolation"][k]["fail_rate_vs_theta0"],
                θsat=s["results"]["extrapolation"][k]["theta_param_sat_frac"],
                nmse=s["results"]["extrapolation"][k]["dynamics_nmse"],
                cosF=s["results"]["extrapolation"][k]["cosine_full_median"],
                cosS=s["results"]["extrapolation"][k]["cosine_stable_median"],
            ) for k in s["results"]["extrapolation"]},
        } for s in all_seed],
        seed_results=all_seed,
        wall_seconds=time.time() - t0,
        decision_gate=(
            "STOP for human review. Do not auto-start Stage 6B.2. "
            "Decision tree: (A) Dir−Base stable → HistC; "
            "(B) Dir unstable → diagnose labels/FD; "
            "(C) Dir stable but Hist+Dir ≰ Mc → execution-state estimation focus."
        ),
    )
    (root / "results.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(paired, indent=2), flush=True)
    print("wrote", root, "wall", round(out["wall_seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
