#!/usr/bin/env python3
"""Stage 6B.2: HistC execution-state history ablation + capacity controls + probes.

Does NOT start Stage 6B.3 or Stage 6C. No privileged B state as model inputs.
c_obs history = observable (target, filter, delay) only.
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
from models.eagm import (
    EAGMBase,
    EAGMHistory,
    EAGMHistoryC,
    McWorld,
    count_params,
    match_hist_encoder_hidden_to_histc,
)
from training.dataset import collect_trajectories, fit_stats, flatten_split, make_split, pick_starts
from training.eval_metrics import gradient_quality, stage5e_budget_breakdown
from training.exec_state_probe import fit_exec_state_probe
from training.gradients import sim_grad_masked
from training.paired_train import clone_model, train_mc, train_pair
from training.policy_opt import eval_physical_paired_vs_theta0, policy_optimize
from training.rollout import imagined_eagm, imagined_mc
from training.seeds import seed_everything

FROZEN6 = json.loads((EAGM_ROOT / "configs" / "stage6_frozen.json").read_text())
OOD = json.loads((EAGM_ROOT / "configs" / "stage6_ood_configs.json").read_text())
UPDATES = 3000
H_IMAG = 8
H_EVAL = 16
LAM_U, LAM_D = 1.0, 0.1
BATCH, LR_M, LR_P = 128, 1e-3, 0.03
SEEDS = [10, 11, 12, 13, 14]
# Collect at max length; slice per model
HIST_COLLECT = 16
K_MAIN = 8
K_ABLATION = [4, 16]  # Hist+Dir vs HistC+Dir only
EPISODES_PER_CFG = 2  # comparable to Stage 5E / 6B.1 main tables
DTYPE = torch.float32
TRAIN_RANGES = dict(
    gain_range=tuple(OOD["train_ranges"]["gain_range"]),
    extra_lag_choices=tuple(OOD["train_ranges"]["extra_lag_choices"]),
    rate_limit_range=tuple(OOD["train_ranges"]["rate_limit_range"]),
    smooth_beta_range=tuple(OOD["train_ranges"]["smooth_beta_range"]),
)
INTERP = OOD["interpolation"]
EXTRAP = OOD["single_factor_extrapolation"]


def pack_rows(rows, hist_len: int):
    keys = ["x", "y", "a", "c", "c_obs", "u_exec"]
    out = {k: torch.as_tensor(np.stack([r[k] for r in rows]), dtype=DTYPE) for k in keys}
    out["x_hist"] = torch.as_tensor(np.stack([r["x_hist"][-hist_len:] for r in rows]), dtype=DTYPE)
    out["a_hist"] = torch.as_tensor(np.stack([r["a_hist"][-hist_len:] for r in rows]), dtype=DTYPE)
    out["c_obs_hist"] = torch.as_tensor(np.stack([r["c_obs_hist"][-hist_len:] for r in rows]), dtype=DTYPE)
    if "exec_state_scalars" in rows[0]:
        out["exec_state_scalars"] = torch.as_tensor(
            np.stack([r["exec_state_scalars"] for r in rows]), dtype=DTYPE
        )
    return out


def slice_start_hist(start: dict, hist_len: int) -> dict:
    s = dict(start)
    s["x_hist"] = np.asarray(start["x_hist"])[-hist_len:]
    s["a_hist"] = np.asarray(start["a_hist"])[-hist_len:]
    s["c_obs_hist"] = np.asarray(start["c_obs_hist"])[-hist_len:]
    return s


def collect_fixed(configs, seed_base, hist_len, episodes_per_cfg=EPISODES_PER_CFG):
    episodes = []
    ep_id = 0
    for cfg in configs:
        env = PartialUnknownArmEnv(ControllerBConfig(
            gain=cfg["gain"], extra_lag=cfg["extra_lag"],
            rate_limit=cfg["rate_limit"], smooth_beta=cfg["smooth_beta"],
            episode_randomize=False,
        ))
        eps = collect_trajectories(
            env, n_episodes=episodes_per_cfg, horizon=200,
            seed=seed_base + ep_id * 17, hist_len=hist_len, config_id=cfg["id"],
        )
        for ep in eps:
            ep["episode_id"] = ep_id
            for r in ep["rows"]:
                r["episode_id"] = ep_id
                r["start"]["episode_id"] = ep_id
                r["config_id"] = cfg["id"]
                r["start"]["config_id"] = cfg["id"]
            episodes.append(ep)
            ep_id += 1
    return episodes


def build_labels(env, starts, thetas, hist_len):
    labels, steps = [], 0
    for i, s0 in enumerate(starts):
        s = slice_start_hist(s0, hist_len)
        th = np.asarray(thetas[i % len(thetas)], dtype=np.float64)
        g, mask, st = sim_grad_masked(env, s, th, horizon=H_IMAG)
        steps += st
        s0s = float(np.median(np.abs(g[mask > 0.5]))) if (mask > 0.5).any() else float(np.median(np.abs(g))) or 1.0
        labels.append(dict(start=s, theta=th.copy(), g=g, mask=mask, s0=s0s or 1.0))
    return labels, steps


def model_grad(model, start, theta, use_history, is_mc=False):
    th = torch.as_tensor(theta, dtype=DTYPE).clone().requires_grad_(True)
    J = imagined_mc(model, th, start, horizon=H_IMAG) if is_mc else imagined_eagm(
        model, th, start, horizon=H_IMAG, use_history=use_history
    )
    return torch.autograd.grad(J, th)[0].detach().numpy()


def probe_grads(model, labels, use_history, is_mc):
    full, stab, bacc = [], [], []
    for lab in labels:
        gh = model_grad(model, lab["start"], lab["theta"], use_history, is_mc)
        q = gradient_quality(gh, lab["g"], lab["mask"])
        full.append(q["cosine_full"])
        stab.append(q["cosine_stable"])
        bacc.append(q["balanced_dir_acc"])
    return dict(
        cosine_full_median=float(np.nanmedian(full)),
        cosine_stable_median=float(np.nanmedian(stab)),
        balanced_dir_acc_median=float(np.nanmedian(bacc)),
    )


def eval_model(model, te_rows, labels, train_pol, env, starts, seed, *, use_h, is_mc, hist_len):
    te = pack_rows(te_rows, hist_len)
    train_pol_s = [slice_start_hist(s, hist_len) for s in train_pol]
    starts_s = [slice_start_hist(s, hist_len) for s in starts]
    with torch.no_grad():
        if is_mc:
            nmse = (((model(te["x"], te["c"], te["a"]) - te["y"]) / model.ystd) ** 2).mean().item()
            cmd = None
        elif use_h:
            if getattr(model, "uses_c_obs_hist", False):
                y, u, _ = model(te["x"], te["c_obs"], te["a"], te["x_hist"], te["a_hist"], te["c_obs_hist"])
            else:
                y, u, _ = model(te["x"], te["c_obs"], te["a"], te["x_hist"], te["a_hist"])
            nmse = (((y - te["y"]) / model.ystd) ** 2).mean().item()
            cmd = (u - te["u_exec"]).abs().mean().item()
        else:
            y, u = model(te["x"], te["c_obs"], te["a"])
            nmse = (((y - te["y"]) / model.ystd) ** 2).mean().item()
            cmd = (u - te["u_exec"]).abs().mean().item()
    gq = probe_grads(model, labels, use_h, is_mc)
    pol = policy_optimize(
        model, train_pol_s, use_history=use_h, is_mc=is_mc, updates=50, lr=LR_P, seed=seed + 50,
    )
    paired = eval_physical_paired_vs_theta0(
        env, pol["theta"], starts_s, horizon=H_EVAL, theta0=pol["theta0"]
    )
    return dict(
        dynamics_nmse=nmse,
        cmd_mae=cmd,
        n_params=count_params(model),
        **gq,
        policy_return_mean=paired["policy_return_mean"],
        delta_vs_theta0=paired["delta_vs_theta0_mean"],
        fail_rate_vs_theta0=paired["fail_rate_vs_theta0"],
        fail_rate_detail=paired["fail_rate_detail"],
        theta_param_sat_frac=paired["theta_param_sat_frac"],
        mean_action_clip_frac=paired["mean_action_clip_frac"],
        per_config=paired["per_config"],
        per_start=paired["per_start"],
    )


def run_seed(seed: int, out_dir: Path) -> dict:
    seed_everything(seed)
    t0 = time.time()
    train_env = PartialUnknownArmEnv(ControllerBConfig(
        episode_randomize=True, **TRAIN_RANGES,
        gain=1.0, extra_lag=1, rate_limit=0.05, smooth_beta=0.2,
    ))
    episodes = collect_trajectories(
        train_env, n_episodes=36, horizon=200, seed=seed, hist_len=HIST_COLLECT, config_id="B_rand_train"
    )
    # Stamp config_id / b_active onto rows for analysis
    for ep in episodes:
        ba = dict(train_env._active)  # last episode's; better re-read from snapshot if needed
        # Prefer per-row from snapshot after collection — use privileged label if present
        pass
    split = make_split(36, seed=seed, n_train=24, n_val=6, n_test=6)
    train_rows = flatten_split(episodes, split.train_eps)
    stats = fit_stats(train_rows, env_lower=train_env.lower, env_upper=train_env.upper)

    interp_eps = collect_fixed(INTERP, 10_000 + seed, HIST_COLLECT)
    extrap_eps = collect_fixed(EXTRAP, 20_000 + seed, HIST_COLLECT)
    interp_rows = [r for ep in interp_eps for r in ep["rows"]]
    extrap_rows = [r for ep in extrap_eps for r in ep["rows"]]

    dir_starts0 = pick_starts(train_rows, delay=None, max_per_episode=1, max_total=12)
    train_pol0 = pick_starts(train_rows, delay=None, max_per_episode=1, max_total=12)
    interp_starts0 = pick_starts(interp_rows, delay=None, max_per_episode=1, max_total=12)
    extrap_starts0 = pick_starts(extrap_rows, delay=None, max_per_episode=1, max_total=12)

    rng_th = np.random.default_rng(seed + 101)
    thetas = [np.zeros(28)] + [rng_th.normal(0, 0.3, 28).clip(-1, 1) for _ in range(11)]

    # --- Main k=8 suite ---
    k = K_MAIN
    labels_tr, steps_tr = build_labels(train_env, dir_starts0, thetas, k)
    labels_i, steps_i = build_labels(train_env, interp_starts0, thetas, k)
    labels_e, steps_e = build_labels(train_env, extrap_starts0, thetas, k)
    tr = pack_rows(train_rows, k)

    matched_h = match_hist_encoder_hidden_to_histc(stats, hist_len=k)
    histc_ref = EAGMHistoryC(stats, hist_len=k)
    hist_wide = EAGMHistory(stats, hist_len=k, encoder_hidden=matched_h)
    print(
        f"  seed{seed} param match: HistC={count_params(histc_ref)} "
        f"HistWide(h={matched_h})={count_params(hist_wide)} "
        f"Hist64={count_params(EAGMHistory(stats, hist_len=k))}",
        flush=True,
    )

    seed_everything(seed + 7)
    mc = McWorld(stats)
    train_mc(mc, tr, UPDATES, batch_seed=seed + 20, batch_size=BATCH, lr=LR_M)

    seed_everything(seed + 11)
    base = EAGMBase(stats)
    dir_m = clone_model(base)
    train_pair(base, dir_m, tr, updates=UPDATES, use_history=False, dir_labels=labels_tr,
               lam_u=LAM_U, lam_d=LAM_D, batch_size=BATCH, batch_seed=seed + 30, lr=LR_M)

    seed_everything(seed + 13)
    hist = EAGMHistory(stats, hist_len=k)
    hist_dir = clone_model(hist)
    train_pair(hist, hist_dir, tr, updates=UPDATES, use_history=True, dir_labels=labels_tr,
               lam_u=LAM_U, lam_d=LAM_D, batch_size=BATCH, batch_seed=seed + 40, lr=LR_M)

    seed_everything(seed + 15)
    histc = EAGMHistoryC(stats, hist_len=k)
    histc_dir = clone_model(histc)
    train_pair(histc, histc_dir, tr, updates=UPDATES, use_history=True, dir_labels=labels_tr,
               lam_u=LAM_U, lam_d=LAM_D, batch_size=BATCH, batch_seed=seed + 50, lr=LR_M)

    seed_everything(seed + 17)
    hist_wide = EAGMHistory(stats, hist_len=k, encoder_hidden=matched_h)
    hist_wide.kind = "eagm_history_param_matched"
    hist_wide_dir = clone_model(hist_wide)
    train_pair(hist_wide, hist_wide_dir, tr, updates=UPDATES, use_history=True, dir_labels=labels_tr,
               lam_u=LAM_U, lam_d=LAM_D, batch_size=BATCH, batch_seed=seed + 60, lr=LR_M)

    seed_everything(seed + 19)
    histc_shuf = EAGMHistoryC(stats, hist_len=k, shuffle_c_obs_hist=True)
    histc_shuf_dir = clone_model(histc_shuf)
    train_pair(histc_shuf, histc_shuf_dir, tr, updates=UPDATES, use_history=True, dir_labels=labels_tr,
               lam_u=LAM_U, lam_d=LAM_D, batch_size=BATCH, batch_seed=seed + 70, lr=LR_M)

    models_main = {
        "mc": (mc, False, True),
        "eagm_base": (base, False, False),
        "eagm_dir_matched": (dir_m, False, False),
        "eagm_hist": (hist, True, False),
        "eagm_hist_dir": (hist_dir, True, False),
        "eagm_histc_dir": (histc_dir, True, False),
        "eagm_hist_wide_dir": (hist_wide_dir, True, False),
        "eagm_histc_shuf_dir": (histc_shuf_dir, True, False),
    }

    results = {"interpolation": {}, "extrapolation": {}, "k_ablation": {}}
    rows_out = []
    probes = {}

    for name, (m, use_h, is_mc) in models_main.items():
        for pool, rows, labs, starts in [
            ("interpolation", interp_rows, labels_i, interp_starts0),
            ("extrapolation", extrap_rows, labels_e, extrap_starts0),
        ]:
            ev = eval_model(m, rows, labs, train_pol0, train_env, starts, seed,
                            use_h=use_h, is_mc=is_mc, hist_len=k)
            results[pool][name] = {kk: vv for kk, vv in ev.items() if kk != "per_start"}
            for ps in ev["per_start"]:
                rows_out.append(dict(
                    seed=seed, pool=pool, model=name, hist_len=k,
                    episode_id=ps["episode_id"], config_id=ps.get("config_id"),
                    return_=ps["return_"], return_theta0=ps.get("return_theta0"),
                    delta_vs_theta0=ps.get("delta_vs_theta0"),
                ))
            print(
                f"  seed{seed} k{k} {pool[:5]} {name:22s} "
                f"nmse {ev['dynamics_nmse']:.3f} cosF {ev['cosine_full_median']:.3f} "
                f"ret {ev['policy_return_mean']:.6f} fail {ev['fail_rate_vs_theta0']:.2f} "
                f"ΔR {ev['delta_vs_theta0']:.6g}",
                flush=True,
            )
        torch.save({"state": m.state_dict(), "kind": name, "seed": seed, "hist_len": k},
                   out_dir / f"{name}_k{k}.pt")
        if use_h and not is_mc:
            probes[name] = fit_exec_state_probe(m, tr, seed=seed + 99)
            print(f"    probe {name}: mean|corr|={probes[name].get('mean_abs_corr')}", flush=True)

    # --- k ablation: Hist+Dir vs HistC+Dir at k=4,16 ---
    for k_ab in K_ABLATION:
        tr_ab = pack_rows(train_rows, k_ab)
        labels_tr_ab, _ = build_labels(train_env, dir_starts0, thetas, k_ab)
        labels_i_ab, _ = build_labels(train_env, interp_starts0, thetas, k_ab)
        labels_e_ab, _ = build_labels(train_env, extrap_starts0, thetas, k_ab)

        seed_everything(seed + 13 + k_ab)
        h = EAGMHistory(stats, hist_len=k_ab)
        hd = clone_model(h)
        train_pair(h, hd, tr_ab, updates=UPDATES, use_history=True, dir_labels=labels_tr_ab,
                   lam_u=LAM_U, lam_d=LAM_D, batch_size=BATCH, batch_seed=seed + 40 + k_ab, lr=LR_M)

        seed_everything(seed + 15 + k_ab)
        hc = EAGMHistoryC(stats, hist_len=k_ab)
        hcd = clone_model(hc)
        train_pair(hc, hcd, tr_ab, updates=UPDATES, use_history=True, dir_labels=labels_tr_ab,
                   lam_u=LAM_U, lam_d=LAM_D, batch_size=BATCH, batch_seed=seed + 50 + k_ab, lr=LR_M)

        results["k_ablation"][str(k_ab)] = {}
        for name, m in [("eagm_hist_dir", hd), ("eagm_histc_dir", hcd)]:
            results["k_ablation"][str(k_ab)][name] = {}
            for pool, rows, labs, starts in [
                ("interpolation", interp_rows, labels_i_ab, interp_starts0),
                ("extrapolation", extrap_rows, labels_e_ab, extrap_starts0),
            ]:
                ev = eval_model(m, rows, labs, train_pol0, train_env, starts, seed,
                                use_h=True, is_mc=False, hist_len=k_ab)
                results["k_ablation"][str(k_ab)][name][pool] = {
                    kk: vv for kk, vv in ev.items() if kk != "per_start"
                }
                print(
                    f"  seed{seed} k{k_ab} {pool[:5]} {name:22s} "
                    f"ret {ev['policy_return_mean']:.6f} fail {ev['fail_rate_vs_theta0']:.2f}",
                    flush=True,
                )
            torch.save({"state": m.state_dict(), "kind": name, "seed": seed, "hist_len": k_ab},
                       out_dir / f"{name}_k{k_ab}.pt")

    def cdiff(pool, a, b):
        return results[pool][a]["policy_return_mean"] - results[pool][b]["policy_return_mean"]

    contrasts = {}
    for pool in ("interpolation", "extrapolation"):
        contrasts[pool] = dict(
            histc_dir_minus_hist_dir=cdiff(pool, "eagm_histc_dir", "eagm_hist_dir"),
            histc_dir_minus_mc=cdiff(pool, "eagm_histc_dir", "mc"),
            histc_dir_minus_hist_wide=cdiff(pool, "eagm_histc_dir", "eagm_hist_wide_dir"),
            histc_dir_minus_histc_shuf=cdiff(pool, "eagm_histc_dir", "eagm_histc_shuf_dir"),
            hist_dir_minus_mc=cdiff(pool, "eagm_hist_dir", "mc"),
            dir_minus_base=cdiff(pool, "eagm_dir_matched", "eagm_base"),
        )

    budget = stage5e_budget_breakdown(
        n_train_eps=24, n_val_eps=6, n_iid_hold_eps=6, horizon=200,
        dir_label_train_steps=steps_tr, ood_fd_diagnostic_steps=steps_i + steps_e,
    )
    payload = dict(
        seed=seed,
        stage="6B.2",
        k_main=K_MAIN,
        matched_encoder_hidden=matched_h,
        param_counts={n: count_params(m) for n, (m, _, _) in models_main.items()},
        c_obs_definition=dict(
            dim=29,
            fields=["target[0:14]", "filter[14:28]", "delay[28]"],
            note="Observable only. Privileged B state never used as model input.",
        ),
        sim_budget=budget,
        results=results,
        probes=probes,
        contrasts=contrasts,
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
        note="Seed-level paired bootstrap (n=5).",
    )


def main():
    root = EAGM_ROOT / "runs" / "stage6b2_histc"
    root.mkdir(parents=True, exist_ok=False)
    all_seed = []
    t0 = time.time()
    for seed in SEEDS:
        print(f"==== Stage6B.2 HistC seed {seed} ====", flush=True)
        sd = root / f"seed{seed}"
        sd.mkdir(exist_ok=False)
        all_seed.append(run_seed(seed, sd))
        (root / "partial.json").write_text(json.dumps(all_seed, indent=2, default=str))

    keys = list(all_seed[0]["contrasts"]["interpolation"].keys())
    paired = {
        pool: {k: paired_ci([s["contrasts"][pool][k] for s in all_seed]) for k in keys}
        for pool in ("interpolation", "extrapolation")
    }

    # Aggregate ΔR stats for Dir/Base/HistC
    def mean_field(pool, model, field):
        return float(np.mean([s["results"][pool][model][field] for s in all_seed]))

    pool_summary = {
        pool: {
            name: dict(
                return_mean=mean_field(pool, name, "policy_return_mean"),
                delta_r_mean=mean_field(pool, name, "delta_vs_theta0"),
                fail_rate_mean=mean_field(pool, name, "fail_rate_vs_theta0"),
                theta_sat_mean=mean_field(pool, name, "theta_param_sat_frac"),
                nmse_mean=mean_field(pool, name, "dynamics_nmse"),
                cos_full_mean=mean_field(pool, name, "cosine_full_median"),
                cos_stable_mean=mean_field(pool, name, "cosine_stable_median"),
                cmd_mae_mean=mean_field(pool, name, "cmd_mae") if name != "mc" else None,
            )
            for name in all_seed[0]["results"][pool]
        }
        for pool in ("interpolation", "extrapolation")
    }

    summary = dict(
        stage="6B.2",
        seeds=SEEDS,
        k_main=K_MAIN,
        k_ablation=K_ABLATION,
        paired=paired,
        pool_summary=pool_summary,
        per_seed=[{"seed": s["seed"], "contrasts": s["contrasts"],
                   "probes": {k: {"mean_abs_corr": v.get("mean_abs_corr"), "corr": v.get("corr")}
                              for k, v in s.get("probes", {}).items()},
                   "param_counts": s["param_counts"]} for s in all_seed],
        seed_results=all_seed,
        wall_seconds=time.time() - t0,
        decision_note=(
            "STOP for human review before Stage 6B.3 ExecState or Stage 6C. "
            "Primary: HistC+Dir − Hist+Dir and HistC+Dir − Mc; "
            "controls: param-matched Hist+Dir and shuffled c_obs."
        ),
        failrate_caveat=(
            "Do not compare FailRate levels across Stage 6A corrected (seeds 5–9) and "
            "Stage 6B.1/6B.2 (seeds 10–14) as a Stage progression. Differences mix new seeds "
            "and (if any) evaluation-sample changes. Within-stage Dir vs Base contrasts are the claim."
        ),
    )
    (root / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(paired, indent=2), flush=True)
    print("wrote", root, "wall", round(summary["wall_seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
