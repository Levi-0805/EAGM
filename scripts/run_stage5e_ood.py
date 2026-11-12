#!/usr/bin/env python3
"""Stage 5E: train-time Controller B randomization + frozen interp/extrap tests.

Gated: only run after Stage 5A–5C pass. Prefer History + Dir; also report Base/Mc.
Does not add CF loss. Interpolation and extrapolation reported separately.
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
from training.dataset import fit_stats, flatten_split, make_split, pick_starts
from training.gradients import sim_grad_masked
from training.paired_train import clone_model, train_mc, train_pair
from training.policy_opt import eval_physical, policy_optimize
from training.rollout import imagined_eagm, imagined_mc
from training.seeds import seed_everything

FROZEN = json.loads((EAGM_ROOT / "configs" / "stage5_frozen.json").read_text())
UPDATES = int(FROZEN["world_model_updates"])
H_IMAG = int(FROZEN["imagined_horizon"])
H_EVAL = int(FROZEN["eval_horizon"])
HIST = int(FROZEN["hist_len"])
SEEDS = [5, 6, 7, 8, 9]
DTYPE = torch.float32

# Train randomization ranges (same as ControllerBConfig defaults)
TRAIN_RANGES = dict(
    gain_range=(0.92, 1.08),
    extra_lag_choices=(0, 1, 2),
    rate_limit_range=(0.03, 0.08),
    smooth_beta_range=(0.0, 0.35),
)

# Frozen held-out configs — NEVER used to retune
INTERP_CONFIGS = [
    # inside training ranges; specific combinations
    dict(gain=0.95, extra_lag=0, rate_limit=0.035, smooth_beta=0.10, id="interp_0"),
    dict(gain=1.02, extra_lag=1, rate_limit=0.055, smooth_beta=0.20, id="interp_1"),
    dict(gain=1.06, extra_lag=2, rate_limit=0.070, smooth_beta=0.05, id="interp_2"),
    dict(gain=0.98, extra_lag=2, rate_limit=0.040, smooth_beta=0.30, id="interp_3"),
    dict(gain=1.04, extra_lag=0, rate_limit=0.065, smooth_beta=0.15, id="interp_4"),
    dict(gain=0.93, extra_lag=1, rate_limit=0.050, smooth_beta=0.25, id="interp_5"),
]
EXTRAP_CONFIGS = [
    # outside training ranges (reasonable)
    dict(gain=0.85, extra_lag=0, rate_limit=0.045, smooth_beta=0.20, id="extrap_gain_lo"),
    dict(gain=1.15, extra_lag=1, rate_limit=0.045, smooth_beta=0.20, id="extrap_gain_hi"),
    dict(gain=1.00, extra_lag=3, rate_limit=0.045, smooth_beta=0.20, id="extrap_lag_hi"),
    dict(gain=1.00, extra_lag=1, rate_limit=0.020, smooth_beta=0.20, id="extrap_rate_lo"),
    dict(gain=1.00, extra_lag=1, rate_limit=0.100, smooth_beta=0.20, id="extrap_rate_hi"),
    dict(gain=1.00, extra_lag=1, rate_limit=0.045, smooth_beta=0.50, id="extrap_smooth_hi"),
]


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def pack_rows(rows):
    keys = ["x", "y", "a", "c", "c_obs", "u_exec", "x_hist", "a_hist"]
    return {k: torch.as_tensor(np.stack([r[k] for r in rows]), dtype=DTYPE) for k in keys}


def collect_randomized(env, n_episodes, horizon, seed, hist_len=HIST):
    """Like dataset.collect_trajectories but records per-episode b_active."""
    from training.dataset import HIST_LEN_DEFAULT  # noqa: F401
    rng = np.random.default_rng(seed)
    episodes = []
    for ep in range(n_episodes):
        delay = int(ep % 3)
        x = env.reset(seed=seed * 1000 + ep, delay=delay, jitter=0.01)
        b_active = dict(env._active)
        config_id = (
            f"B_rand_g{b_active['gain']:.3f}_L{b_active['extra_lag']}"
            f"_r{b_active['rate_limit']:.3f}_s{b_active['smooth_beta']:.3f}"
        )
        center = x[:14].copy()
        tt = np.arange(horizon + 16) * float(env.control_dt)
        refs = center[None, :] + 0.08 * np.sin(
            2 * np.pi * (0.1 + 0.02 * (np.arange(14) % 5))[None, :] * tt[:, None]
        )
        prev_a = np.zeros(14, dtype=np.float64)
        x_buf = [x.copy()] * hist_len
        a_buf = [np.zeros(14, dtype=np.float64)] * hist_len
        h_stack = np.zeros((8, 42), dtype=np.float64)
        h_stack[:, :28] = x
        rows = []
        for t_i in range(horizon):
            c_full = env.control_state().copy()
            c_obs = env.observed_control_state().copy()
            snap = env.snapshot()
            a = np.clip(
                0.55 * (refs[t_i] - c_full[:14]) + rng.normal(0, 0.003, 14),
                -0.03, 0.03,
            )
            x_hist = np.stack(x_buf[-hist_len:], axis=0)
            a_hist = np.stack(a_buf[-hist_len:], axis=0)
            start = dict(
                x=x.copy(), c=c_full, c_obs=c_obs, h=h_stack.copy(), prev=prev_a.copy(),
                refs=refs[t_i : t_i + 16].copy(), snapshot=snap, delay=delay,
                episode_id=ep, frame=t_i, config_id=config_id, b_active=b_active,
                x_hist=x_hist.copy(), a_hist=a_hist.copy(),
            )
            x_next = env.step(a)
            u_exec = env.last_true_command.copy()
            rows.append(dict(
                x=x.copy(), y=x_next.copy(), a=a.copy(), c=c_full, c_obs=c_obs,
                u_exec=u_exec, delay=delay, episode_id=ep, frame=t_i,
                config_id=config_id, b_active=b_active, start=start,
                x_hist=x_hist, a_hist=a_hist,
            ))
            h_stack = np.concatenate([h_stack[1:], np.concatenate([x, a])[None]], 0)
            x_buf.append(x_next.copy())
            a_buf.append(a.copy())
            prev_a = a.copy()
            x = x_next
        episodes.append(dict(
            episode_id=ep, delay=delay, config_id=config_id, b_active=b_active,
            rows=rows, n_steps=len(rows),
        ))
    return episodes


def collect_fixed_configs(configs, horizon, seed_base, hist_len=HIST, episodes_per_cfg=2):
    """Collect frozen test trajectories for each config (interp or extrap)."""
    episodes = []
    ep_id = 0
    for cfg in configs:
        env = PartialUnknownArmEnv(ControllerBConfig(
            gain=cfg["gain"], extra_lag=cfg["extra_lag"],
            rate_limit=cfg["rate_limit"], smooth_beta=cfg["smooth_beta"],
            episode_randomize=False,
        ))
        for k in range(episodes_per_cfg):
            delay = int((ep_id) % 3)
            rng = np.random.default_rng(seed_base + ep_id * 17)
            x = env.reset(seed=seed_base * 1000 + ep_id, delay=delay, jitter=0.01)
            b_active = dict(env._active)
            config_id = cfg["id"]
            center = x[:14].copy()
            tt = np.arange(horizon + 16) * float(env.control_dt)
            refs = center[None, :] + 0.08 * np.sin(
                2 * np.pi * (0.1 + 0.02 * (np.arange(14) % 5))[None, :] * tt[:, None]
            )
            prev_a = np.zeros(14, dtype=np.float64)
            x_buf = [x.copy()] * hist_len
            a_buf = [np.zeros(14, dtype=np.float64)] * hist_len
            h_stack = np.zeros((8, 42), dtype=np.float64)
            h_stack[:, :28] = x
            rows = []
            for t_i in range(horizon):
                c_full = env.control_state().copy()
                c_obs = env.observed_control_state().copy()
                snap = env.snapshot()
                a = np.clip(
                    0.55 * (refs[t_i] - c_full[:14]) + rng.normal(0, 0.003, 14),
                    -0.03, 0.03,
                )
                x_hist = np.stack(x_buf[-hist_len:], axis=0)
                a_hist = np.stack(a_buf[-hist_len:], axis=0)
                start = dict(
                    x=x.copy(), c=c_full, c_obs=c_obs, h=h_stack.copy(), prev=prev_a.copy(),
                    refs=refs[t_i : t_i + 16].copy(), snapshot=snap, delay=delay,
                    episode_id=ep_id, frame=t_i, config_id=config_id, b_active=b_active,
                    x_hist=x_hist.copy(), a_hist=a_hist.copy(),
                )
                x_next = env.step(a)
                u_exec = env.last_true_command.copy()
                rows.append(dict(
                    x=x.copy(), y=x_next.copy(), a=a.copy(), c=c_full, c_obs=c_obs,
                    u_exec=u_exec, delay=delay, episode_id=ep_id, frame=t_i,
                    config_id=config_id, b_active=b_active, start=start,
                    x_hist=x_hist, a_hist=a_hist,
                ))
                h_stack = np.concatenate([h_stack[1:], np.concatenate([x, a])[None]], 0)
                x_buf.append(x_next.copy())
                a_buf.append(a.copy())
                prev_a = a.copy()
                x = x_next
            episodes.append(dict(
                episode_id=ep_id, delay=delay, config_id=config_id, b_active=b_active,
                rows=rows, n_steps=len(rows), frozen_cfg=cfg,
            ))
            ep_id += 1
    return episodes


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
    cos = [cosine(model_grad(model, lab["start"], lab["theta"], use_h, is_mc), lab["g"]) for lab in labels]
    pol = policy_optimize(
        model, train_pol, use_history=use_h, is_mc=is_mc,
        updates=int(FROZEN["policy_updates"]), lr=float(FROZEN["lr_policy"]), seed=seed + 50,
    )
    # Physical eval must use an env that can restore snapshots from test starts
    # (snapshots embed B state). Use a randomize-capable env shell.
    phys = eval_physical(env_eval, pol["theta"], test_starts, horizon=H_EVAL)
    th0 = eval_physical(env_eval, np.zeros(28), test_starts, horizon=H_EVAL)
    # Per-config failure: return worse than θ0 by > 0 (cost higher)
    by_cfg = {}
    for ps in phys["per_start"]:
        cid = ps.get("config_id") or "unk"
        by_cfg.setdefault(cid, []).append(ps["return_"])
    cfg_stats = {
        cid: dict(mean=float(np.mean(v)), n=len(v),
                  fail_frac=float(np.mean(np.asarray(v) > 0)))  # returns are negative costs-ish; keep raw
        for cid, v in by_cfg.items()
    }
    return dict(
        dynamics_nmse=nmse,
        cmd_mae=cmd,
        cosine_median=float(np.median(cos)) if cos else None,
        policy_return_mean=phys["mean"],
        delta_vs_theta0=phys["mean"] - th0["mean"],
        per_config=cfg_stats,
        per_start=phys["per_start"],
    )


def run_seed(seed: int, out_dir: Path) -> dict:
    seed_everything(seed)
    t0 = time.time()
    train_env = PartialUnknownArmEnv(ControllerBConfig(
        episode_randomize=True, **{k: TRAIN_RANGES[k] for k in TRAIN_RANGES},
        gain=1.0, extra_lag=1, rate_limit=0.05, smooth_beta=0.2,
    ))
    # More coverage under randomization
    n_collect = 36
    episodes = collect_randomized(train_env, n_collect, 200, seed)
    split = make_split(n_collect, seed=seed, n_train=24, n_val=6, n_test=6)
    # Internal random-B test split kept for sanity; primary OOD is frozen pools
    train_rows = flatten_split(episodes, split.train_eps)
    val_rows = flatten_split(episodes, split.val_eps)
    stats = fit_stats(train_rows, env_lower=train_env.lower, env_upper=train_env.upper)

    interp_eps = collect_fixed_configs(INTERP_CONFIGS, 200, seed_base=10_000 + seed)
    extrap_eps = collect_fixed_configs(EXTRAP_CONFIGS, 200, seed_base=20_000 + seed)
    interp_rows = [r for ep in interp_eps for r in ep["rows"]]
    extrap_rows = [r for ep in extrap_eps for r in ep["rows"]]

    dir_starts = pick_starts(train_rows, delay=None, max_per_episode=1, max_total=12)
    train_pol = pick_starts(train_rows, delay=None, max_per_episode=1, max_total=12)
    interp_starts = pick_starts(interp_rows, delay=None, max_per_episode=1, max_total=12)
    extrap_starts = pick_starts(extrap_rows, delay=None, max_per_episode=1, max_total=12)

    rng_th = np.random.default_rng(seed + 101)
    thetas = [np.zeros(28)] + [rng_th.normal(0, 0.3, 28).clip(-1, 1) for _ in range(11)]
    labels_tr, steps_tr = build_matched_labels(train_env, dir_starts, thetas)

    # Labels / FD on frozen pools (diagnostic; counted separately)
    # Need env that can restore those snapshots — use train_env shell after restore
    labels_interp, steps_interp = build_matched_labels(train_env, interp_starts, thetas)
    labels_extrap, steps_extrap = build_matched_labels(train_env, extrap_starts, thetas)

    tr = pack_rows(train_rows)

    seed_everything(seed + 7)
    mc = McWorld(stats)
    train_mc(mc, tr, UPDATES, batch_seed=seed + 20)

    seed_everything(seed + 11)
    base = EAGMBase(stats)
    dir_m = clone_model(base)
    train_pair(
        base, dir_m, tr, updates=UPDATES, use_history=False, dir_labels=labels_tr,
        lam_u=float(FROZEN["lam_u"]), lam_d=float(FROZEN["lam_d"]),
        batch_size=int(FROZEN["batch_size"]), batch_seed=seed + 30, lr=float(FROZEN["lr_model"]),
    )

    seed_everything(seed + 13)
    hist = EAGMHistory(stats, hist_len=HIST)
    hist_dir = clone_model(hist)
    train_pair(
        hist, hist_dir, tr, updates=UPDATES, use_history=True, dir_labels=labels_tr,
        lam_u=float(FROZEN["lam_u"]), lam_d=float(FROZEN["lam_d"]),
        batch_size=int(FROZEN["batch_size"]), batch_seed=seed + 40, lr=float(FROZEN["lr_model"]),
    )

    models = {
        "mc": (mc, False, True),
        "eagm_base": (base, False, False),
        "eagm_dir_matched": (dir_m, False, False),
        "eagm_hist": (hist, True, False),
        "eagm_hist_dir_matched": (hist_dir, True, False),
    }

    results = {"interpolation": {}, "extrapolation": {}}
    rows_out = []
    for name, (m, use_h, is_mc) in models.items():
        for pool_name, rows, labels, starts in [
            ("interpolation", interp_rows, labels_interp, interp_starts),
            ("extrapolation", extrap_rows, labels_extrap, extrap_starts),
        ]:
            ev = eval_on_pool(m, rows, labels, train_pol, train_env, starts, seed, use_h, is_mc)
            results[pool_name][name] = {k: v for k, v in ev.items() if k != "per_start"}
            for ps in ev["per_start"]:
                rows_out.append(dict(
                    seed=seed, pool=pool_name, model=name,
                    episode_id=ps["episode_id"], config_id=ps.get("config_id"),
                    return_=ps["return_"],
                ))
            print(
                f"  seed{seed} {pool_name[:5]} {name:22s} "
                f"nmse {results[pool_name][name]['dynamics_nmse']:.3f} "
                f"cos {results[pool_name][name]['cosine_median']:.3f} "
                f"ret {results[pool_name][name]['policy_return_mean']:.6f}",
                flush=True,
            )
        torch.save({"state": m.state_dict(), "kind": name, "seed": seed}, out_dir / f"{name}.pt")

    payload = dict(
        seed=seed,
        stage="5E",
        train_ranges=TRAIN_RANGES,
        interp_configs=INTERP_CONFIGS,
        extrap_configs=EXTRAP_CONFIGS,
        split=dict(train=split.train_eps, val=split.val_eps, iid_hold=split.test_eps),
        n_train_episodes=len(split.train_eps),
        train_b_actives=[ep["b_active"] for ep in episodes if ep["episode_id"] in set(split.train_eps)],
        sim_budget=dict(
            dynamics_train_collect=n_collect * 200,
            dir_label_train_steps=steps_tr,
            interp_fd_diagnostic=steps_interp,
            extrap_fd_diagnostic=steps_extrap,
        ),
        results=results,
        contrasts=dict(
            interp_hist_dir_minus_mc=results["interpolation"]["eagm_hist_dir_matched"]["policy_return_mean"]
            - results["interpolation"]["mc"]["policy_return_mean"],
            interp_hist_dir_minus_hist=results["interpolation"]["eagm_hist_dir_matched"]["policy_return_mean"]
            - results["interpolation"]["eagm_hist"]["policy_return_mean"],
            interp_dir_minus_base=results["interpolation"]["eagm_dir_matched"]["policy_return_mean"]
            - results["interpolation"]["eagm_base"]["policy_return_mean"],
            extrap_hist_dir_minus_mc=results["extrapolation"]["eagm_hist_dir_matched"]["policy_return_mean"]
            - results["extrapolation"]["mc"]["policy_return_mean"],
            extrap_hist_dir_minus_hist=results["extrapolation"]["eagm_hist_dir_matched"]["policy_return_mean"]
            - results["extrapolation"]["eagm_hist"]["policy_return_mean"],
            extrap_dir_minus_base=results["extrapolation"]["eagm_dir_matched"]["policy_return_mean"]
            - results["extrapolation"]["eagm_base"]["policy_return_mean"],
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
        median=float(np.median(diffs)),
        ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        n=int(len(diffs)),
        n_positive=int((diffs > 0).sum()),
        note="n=5 seed percentile bootstrap is unstable.",
    )


def main():
    # Gate check
    a5 = EAGM_ROOT / "runs" / "stage5a_confirm" / "results.json"
    b5 = EAGM_ROOT / "runs" / "stage5b_budget" / "results.json"
    c5 = EAGM_ROOT / "runs" / "stage5c_baselines" / "results.json"
    for p in (a5, b5, c5):
        if not p.exists():
            raise SystemExit(f"Stage 5E gated: missing {p}")
    a = json.loads(a5.read_text())
    b = json.loads(b5.read_text())
    c = json.loads(c5.read_text())
    if a["confirmatory"]["paired_seed_stats"]["dir_vs_base"]["n_positive"] < 3:
        raise SystemExit("Stage 5E gated: 5A Dir-Base not confirmed")
    if b["paired"]["C_minus_B"]["n_positive"] < 3:
        raise SystemExit("Stage 5E gated: 5B budget match failed")
    if c["paired"]["base_minus_aux"]["mean"] <= 0:
        raise SystemExit("Stage 5E gated: 5C structured bottleneck not > aux")

    root = EAGM_ROOT / "runs" / "stage5e_ood"
    root.mkdir(parents=True, exist_ok=False)
    (root / "protocol.json").write_text(json.dumps(dict(
        train_ranges=TRAIN_RANGES, interp=INTERP_CONFIGS, extrap=EXTRAP_CONFIGS,
        note="Interp/extrap frozen before looking at results; do not retune.",
    ), indent=2))

    all_seed = []
    t0 = time.time()
    for seed in SEEDS:
        print(f"==== Stage5E OOD seed {seed} ====", flush=True)
        sd = root / f"seed{seed}"
        sd.mkdir(exist_ok=False)
        all_seed.append(run_seed(seed, sd))
        (root / "partial.json").write_text(json.dumps(all_seed, indent=2, default=str))

    keys = [
        "interp_hist_dir_minus_mc", "interp_hist_dir_minus_hist", "interp_dir_minus_base",
        "extrap_hist_dir_minus_mc", "extrap_hist_dir_minus_hist", "extrap_dir_minus_base",
    ]
    paired = {k: paired_ci([s["contrasts"][k] for s in all_seed]) for k in keys}
    summary = dict(
        stage="5E",
        seeds=SEEDS,
        paired=paired,
        per_seed=[{"seed": s["seed"], **s["contrasts"],
                   "interp_returns": {k: s["results"]["interpolation"][k]["policy_return_mean"]
                                      for k in s["results"]["interpolation"]},
                   "extrap_returns": {k: s["results"]["extrapolation"][k]["policy_return_mean"]
                                      for k in s["results"]["extrapolation"]},
                   "interp_nmse": {k: s["results"]["interpolation"][k]["dynamics_nmse"]
                                   for k in s["results"]["interpolation"]},
                   "extrap_nmse": {k: s["results"]["extrapolation"][k]["dynamics_nmse"]
                                   for k in s["results"]["extrapolation"]},
                   } for s in all_seed],
        seed_results=all_seed,
        wall_seconds=time.time() - t0,
        question="Does train-time B randomization improve unseen-config generalization?",
    )
    (root / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(paired, indent=2), flush=True)
    print("wrote", root, "wall", round(summary["wall_seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
