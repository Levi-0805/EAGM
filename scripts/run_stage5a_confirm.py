#!/usr/bin/env python3
"""Stage 5A: independent confirmatory seeds [5,6,7,8,9] on frozen Stage-4 protocol.

Primary contrast: EAGM-Dir-Matched − EAGM-Base (policy return).
Does not overwrite Stage 4 runs. Does not retune hyperparameters.
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

FROZEN = json.loads((EAGM_ROOT / "configs" / "stage5_frozen.json").read_text())
UPDATES = int(FROZEN["world_model_updates"])
H_IMAG = int(FROZEN["imagined_horizon"])
H_EVAL = int(FROZEN["eval_horizon"])
HIST = int(FROZEN["hist_len"])
SEEDS = [5, 6, 7, 8, 9]
DTYPE = torch.float32
STAGE4_SEEDS = [0, 1, 2, 3, 4]


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
        note="n=5 seed percentile bootstrap is unstable; report all seeds.",
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

    seed_everything(seed + 7)
    mc = McWorld(stats)
    train_mc(mc, tr, UPDATES, batch_seed=seed + 20)

    seed_everything(seed + 11)
    base = EAGMBase(stats)
    dir_m = clone_model(base)
    train_info = train_pair(
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
    train_pol = pick_starts(train_rows, delay=None, max_per_episode=1, max_total=12)
    th0 = eval_physical(env, np.zeros(28), test_starts, horizon=H_EVAL)
    # Count physical eval steps (test-only diagnostics separately from train budget)
    n_eval_starts = len(test_starts)
    phys_eval_steps_per_model = n_eval_starts * H_EVAL  # one θ_final eval; θ0 shared

    rows_out = []
    summary = {}
    theta_traces = {}
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
        pol = policy_optimize(
            m, train_pol, use_history=use_h, is_mc=is_mc,
            updates=int(FROZEN["policy_updates"]), lr=float(FROZEN["lr_policy"]), seed=seed + 50,
        )
        phys = eval_physical(env, pol["theta"], test_starts, horizon=H_EVAL)
        summary[name] = dict(
            dynamics_nmse=nmse,
            cmd_mae=cmd,
            cosine_median=float(np.median(cos)),
            cosine_mean=float(np.mean(cos)),
            policy_return_mean=phys["mean"],
            delta_vs_theta0=phys["mean"] - th0["mean"],
            theta_l2_delta=float(np.linalg.norm(pol["theta"] - pol["theta0"])),
            sat_frac_final=pol["curve"][-1]["sat_frac"] if pol["curve"] else None,
        )
        theta_traces[name] = dict(
            theta0=pol["theta0"].tolist(),
            theta_final=pol["theta"].tolist(),
            curve=[{"update": c["update"], "imagined_return": c["imagined_return"],
                    "grad_norm": c["grad_norm"], "theta_norm": c["theta_norm"],
                    "sat_frac": c["sat_frac"]} for c in pol["curve"]],
            # keep sparse θ checkpoints for 5D
            theta_checkpoints={
                "u0": pol["theta0"].tolist(),
                "u25": pol["curve"][24]["theta"] if len(pol["curve"]) >= 25 else None,
                "u50": pol["theta"].tolist(),
            },
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

    sim_budget = dict(
        dynamics_collect_steps=24 * 200,
        dir_label_train_steps=steps_tr,
        dir_label_test_diagnostic_steps=steps_te,
        train_interaction_budget=24 * 200 + steps_tr,  # test FD excluded
        phys_eval_theta0_steps=n_eval_starts * H_EVAL,
        phys_eval_per_model_steps=phys_eval_steps_per_model,
        note="Train budget = dynamics collect + train dir labels. Test FD is diagnostic-only.",
    )
    payload = dict(
        seed=seed,
        set_label="stage5a_confirmatory",
        frozen_from=FROZEN["frozen_from_stage4_commit"],
        split=dict(train=split.train_eps, val=split.val_eps, test=split.test_eps),
        sim_budget=sim_budget,
        dir_steps_train=steps_tr,
        dir_steps_test_diagnostic=steps_te,
        theta0_return=th0["mean"],
        results=summary,
        train_pair_info=dict(
            dir_applications=train_info["dir_applications"],
            dir_skips=train_info["dir_skips"],
            batch_seed=train_info["batch_seed"],
        ),
        wall_seconds=time.time() - t0,
    )
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "theta_traces.json").write_text(json.dumps(theta_traces, indent=2))
    with (out_dir / "per_start.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    return payload


def summarize(all_seed, label):
    names = ["eagm_base", "eagm_dir_matched", "eagm_hist", "eagm_hist_dir_matched"]
    stats = {}
    for name in names:
        dret = [s["results"][name]["policy_return_mean"] - s["results"]["mc"]["policy_return_mean"] for s in all_seed]
        dcos = [s["results"][name]["cosine_median"] - s["results"]["mc"]["cosine_median"] for s in all_seed]
        stats[f"{name}_vs_mc"] = dict(return_=paired_ci(dret), cosine=paired_ci(dcos))
    stats["dir_vs_base"] = paired_ci(
        [
            s["results"]["eagm_dir_matched"]["policy_return_mean"]
            - s["results"]["eagm_base"]["policy_return_mean"]
            for s in all_seed
        ]
    )
    stats["hist_dir_vs_hist"] = paired_ci(
        [
            s["results"]["eagm_hist_dir_matched"]["policy_return_mean"]
            - s["results"]["eagm_hist"]["policy_return_mean"]
            for s in all_seed
        ]
    )
    per_seed = [
        dict(
            seed=s["seed"],
            mc=s["results"]["mc"]["policy_return_mean"],
            base=s["results"]["eagm_base"]["policy_return_mean"],
            dir=s["results"]["eagm_dir_matched"]["policy_return_mean"],
            dir_minus_base=s["results"]["eagm_dir_matched"]["policy_return_mean"]
            - s["results"]["eagm_base"]["policy_return_mean"],
            dir_minus_mc=s["results"]["eagm_dir_matched"]["policy_return_mean"]
            - s["results"]["mc"]["policy_return_mean"],
            cos_mc=s["results"]["mc"]["cosine_median"],
            cos_base=s["results"]["eagm_base"]["cosine_median"],
            cos_dir=s["results"]["eagm_dir_matched"]["cosine_median"],
        )
        for s in all_seed
    ]
    return dict(set_label=label, paired_seed_stats=stats, per_seed=per_seed)


def load_stage4_exploratory():
    p = EAGM_ROOT / "runs" / "stage4_fixedB_five_seed" / "results.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    # Re-wrap seed_results to match summarize expectations
    return data.get("seed_results")


def main():
    root = EAGM_ROOT / "runs" / "stage5a_confirm"
    root.mkdir(parents=True, exist_ok=False)
    (root / "frozen_config.json").write_text(json.dumps(FROZEN, indent=2))
    all_seed = []
    t0 = time.time()
    for seed in SEEDS:
        print(f"==== Stage5A confirmatory seed {seed} ====", flush=True)
        sd = root / f"seed{seed}"
        sd.mkdir(exist_ok=False)
        all_seed.append(run_seed(seed, sd))
        (root / "partial.json").write_text(json.dumps(all_seed, indent=2, default=str))

    conf = summarize(all_seed, "stage5a_confirmatory")
    explor_raw = load_stage4_exploratory()
    explor = summarize(explor_raw, "stage4_exploratory") if explor_raw else None

    # Descriptive merge (not confirmatory claim)
    merged = None
    if explor_raw:
        merged_seeds = explor_raw + all_seed
        merged = summarize(merged_seeds, "exploratory_plus_confirmatory_descriptive_only")
        merged["warning"] = (
            "Merged stats mix exploratory (seeds 0-4, seed0 used in Stage3.2 selection) "
            "with confirmatory (5-9). Do NOT use merged alone to claim confirmation."
        )

    out = dict(
        stage="5A",
        confirmatory_seeds=SEEDS,
        exploratory_seeds=STAGE4_SEEDS,
        frozen_from=FROZEN["frozen_from_stage4_commit"],
        confirmatory=conf,
        exploratory=explor,
        merged_descriptive=merged,
        seed_results=all_seed,
        wall_seconds=time.time() - t0,
        primary_metric="eagm_dir_matched - eagm_base policy_return_mean",
    )
    (root / "results.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(conf["paired_seed_stats"]["dir_vs_base"], indent=2), flush=True)
    print("wrote", root, "wall", round(out["wall_seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
