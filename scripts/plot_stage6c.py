#!/usr/bin/env python3
"""Generate Stage 6C figures from completed run JSONs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures" / "stage6c"
FIG.mkdir(parents=True, exist_ok=True)


def plot_lr_sat():
    d = json.loads((ROOT / "runs/stage6c1_lr_ablation/results.json").read_text())
    lrs = [0.03, 0.015, 0.0075]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=False)
    for ax, name, title in zip(
        axes,
        ["eagm_dir_matched", "eagm_base", "mc"],
        ["EAGM-Dir", "EAGM-Base", "Mc"],
    ):
        sats = [d["summary"]["models"][name][str(lr)]["final_sat"]["mean"] for lr in lrs]
        rets = [d["summary"]["models"][name][str(lr)]["interp_delta_vs_theta0"]["mean"] for lr in lrs]
        ax2 = ax.twinx()
        ax.plot(lrs, sats, "o-", color="#1f4e79", label="θ-sat")
        ax2.plot(lrs, rets, "s--", color="#c45911", label="interp ΔR")
        ax.set_xlabel("policy LR")
        ax.set_title(title)
        ax.set_xticks(lrs)
        ax.invert_xaxis()
    axes[0].set_ylabel("final θ saturation")
    axes[-1].twinx().set_ylabel("interp ΔR vs θ0")  # noqa — decorative
    fig.suptitle("Stage 6C.1: lower LR removes sat but Dir return drops")
    fig.tight_layout()
    fig.savefig(FIG / "stage6c1_lr_sat_vs_return.png", dpi=140)
    plt.close()


def plot_pathdir_paired():
    p = ROOT / "runs/stage6c2_pathdir/results.json"
    if not p.exists():
        print("skip pathdir fig — results missing")
        return
    d = json.loads(p.read_text())
    keys = [
        ("interp_path_minus_random", "Interp Path−Random"),
        ("interp_path_minus_base", "Interp Path−Base"),
        ("extrap_path_minus_random", "Extrap Path−Random"),
        ("extrap_path_minus_base", "Extrap Path−Base"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), sharey=True)
    for ax, (k, title) in zip(axes, keys):
        st = d["paired_seed_stats"][k]
        xs = np.arange(1, st["n"] + 1)
        ax.bar(xs, st["per_seed"], color="#2e75b6")
        ax.axhline(0, color="k", lw=0.8)
        ax.axhline(st["mean"], color="#c45911", ls="--", label=f"mean={st['mean']:.4g}")
        ax.set_title(title)
        ax.set_xlabel("seed rank")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(s) for s in d["seeds"]])
    axes[0].set_ylabel("Δ policy return")
    fig.suptitle(f"Stage 6C.3 paired seeds — decision {d['decision_gate']['decision']}")
    fig.tight_layout()
    fig.savefig(FIG / "stage6c3_paired_returns.png", dpi=140)
    plt.close()


def plot_seed15_curves_if_any():
    # optional: rho/sat from one seed's stored diagnostics in seed results
    p = ROOT / "runs/stage6c2_pathdir/seed15/results.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for name, color in [
        ("eagm_dir_path", "#2e75b6"),
        ("eagm_dir_random", "#c45911"),
        ("eagm_base", "#833c0c"),
        ("mc", "#548235"),
    ]:
        diag = d["results"]["interpolation"][name].get("diagnostics", {})
        sat = diag.get("sat_at_updates", {})
        if not sat:
            continue
        xs = sorted(int(k) for k in sat)
        ys = [sat[str(x)] for x in xs]
        ax.plot(xs, ys, "-o", ms=3, label=name, color=color)
    ax.set_xlabel("policy update")
    ax.set_ylabel("θ saturation")
    ax.set_title("Seed 15 policy saturation trajectories")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "stage6c_seed15_sat_curves.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    plot_lr_sat()
    plot_pathdir_paired()
    plot_seed15_curves_if_any()
    print("figures →", FIG)
