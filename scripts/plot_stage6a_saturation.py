#!/usr/bin/env python3
"""Stage 6A: θ-parameter saturation diagnostic from Stage 5D path dumps."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [5, 6, 7, 8, 9]


def main():
    rows = []
    for seed in SEEDS:
        p = ROOT / "runs" / "stage5d_gradpath" / f"seed{seed}"
        b = json.loads((p / "eagm_base_path.json").read_text())
        d = json.loads((p / "eagm_dir_matched_path.json").read_text())
        for label in ("theta0", "theta_mid", "theta_final"):
            rows.append(dict(
                seed=seed, label=label,
                base_sat=b["path"][label]["sat_frac"],
                dir_sat=d["path"][label]["sat_frac"],
                base_cos=b["path"][label]["cosine"],
                dir_cos=d["path"][label]["cosine"],
            ))

    # Bar: final θ param saturation
    base_f = [r["base_sat"] for r in rows if r["label"] == "theta_final"]
    dir_f = [r["dir_sat"] for r in rows if r["label"] == "theta_final"]
    out = ROOT / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=160)
    x = np.arange(2)
    means = [float(np.mean(base_f)), float(np.mean(dir_f))]
    ax.bar(x, means, color=["#F58518", "#54A24B"], width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(["Base", "Dir-Matched"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Final θ param saturation rate")
    ax.set_title("Stage 5D · mean over seeds 5–9")
    for i, v in enumerate(means):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "stage6a_theta_saturation.png")
    plt.close()

    # Scatter: final sat vs final cosine gap
    fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=160)
    for seed in SEEDS:
        rb = next(r for r in rows if r["seed"] == seed and r["label"] == "theta_final")
        ax.scatter(rb["base_sat"], rb["base_cos"], c="#F58518", marker="o", s=40)
        ax.scatter(rb["dir_sat"], rb["dir_cos"], c="#54A24B", marker="s", s=40)
    ax.set_xlabel("θ param saturation at θ_final")
    ax.set_ylabel("Path gradient cosine at θ_final")
    ax.set_title("Saturation vs path cosine (orange=Base, green=Dir)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "stage6a_sat_vs_cosine.png")
    plt.close()

    summary = dict(
        mean_final_sat_base=float(np.mean(base_f)),
        mean_final_sat_dir=float(np.mean(dir_f)),
        per_seed=[{
            "seed": s,
            "base_final_sat": next(r["base_sat"] for r in rows if r["seed"] == s and r["label"] == "theta_final"),
            "dir_final_sat": next(r["dir_sat"] for r in rows if r["seed"] == s and r["label"] == "theta_final"),
            "base_final_cos": next(r["base_cos"] for r in rows if r["seed"] == s and r["label"] == "theta_final"),
            "dir_final_cos": next(r["dir_cos"] for r in rows if r["seed"] == s and r["label"] == "theta_final"),
        } for s in SEEDS],
        hypothesis=(
            "Dir ends with higher θ-bound saturation (~92%) than Base (~74%); "
            "co-occurs with weakened θ_final cosine advantage. Causal test deferred to Stage 6C."
        ),
    )
    (ROOT / "runs" / "stage6a_diagnostics").mkdir(parents=True, exist_ok=True)
    (ROOT / "runs" / "stage6a_diagnostics" / "saturation_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))
    print("wrote figures to", out)


if __name__ == "__main__":
    main()
