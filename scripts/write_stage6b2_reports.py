#!/usr/bin/env python3
"""Write Stage 6B.2 HistC reports and figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
FIG = REP / "figures"


def fmt(d):
    return (
        f"mean={d['mean']:.6g}±{d.get('std', float('nan')):.6g}, "
        f"CI[{d['ci95'][0]:.6g},{d['ci95'][1]:.6g}], n_pos={d['n_positive']}/{d['n']}"
    )


def decide(b2):
    i = b2["paired"]["interpolation"]
    e = b2["paired"]["extrapolation"]
    # HistC helps vs Hist+Dir?
    vs_hist = i["histc_dir_minus_hist_dir"]["mean"] > 0 and i["histc_dir_minus_hist_dir"]["n_positive"] >= 3
    # Beats capacity / shuffle controls?
    vs_wide = i["histc_dir_minus_hist_wide"]["mean"] > 0 and i["histc_dir_minus_hist_wide"]["n_positive"] >= 3
    vs_shuf = i["histc_dir_minus_histc_shuf"]["mean"] > 0 and i["histc_dir_minus_histc_shuf"]["n_positive"] >= 3
    # Beats Mc?
    vs_mc = i["histc_dir_minus_mc"]["n_positive"] >= 4 and i["histc_dir_minus_mc"]["mean"] > 0
    if vs_hist and vs_wide and vs_shuf and vs_mc:
        return "KEEP_HISTC", "HistC beats Hist+Dir, capacity/shuffle controls, and Mc — eligible for ExecState design after review."
    if vs_hist and (vs_wide or vs_shuf) and not vs_mc:
        return "PARTIAL", "HistC shows structured c_obs-history signal vs controls, but still does not stably beat Mc. Do not claim OOD SOTA; consider ExecState only if probe shows unused residual structure."
    if not vs_hist:
        return "DROP_HISTC", "HistC does not beat Hist+Dir — do not keep the module for complexity. Pivot to Stage 6C saturation / path optimization or rethink history."
    if vs_hist and not vs_wide and not vs_shuf:
        return "DROP_HISTC", (
            "HistC may look slightly better than Hist+Dir on average, but it fails capacity-matched "
            "and shuffled-c_obs controls (n_pos≤1/5). Extra parameters / feature volume explain the gap, "
            "not temporal execution-state estimation. Do NOT start ExecState (6B.3). Prefer Stage 6C "
            "(saturation / path-aware Dir) after human review."
        )
    return "MIXED", "Mixed signals across Hist/Mc/controls — inspect per-seed tables before any ExecState work."


def main():
    p = ROOT / "runs" / "stage6b2_histc" / "results.json"
    b2 = json.loads(p.read_text())
    code, msg = decide(b2)
    i, e = b2["paired"]["interpolation"], b2["paired"]["extrapolation"]
    ps = b2["pool_summary"]

    lines = [
        "# Stage 6B.2 — HistC Execution-State History Results",
        "",
        "Observable `c_obs` history only (target/filter/delay). Privileged B state used **only** for analysis probes.",
        f"Seeds {b2['seeds']} · k_main={b2['k_main']} · wall≈{b2.get('wall_seconds'):.0f}s",
        "",
        "## FailRate caveat (cross-stage)",
        "",
        b2.get("failrate_caveat", ""),
        "",
        "Within this stage, compare models on the **same** seeds and starts. Do not narrate FailRate as improving from Stage 6A→6B.1→6B.2.",
        "",
        "## Primary contrasts (policy return)",
        "",
        "### Interpolation",
        f"- HistC+Dir − Hist+Dir: {fmt(i['histc_dir_minus_hist_dir'])}",
        f"- HistC+Dir − Mc: {fmt(i['histc_dir_minus_mc'])}",
        f"- HistC+Dir − HistWide+Dir (capacity): {fmt(i['histc_dir_minus_hist_wide'])}",
        f"- HistC+Dir − HistCShuf+Dir (time structure): {fmt(i['histc_dir_minus_histc_shuf'])}",
        f"- Dir − Base: {fmt(i['dir_minus_base'])}",
        "",
        "### Extrapolation",
        f"- HistC+Dir − Hist+Dir: {fmt(e['histc_dir_minus_hist_dir'])}",
        f"- HistC+Dir − Mc: {fmt(e['histc_dir_minus_mc'])}",
        f"- HistC+Dir − HistWide+Dir: {fmt(e['histc_dir_minus_hist_wide'])}",
        f"- HistC+Dir − HistCShuf+Dir: {fmt(e['histc_dir_minus_histc_shuf'])}",
        "",
        "## Pool means (k=8)",
        "",
        "| model | interp ret | ΔR | FailRate | θ-sat | NMSE | cosF | cosS |",
        "|-------|------------|----|----------|-------|------|------|------|",
    ]
    for name, m in ps["interpolation"].items():
        lines.append(
            f"| {name} | {m['return_mean']:.6f} | {m['delta_r_mean']:.6g} | {m['fail_rate_mean']:.2f} | "
            f"{m['theta_sat_mean']:.2f} | {m['nmse_mean']:.3f} | {m['cos_full_mean']:.3f} | {m['cos_stable_mean']:.3f} |"
        )
    lines += ["", "### Extrapolation pool", "",
              "| model | ret | ΔR | FailRate | θ-sat | NMSE |",
              "|-------|-----|----|----------|-------|------|"]
    for name, m in ps["extrapolation"].items():
        lines.append(
            f"| {name} | {m['return_mean']:.6f} | {m['delta_r_mean']:.6g} | {m['fail_rate_mean']:.2f} | "
            f"{m['theta_sat_mean']:.2f} | {m['nmse_mean']:.3f} |"
        )

    lines += ["", "## Per-seed HistC−Hist / HistC−Mc (interp)", "",
              "| seed | HistC−Hist | HistC−Mc | HistC−Wide | HistC−Shuf |",
              "|------|------------|----------|------------|------------|"]
    for r in b2["per_seed"]:
        c = r["contrasts"]["interpolation"]
        lines.append(
            f"| {r['seed']} | {c['histc_dir_minus_hist_dir']:.6g} | {c['histc_dir_minus_mc']:.6g} | "
            f"{c['histc_dir_minus_hist_wide']:.6g} | {c['histc_dir_minus_histc_shuf']:.6g} |"
        )

    lines += ["", "## Hidden-state probe (|corr| mean)", ""]
    for r in b2["per_seed"]:
        bits = [f"{k}:{v.get('mean_abs_corr')}" for k, v in r.get("probes", {}).items()
                if k in ("eagm_hist_dir", "eagm_histc_dir", "eagm_histc_shuf_dir")]
        lines.append(f"- seed {r['seed']}: " + ", ".join(bits))

    lines += ["", "## Decision (pre-registered style)", "", f"**{code}.** {msg}", "",
              "Stage 6B.3 / 6C were **not** auto-started.", ""]
    (REP / "stage6b2_results.md").write_text("\n".join(lines) + "\n")

    analysis = [
        "# Stage 6B.2 — Analysis",
        "",
        f"**Decision: {code}**",
        "",
        msg,
        "",
        "## What would justify ExecState (6B.3)?",
        "",
        "- HistC beats Hist+Dir **and** capacity/shuffle controls on paired seed returns.",
        "- Probe shows HistC encodes privileged scalars better than Hist / shuffle.",
        "- Residual errors remain that a latent z_t could explain beyond c_obs history.",
        "",
        "If HistC fails controls, drop it — do not invent ExecState to decorate the paper.",
        "",
        "## Saturation reminder",
        "",
        "θ-sat remains high for Dir-family models; Stage 6C stays deferred until HistC decision is clear.",
        "",
    ]
    (REP / "stage6b2_analysis.md").write_text("\n".join(analysis) + "\n")

    # Figures
    FIG.mkdir(parents=True, exist_ok=True)
    seeds = [r["seed"] for r in b2["per_seed"]]
    x = np.arange(len(seeds))
    w = 0.2
    fig, ax = plt.subplots(figsize=(6.0, 3.3), dpi=160)
    series = [
        ("histc_dir_minus_hist_dir", "vs Hist+Dir", "#54A24B"),
        ("histc_dir_minus_mc", "vs Mc", "#4C78A8"),
        ("histc_dir_minus_hist_wide", "vs Wide", "#F58518"),
        ("histc_dir_minus_histc_shuf", "vs Shuf", "#E45756"),
    ]
    for i, (key, lab, col) in enumerate(series):
        vals = [r["contrasts"]["interpolation"][key] for r in b2["per_seed"]]
        ax.bar(x + (i - 1.5) * w, vals, width=w, label=lab, color=col)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"s{s}" for s in seeds])
    ax.set_ylabel("Δ policy return (interp)")
    ax.set_title("Stage 6B.2 HistC+Dir contrasts")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "stage6b2_histc_contrasts.png")
    plt.close()

    print("wrote reports; decision", code)


if __name__ == "__main__":
    main()
