#!/usr/bin/env python3
"""Assemble Stage 6B.1 reports + figures from runs/stage6b1_confirm/results.json."""
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


def fmt_ci(d):
    return (
        f"mean={d['mean']:.6g} ± {d.get('std', float('nan')):.6g}, "
        f"median={d['median']:.6g}, 95% CI [{d['ci95'][0]:.6g}, {d['ci95'][1]:.6g}], "
        f"n_pos={d['n_positive']}/{d['n']}"
    )


def load_stage5e_corrected():
    p = ROOT / "runs" / "stage6a_stage5e_corrected" / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


def decision(b1):
    db = b1["paired_seed_stats"]["interp_dir_minus_base"]
    hd = b1["paired_seed_stats"]["interp_hist_dir_minus_mc"]
    edb = b1["paired_seed_stats"]["extrap_dir_minus_base"]
    ehd = b1["paired_seed_stats"]["extrap_hist_dir_minus_mc"]
    dir_ok = db["n_positive"] >= 4 and db["mean"] > 0 and edb["n_positive"] >= 4
    hist_ok = hd["n_positive"] >= 4 and hd["mean"] > 0 and ehd["n_positive"] >= 4
    if not dir_ok:
        return (
            "B",
            "Dir−Base advantage did NOT stably reproduce on seeds 10–14. "
            "PAUSE HistC/ExecState; diagnose direction labels, FD stability, and θ saturation first.",
        )
    if dir_ok and not hist_ok:
        return (
            "C",
            "Dir−Base reproduced, but Hist+Dir still does not stably beat Mc. "
            "Focus next on execution-state / history representation (Stage 6B.2 HistC), "
            "not more Dir complexity. Do not auto-start without human review.",
        )
    return (
        "A",
        "Dir−Base stable and Hist+Dir shows stronger vs-Mc signal. "
        "Eligible for HistC ablation after human review — do not auto-start 6B.2.",
    )


def write_results(b1, s5):
    lines = [
        "# Stage 6B.1 — Independent Confirmation (seeds 10–14)",
        "",
        "Protocol: Stage 5E legacy (2 episodes/config). Stage 6A metrics. "
        "No model/hyperparameter changes. Compound OOD excluded from main tables.",
        "",
        f"Artifacts: `runs/stage6b1_confirm/` · wall ≈ {b1.get('wall_seconds'):.0f}s",
        "",
        "## Confirmatory seed contrasts (policy return)",
        "",
        "### Interpolation",
        "",
        "| seed | Dir−Base | Dir−Mc | HistDir−Mc | HistDir−Hist |",
        "|------|----------|--------|------------|--------------|",
    ]
    for r in b1["per_seed"]:
        lines.append(
            f"| {r['seed']} | {r['interp_dir_minus_base']:.6g} | {r['interp_dir_minus_mc']:.6g} | "
            f"{r['interp_hist_dir_minus_mc']:.6g} | {r['interp_hist_dir_minus_hist']:.6g} |"
        )
    st = b1["paired_seed_stats"]
    lines += [
        "",
        f"- Dir−Base: {fmt_ci(st['interp_dir_minus_base'])}",
        f"- Dir−Mc: {fmt_ci(st['interp_dir_minus_mc'])}",
        f"- HistDir−Mc: {fmt_ci(st['interp_hist_dir_minus_mc'])}",
        f"- HistDir−Hist: {fmt_ci(st['interp_hist_dir_minus_hist'])}",
        "",
        "### Extrapolation",
        "",
        "| seed | Dir−Base | Dir−Mc | HistDir−Mc | HistDir−Hist |",
        "|------|----------|--------|------------|--------------|",
    ]
    for r in b1["per_seed"]:
        lines.append(
            f"| {r['seed']} | {r['extrap_dir_minus_base']:.6g} | {r['extrap_dir_minus_mc']:.6g} | "
            f"{r['extrap_hist_dir_minus_mc']:.6g} | {r['extrap_hist_dir_minus_hist']:.6g} |"
        )
    lines += [
        "",
        f"- Dir−Base: {fmt_ci(st['extrap_dir_minus_base'])}",
        f"- Dir−Mc: {fmt_ci(st['extrap_dir_minus_mc'])}",
        f"- HistDir−Mc: {fmt_ci(st['extrap_hist_dir_minus_mc'])}",
        f"- HistDir−Hist: {fmt_ci(st['extrap_hist_dir_minus_hist'])}",
        "",
        "## Pool summaries (means over seeds 10–14)",
        "",
    ]
    for pool, title in [("interpolation", "Interpolation"), ("extrapolation", "Extrapolation")]:
        lines += [f"### {title}", "",
                  "| model | return | FailRate | θ-sat | action-clip | NMSE | cos_full | cos_stable |",
                  "|-------|--------|----------|-------|-------------|------|----------|------------|"]
        for name, m in b1["pool_summary"][pool].items():
            lines.append(
                f"| {name} | {m['return_mean']:.6f} | {m['fail_rate_mean']:.2f} | "
                f"{m['theta_sat_mean']:.2f} | {m['action_clip_mean']:.2f} | "
                f"{m['nmse_mean']:.3f} | {m['cos_full_mean']:.3f} | {m['cos_stable_mean']:.3f} |"
            )
        lines.append("")

    code, msg = decision(b1)
    lines += [
        "## Pre-registered decision",
        "",
        f"**Case {code}.** {msg}",
        "",
        "Stage 6B.2 was **not** auto-started.",
        "",
    ]
    if s5:
        lines += [
            "## Comparison to Stage 5E corrected (seeds 5–9)",
            "",
            "Stage 5E corrected (exploratory OOD set) vs 6B.1 (confirmatory):",
            "",
            "| contrast | Stage5E corr. n_pos | Stage6B.1 n_pos |",
            "|----------|---------------------|-----------------|",
        ]
        # map from corrected 5E contrasts
        c5 = s5.get("paired", {})
        mapping = [
            ("interp Dir−Base", "interp_dir_minus_base", "interp_dir_minus_base"),
            ("interp HistDir−Mc", "interp_hist_dir_minus_mc", "interp_hist_dir_minus_mc"),
            ("extrap Dir−Base", "extrap_dir_minus_base", "extrap_dir_minus_base"),
            ("extrap HistDir−Mc", "extrap_hist_dir_minus_mc", "extrap_hist_dir_minus_mc"),
        ]
        for label, k5, k6 in mapping:
            a = c5.get(k5, {})
            b = st.get(k6, {})
            lines.append(
                f"| {label} | {a.get('n_positive', '?')}/{a.get('n', 5)} | {b.get('n_positive', '?')}/{b.get('n', 5)} |"
            )
        lines.append("")
    (REP / "stage6b1_results.md").write_text("\n".join(lines) + "\n")


def write_analysis(b1, s5):
    code, msg = decision(b1)
    st = b1["paired_seed_stats"]
    ps = b1["pool_summary"]
    lines = [
        "# Stage 6B.1 — Analysis for Human Review",
        "",
        "## 1. Did Dir−Base reproduce?",
        "",
        f"- Interp: {fmt_ci(st['interp_dir_minus_base'])}",
        f"- Extrap: {fmt_ci(st['extrap_dir_minus_base'])}",
        "",
        "## 2. Did Hist+Dir beat Mc?",
        "",
        f"- Interp: {fmt_ci(st['interp_hist_dir_minus_mc'])}",
        f"- Extrap: {fmt_ci(st['extrap_hist_dir_minus_mc'])}",
        "",
        "## 3. FailRate (Stage 6A definition)",
        "",
        f"- Interp Dir fail={ps['interpolation']['eagm_dir_matched']['fail_rate_mean']:.2f} "
        f"vs Base {ps['interpolation']['eagm_base']['fail_rate_mean']:.2f} "
        f"vs Mc {ps['interpolation']['mc']['fail_rate_mean']:.2f}",
        f"- Extrap Dir fail={ps['extrapolation']['eagm_dir_matched']['fail_rate_mean']:.2f} "
        f"vs Base {ps['extrapolation']['eagm_base']['fail_rate_mean']:.2f} "
        f"vs Mc {ps['extrapolation']['mc']['fail_rate_mean']:.2f}",
        "",
        "## 4. Saturation",
        "",
        f"- Interp mean θ-sat: Base={ps['interpolation']['eagm_base']['theta_sat_mean']:.2f}, "
        f"Dir={ps['interpolation']['eagm_dir_matched']['theta_sat_mean']:.2f}, "
        f"HistDir={ps['interpolation']['eagm_hist_dir_matched']['theta_sat_mean']:.2f}, "
        f"Mc={ps['interpolation']['mc']['theta_sat_mean']:.2f}",
        f"- Action clip ≈ {ps['interpolation']['eagm_dir_matched']['action_clip_mean']:.2f} (Dir)",
        "",
        "If Dir θ-sat remains high while FailRate stays low, the two problems remain separable: "
        "gradient reliability vs optimization sustainability.",
        "",
        "## 5. Gradient full vs stable cosine",
        "",
        f"- Interp Dir cos_full={ps['interpolation']['eagm_dir_matched']['cos_full_mean']:.3f}, "
        f"cos_stable={ps['interpolation']['eagm_dir_matched']['cos_stable_mean']:.3f}",
        f"- Interp Base cos_full={ps['interpolation']['eagm_base']['cos_full_mean']:.3f}, "
        f"cos_stable={ps['interpolation']['eagm_base']['cos_stable_mean']:.3f}",
        "",
        "## 6. Decision",
        "",
        f"**Case {code}.** {msg}",
        "",
        "## 7. Recommended next step (await approval)",
        "",
    ]
    if code == "A":
        lines.append("- After review: Stage 6B.2 HistC ablation (c_obs history), not ExecState yet.")
    elif code == "B":
        lines.append("- After review: diagnose Dir labels / FD / θ path before any architecture change.")
    else:
        lines.append("- After review: Stage 6B.2 HistC / execution-state estimation; keep Dir as fixed module.")
    lines += [
        "",
        "Do **not** start Stage 6B.2 automatically from this report.",
        "",
    ]
    # failure cases: seeds where HistDir loses to Mc
    lines += ["## 8. Failure / negative seeds", ""]
    for r in b1["per_seed"]:
        notes = []
        if r["interp_hist_dir_minus_mc"] <= 0:
            notes.append(f"interp HistDir−Mc={r['interp_hist_dir_minus_mc']:.6g}")
        if r["extrap_hist_dir_minus_mc"] <= 0:
            notes.append(f"extrap HistDir−Mc={r['extrap_hist_dir_minus_mc']:.6g}")
        if r["interp_dir_minus_base"] <= 0:
            notes.append(f"interp Dir−Base={r['interp_dir_minus_base']:.6g}")
        if r["extrap_dir_minus_base"] <= 0:
            notes.append(f"extrap Dir−Base={r['extrap_dir_minus_base']:.6g}")
        if notes:
            lines.append(f"- seed {r['seed']}: " + "; ".join(notes))
    if not any(
        r["interp_hist_dir_minus_mc"] <= 0 or r["extrap_hist_dir_minus_mc"] <= 0
        or r["interp_dir_minus_base"] <= 0 or r["extrap_dir_minus_base"] <= 0
        for r in b1["per_seed"]
    ):
        lines.append("- None (all listed contrasts positive on all seeds).")
    lines.append("")
    (REP / "stage6b1_analysis.md").write_text("\n".join(lines) + "\n")


def make_figures(b1):
    FIG.mkdir(parents=True, exist_ok=True)
    seeds = [r["seed"] for r in b1["per_seed"]]
    # Dir-Base bars interp/extrap
    fig, ax = plt.subplots(figsize=(5.4, 3.2), dpi=160)
    x = np.arange(len(seeds))
    w = 0.35
    di = [r["interp_dir_minus_base"] for r in b1["per_seed"]]
    de = [r["extrap_dir_minus_base"] for r in b1["per_seed"]]
    ax.bar(x - w / 2, di, width=w, label="Interp Dir−Base", color="#54A24B")
    ax.bar(x + w / 2, de, width=w, label="Extrap Dir−Base", color="#4C78A8")
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"s{s}" for s in seeds])
    ax.set_ylabel("Policy return difference")
    ax.set_title("Stage 6B.1 Dir−Base (seeds 10–14)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "stage6b1_dir_minus_base.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(5.4, 3.2), dpi=160)
    hi = [r["interp_hist_dir_minus_mc"] for r in b1["per_seed"]]
    he = [r["extrap_hist_dir_minus_mc"] for r in b1["per_seed"]]
    ax.bar(x - w / 2, hi, width=w, label="Interp HistDir−Mc", color="#F58518")
    ax.bar(x + w / 2, he, width=w, label="Extrap HistDir−Mc", color="#E45756")
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"s{s}" for s in seeds])
    ax.set_ylabel("Policy return difference")
    ax.set_title("Stage 6B.1 Hist+Dir − Mc")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "stage6b1_histdir_minus_mc.png")
    plt.close()

    # FailRate comparison
    fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=160)
    models = ["mc", "eagm_base", "eagm_dir_matched", "eagm_hist", "eagm_hist_dir_matched"]
    labels = ["Mc", "Base", "Dir", "Hist", "Hist+Dir"]
    xi = np.arange(len(models))
    fi = [b1["pool_summary"]["interpolation"][m]["fail_rate_mean"] for m in models]
    fe = [b1["pool_summary"]["extrapolation"][m]["fail_rate_mean"] for m in models]
    ax.bar(xi - w / 2, fi, width=w, label="Interp", color="#54A24B")
    ax.bar(xi + w / 2, fe, width=w, label="Extrap", color="#4C78A8")
    ax.set_xticks(xi)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("FailRate vs θ₀")
    ax.set_title("Stage 6B.1 OOD FailRate")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "stage6b1_failrate.png")
    plt.close()

    # Saturation
    fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=160)
    si = [b1["pool_summary"]["interpolation"][m]["theta_sat_mean"] for m in models]
    ax.bar(xi, si, color="#B279A2", width=0.6)
    ax.set_xticks(xi)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("θ param saturation")
    ax.set_title("Stage 6B.1 interpolation θ saturation")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "stage6b1_theta_sat.png")
    plt.close()


def main():
    p = ROOT / "runs" / "stage6b1_confirm" / "results.json"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    b1 = json.loads(p.read_text())
    s5 = load_stage5e_corrected()
    write_results(b1, s5)
    write_analysis(b1, s5)
    make_figures(b1)
    code, msg = decision(b1)
    print("wrote reports; decision Case", code)
    print(msg)


if __name__ == "__main__":
    main()
