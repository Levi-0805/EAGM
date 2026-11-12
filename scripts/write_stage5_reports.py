#!/usr/bin/env python3
"""Assemble Stage 5 markdown reports from runs/*/results.json after experiments finish."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def fmt_ci(d):
    if not d:
        return "n/a"
    return (
        f"mean={d['mean']:.6g}, median={d['median']:.6g}, "
        f"95% CI [{d['ci95'][0]:.6g}, {d['ci95'][1]:.6g}], "
        f"n_pos={d['n_positive']}/{d['n']}"
    )


def sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def write_confirmatory(a5, s4):
    lines = [
        "# Stage 5A — Confirmatory Results (seeds 5–9)",
        "",
        f"Frozen from Stage 4 commit `{load(ROOT/'configs'/'stage5_frozen.json')['frozen_from_stage4_commit']}`. "
        f"Report commit: `{sha()}`.",
        "",
        "Primary metric: **EAGM-Dir-Matched − EAGM-Base** physical policy return.",
        "Secondary: Dir − Mc. Exploratory Stage 4 (seeds 0–4) reported separately; merged numbers are descriptive only.",
        "",
        "## Confirmatory seed table",
        "",
        "| seed | Mc | Base | Dir | Dir−Base | Dir−Mc | cos Mc | cos Base | cos Dir |",
        "|------|----|------|-----|----------|--------|--------|----------|---------|",
    ]
    if a5:
        for r in a5["confirmatory"]["per_seed"]:
            lines.append(
                f"| {r['seed']} | {r['mc']:.6f} | {r['base']:.6f} | {r['dir']:.6f} | "
                f"{r['dir_minus_base']:.6g} | {r['dir_minus_mc']:.6g} | "
                f"{r['cos_mc']:.3f} | {r['cos_base']:.3f} | {r['cos_dir']:.3f} |"
            )
        st = a5["confirmatory"]["paired_seed_stats"]
        lines += [
            "",
            "### Paired seed stats (confirmatory only)",
            "",
            f"- Dir − Base: {fmt_ci(st['dir_vs_base'])}",
            f"- Dir − Mc: {fmt_ci(st['eagm_dir_matched_vs_mc']['return_'])}",
            f"- Base − Mc: {fmt_ci(st['eagm_base_vs_mc']['return_'])}",
            "",
            f"Wall seconds: {a5.get('wall_seconds')}",
            "",
            "## Exploratory Stage 4 (seeds 0–4) — not confirmatory",
            "",
        ]
        if a5.get("exploratory"):
            for r in a5["exploratory"]["per_seed"]:
                lines.append(
                    f"- seed {r['seed']}: Dir−Base={r['dir_minus_base']:.6g}, Dir−Mc={r['dir_minus_mc']:.6g}"
                )
            lines.append(f"- Exploratory Dir−Base: {fmt_ci(a5['exploratory']['paired_seed_stats']['dir_vs_base'])}")
        if a5.get("merged_descriptive"):
            lines += [
                "",
                "## Merged descriptive (do not use alone as confirmation)",
                "",
                a5["merged_descriptive"].get("warning", ""),
                f"- Merged Dir−Base: {fmt_ci(a5['merged_descriptive']['paired_seed_stats']['dir_vs_base'])}",
            ]
        # Q1 verdict
        d = st["dir_vs_base"]
        q1 = (
            "YES — confirmatory Dir−Base mean positive with majority seeds positive."
            if d["mean"] > 0 and d["n_positive"] >= 3
            else "NO / WEAK — confirmatory set did not clearly reproduce Dir−Base advantage."
        )
        lines += ["", "## Q1 (confirmatory reproduction)", "", q1, ""]
    else:
        lines.append("_Stage 5A results not found._")
    (REP / "stage5_confirmatory_results.md").write_text("\n".join(lines) + "\n")


def write_budget(b5):
    lines = [
        "# Stage 5B — Budget-Matched Comparison",
        "",
        "Train interaction budget = dynamics collection + train direction labels.",
        "Test FD diagnostics listed separately and excluded from train budget.",
        "",
        "Methods:",
        "- **A** EAGM-Base, original dynamics budget",
        "- **B** EAGM-Base, Dir's train FD budget converted to extra dynamics trajectories",
        "- **C** EAGM-Dir-Matched (dynamics + direction supervision)",
        "",
    ]
    if not b5:
        lines.append("_Stage 5B results not found._")
        (REP / "stage5_budget_matched.md").write_text("\n".join(lines) + "\n")
        return
    lines += [
        f"Extra episodes for B: {b5['extra_episodes']} (target dir steps {b5['dir_train_steps_expected']}).",
        "",
        "| seed | A Base | B Base+dyn | C Dir | C−A | C−B | B−A |",
        "|------|--------|------------|-------|-----|-----|-----|",
    ]
    for r in b5["per_seed"]:
        lines.append(
            f"| {r['seed']} | {r['A']:.6f} | {r['B']:.6f} | {r['C']:.6f} | "
            f"{r['C_minus_A']:.6g} | {r['C_minus_B']:.6g} | {r['B_minus_A']:.6g} |"
        )
    lines += [
        "",
        "### Paired stats",
        f"- C−A: {fmt_ci(b5['paired']['C_minus_A'])}",
        f"- C−B: {fmt_ci(b5['paired']['C_minus_B'])}",
        f"- B−A: {fmt_ci(b5['paired']['B_minus_A'])}",
        "",
        b5.get("interpretation_rule", ""),
        "",
    ]
    c_b = b5["paired"]["C_minus_B"]
    q2 = (
        "Dir retains an edge under matched train sim budget (C>B)."
        if c_b["mean"] > 0 and c_b["n_positive"] >= 3
        else "Under matched budget, Base+extra dynamics matches or beats Dir — "
             "cannot claim free sample-efficiency from direction labels alone."
    )
    lines += ["## Q2", "", q2, ""]
    (REP / "stage5_budget_matched.md").write_text("\n".join(lines) + "\n")


def write_grad(d5):
    lines = [
        "# Stage 5D — Gradient Quality on the Policy Path",
        "",
        "Cosine / balanced directional accuracy / directional error at θ0, θ_mid (u=25), θ_final,",
        "comparing model ∇θJ to independent sim FD on the same starts used for policy opt.",
        "",
    ]
    if not d5:
        lines.append("_Stage 5D results not found._")
        (REP / "stage5_gradient_policy_analysis.md").write_text("\n".join(lines) + "\n")
        return
    lines += [
        "| seed | Δcos θ0 | Δcos mid | Δcos final | Δtrue-return (Dir−Base) | probe cos Base | probe cos Dir |",
        "|------|---------|----------|------------|-------------------------|----------------|---------------|",
    ]
    for r in d5["per_seed"]:
        lines.append(
            f"| {r['seed']} | {r['path_cosine_delta']['theta0']:.3f} | "
            f"{r['path_cosine_delta']['theta_mid']:.3f} | {r['path_cosine_delta']['theta_final']:.3f} | "
            f"{r['update_return_delta']['dir_minus_base']:.6g} | "
            f"{r['probe_cosine']['base']:.3f} | {r['probe_cosine']['dir']:.3f} |"
        )
    lines += [
        "",
        "### Paired",
        f"- path Δcos θ0: {fmt_ci(d5['paired']['path_cos_delta_theta0'])}",
        f"- path Δcos mid: {fmt_ci(d5['paired']['path_cos_delta_mid'])}",
        f"- path Δcos final: {fmt_ci(d5['paired']['path_cos_delta_final'])}",
        f"- update return Dir−Base: {fmt_ci(d5['paired']['update_return_dir_minus_base'])}",
        "",
        "## Q4",
        "",
    ]
    mid = d5["paired"]["path_cos_delta_mid"]
    uret = d5["paired"]["update_return_dir_minus_base"]
    if mid["mean"] > 0 and uret["mean"] > 0:
        q4 = "Dir improves path-region gradient cosine and yields better true update returns — consistent with causal link."
    elif mid["mean"] > 0 and uret["mean"] <= 0:
        q4 = "Path cosine improves but true update returns do not — probe/path accuracy may not drive policy gains."
    else:
        q4 = "No stable path-region gradient advantage for Dir; cannot attribute policy gains to on-path gradient fidelity."
    lines += [q4, ""]
    (REP / "stage5_gradient_policy_analysis.md").write_text("\n".join(lines) + "\n")


def write_ood_protocol():
    text = """# Stage 5E — OOD Controller-B Protocol (gated)

Status: **not started until Stage 5A–5C pass stop-conditions.**

## Design (frozen when launched)

- Train-time randomization of Controller B: gain, extra_lag, rate_limit, smooth_beta.
- All models share identical randomized configs and trajectories.
- EAGM must not read hidden B parameters.
- Val: independently sampled configs. Test: frozen held-out configs.

### Interpolation
Test unseen combinations inside the training ranges.

### Extrapolation
Test reasonable configs outside training ranges.

Report separately (never mix into one OOD average):
- command MAE, dynamics NMSE, gradient cosine, physical policy return, failure rate by config.

Prefer existing History encoder + Dir-Matched. Design CF loss only after documenting residual counterfactual gradient errors unexplained by history.
"""
    (REP / "stage5_ood_protocol.md").write_text(text)


def write_decision(a5, b5, c5, d5):
    lines = [
        "# Stage 5 — Research Decision",
        "",
        f"Git commit: `{sha()}`",
        "",
        "## Answers",
        "",
    ]
    # Q1
    if a5:
        d = a5["confirmatory"]["paired_seed_stats"]["dir_vs_base"]
        q1 = f"Q1: confirmatory Dir−Base mean={d['mean']:.6g}, n_pos={d['n_positive']}/{d['n']}."
    else:
        q1 = "Q1: pending."
    lines.append(q1)

    if b5:
        cb = b5["paired"]["C_minus_B"]
        lines.append(f"Q2: C−B (Dir vs budget-matched Base) mean={cb['mean']:.6g}, n_pos={cb['n_positive']}/{cb['n']}.")
    else:
        lines.append("Q2: pending.")

    if c5:
        ba = c5["paired"]["base_minus_aux"]
        da = c5["paired"]["dir_minus_aux"]
        lines.append(
            f"Q3: Base−Aux mean={ba['mean']:.6g} (n_pos={ba['n_positive']}); "
            f"Dir−Aux mean={da['mean']:.6g} (n_pos={da['n_positive']})."
        )
    else:
        lines.append("Q3: pending.")

    if d5:
        mid = d5["paired"]["path_cos_delta_mid"]
        lines.append(f"Q4: path Δcos mid Dir−Base mean={mid['mean']:.6g}, n_pos={mid['n_positive']}/{mid['n']}.")
    else:
        lines.append("Q4: pending.")

    lines += [
        "Q5: Stage 5E not launched (gated).",
        "",
        "## Q6 — Current contribution scope",
        "",
        "- Fixed-B exploratory (Stage 4): Dir-Matched beat paired Base on policy return (5/5).",
        "- Stage 5 asks whether that survives independent seeds, budget matching, and fair baselines.",
        "- Not yet evidence for public robot-benchmark SOTA or vision-scale claims.",
        "",
        "## Stop conditions",
        "",
    ]
    stop = []
    if a5:
        d = a5["confirmatory"]["paired_seed_stats"]["dir_vs_base"]
        if not (d["mean"] > 0 and d["n_positive"] >= 3):
            stop.append("5A confirmatory Dir−Base advantage not reproduced → pause scale-up / 5E.")
    if b5:
        cb = b5["paired"]["C_minus_B"]
        if not (cb["mean"] > 0 and cb["n_positive"] >= 3):
            stop.append("5B: budget-matched Base matches/beats Dir → do not claim sample-efficiency.")
    if c5:
        ba = c5["paired"]["base_minus_aux"]
        if ba["mean"] <= 0:
            stop.append("5C: structured bottleneck does not beat aux-command MLP under same u_exec labels.")
    if not stop:
        stop.append("No hard stop triggered from available results; 5E may be considered after human review.")
    lines += [f"- {s}" for s in stop]
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "conda activate eagm_gpu",
        "export OPENARM_MODEL_PATH=.../openarm_bimanual.xml",
        "bash scripts/run_stage5_all.sh",
        "python scripts/write_stage5_reports.py",
        "```",
        "",
    ]
    (REP / "stage5_research_decision.md").write_text("\n".join(lines) + "\n")


def write_baselines_section(c5):
    # append into confirmatory or separate brief in decision; also ensure budget report mentions 5C
    p = REP / "stage5_confirmatory_results.md"
    extra = ["", "## Stage 5C baselines (summary)", ""]
    if not c5:
        extra.append("_pending_")
    else:
        extra += [
            "| seed | ObsMLP | AuxMLP | Base | Dir | Base−Obs | Base−Aux | Dir−Aux |",
            "|------|--------|--------|------|-----|----------|----------|---------|",
        ]
        for r in c5["per_seed"]:
            ret = r["returns"]
            extra.append(
                f"| {r['seed']} | {ret['obs_memory_mlp']:.6f} | {ret['aux_command_mlp']:.6f} | "
                f"{ret['eagm_base']:.6f} | {ret['eagm_dir_matched']:.6f} | "
                f"{r['base_minus_obs']:.6g} | {r['base_minus_aux']:.6g} | {r['dir_minus_aux']:.6g} |"
            )
        extra += [
            "",
            f"- Base−Obs: {fmt_ci(c5['paired']['base_minus_obs'])}",
            f"- Base−Aux: {fmt_ci(c5['paired']['base_minus_aux'])}",
            f"- Dir−Aux: {fmt_ci(c5['paired']['dir_minus_aux'])}",
            f"- Aux−Obs: {fmt_ci(c5['paired']['aux_minus_obs'])}",
        ]
    if p.exists():
        p.write_text(p.read_text() + "\n".join(extra) + "\n")


def main():
    a5 = load(ROOT / "runs" / "stage5a_confirm" / "results.json")
    b5 = load(ROOT / "runs" / "stage5b_budget" / "results.json")
    c5 = load(ROOT / "runs" / "stage5c_baselines" / "results.json")
    d5 = load(ROOT / "runs" / "stage5d_gradpath" / "results.json")
    write_confirmatory(a5, None)
    write_budget(b5)
    write_grad(d5)
    write_ood_protocol()
    write_decision(a5, b5, c5, d5)
    write_baselines_section(c5)
    print("wrote reports to", REP)


if __name__ == "__main__":
    main()
