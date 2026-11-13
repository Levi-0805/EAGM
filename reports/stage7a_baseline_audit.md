# Stage 7A — Baseline Audit

**Status:** PASS (`runs/stage7a_baseline_audit/audit.json`)  
**Git HEAD at audit:** `stage7-cross-task-generalization` @ parent `43cfc79`  
**Date:** 2026-11-13

## Frozen method inventory

| Model | Role | Params (construct) |
|-------|------|--------------------|
| Mc | Full control-memory baseline | 46268 |
| EAGM-Base | Structured U0+Δ, no Dir | 40874 |
| EAGM-Dir-Random | **Primary method** | 40874 |
| AuxCommandMLP | Fair aux-u_exec control | 53002 |

Locked hypers: WM 3000, policy 50, lr_p=0.03, λ_u/λ_d=1.0/0.1, batch 128, lr_m=1e-3.

## Path-Dir pilot documentation — RESOLVED

| Source | What it said | Truth |
|--------|--------------|-------|
| `scripts/run_stage6c2_pathdir.py` | Pilot on frozen **trained Base** | **Authoritative** |
| `reports/stage6c_pathdir_results.md` | Base pilot | Consistent |
| `reports/stage6c_saturation_analysis.md` (old) | “frozen Dir” for path probes | **Misleading** — referred to 6C.0 diagnostic smoke |
| 6C.0 `audit.json` | Used Stage 6B.1 Dir ckpt | Diagnostics only (ρ_proj / sat), **not** Path-Dir labels |

**Paper-safe statement:** Path-Dir labels in Stage 6C.2/6C.3 came from a policy-optimization pilot on a **frozen trained EAGM-Base**. Stage 6C.0 used a Dir checkpoint only for saturation/projection smoke. The saturation report now contains an explicit clarification.

## Stage 6C recoverability

- Manifest + 6C.0/6C.1/6C.2 JSON/CSV/protocol present; SHA256 recorded in audit JSON.
- Recomputed Path−Random interp mean matches report (−3.907e−4); n+=2/5; decision `STOP_PATHDIR`.
- Local `seed*/**.pt` exist but remain gitignored (same policy as prior stages).

## Decisions carried into Stage 7

1. Retain Dir-Random; Path-Dir stopped; no Stage 6D.  
2. Do not treat θ-saturation reduction as an optimization objective.  
3. Do not fold Stage 6C/7 into the current CVPR main manuscript without a separate method plan.  
4. Proceed to Reach (7B); no multi-seed 7D until smoke gate.

## Environment stamp (from audit JSON)

See `runs/stage7a_baseline_audit/audit.json` → `platform` (torch, CUDA GPU, MuJoCo version).
