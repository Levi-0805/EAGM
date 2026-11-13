# Stage 6C — Research Decision

**Branch:** `stage6c-path-aware-saturation`  
**Parent commit:** `1f25a971bf42d2dbe600e4b62d9870e8e4a870d7`  
**Date:** 2026-11-13  
**Status:** Complete — **stop for human review. Do not launch Stage 6D automatically.**

## Decisions

| Track | Decision | Rationale |
|-------|----------|-----------|
| HistC / ExecState | **DROP** (carried from 6B.2) | Capacity / reverse-order controls failed |
| Policy LR / desaturation | **Do not promote as method** | For Dir, lower LR removes sat but **hurts** return (0/5) |
| Path-Aware Dir | **STOP_PATHDIR** | Path − Random: 2/5, mean<0, CI∋0 |
| Random / matched Dir | **Keep** as core method | Path & Random both beat Base & Mc 5/5 |

## Evidence maturity (updated)

| Hypothesis | Status |
|------------|--------|
| Direction consistency improves policy gradients / returns vs Base | **Confirmed** (Stages 4–6C) |
| Exec-history encoding (HistC) adds capacity-independent gain | **Rejected** (6B.2) |
| θ saturation is the Dir performance bottleneck | **Not supported** (6C.1) |
| Path-sampled Dir labels beat random probes | **Not supported** (6C.3) |
| Public-benchmark / cross-task SOTA | **Not established** |

## Narrative for the paper (honest)

1. Execution interface has hidden dynamics; Markov physical state is incomplete.  
2. **Directional consistency supervision** improves world-model policy gradients and physical returns vs Base (and typically Mc).  
3. Explicit exec-history inputs and path-aligned probe redistribution were tested and **did not** add reliable gains under matched controls.  
4. High θ-box saturation is common (Dir ~89%, Mc ~94%) but, for Dir, is compatible with strong returns; lowering LR is not a cure.

This is a credibility-positive story: validated core + negative ablations, not module stacking.

## What Stage 6C changed in code (additive)

- `training/gradients.py`: optional `feasible=True` bound-aware FD + metadata.  
- `training/policy_opt.py`: ρ_proj, first-hit, outward pressure, physical_eval hooks.  
- `training/stage6c_common.py`, scripts `run_stage6c0/1/2`, tests, reports under `reports/stage6c_*.md`.  
- Runs only under `runs/stage6c_*`. Prior Stage 5/6A/6B and `CVPR2027/` untouched.

## Go/No-Go application

| Criterion (from taskbook) | Result |
|---------------------------|--------|
| PathDir return > RandomDir, multi-seed stable | **Fail → stop path supervision** |
| Cosine↑ but return flat | N/A as primary; return also fails |
| Lower LR only delays sat, no Dir return gain | **Observed → not a method** |
| Saturation high but return OK | **Observed → diagnostic only** |

## Recommended next step (human choice — not auto 6D)

Options worth discussing, **not** started here:

1. Freeze Dir-Random as the paper method; tighten writing + figures for CVPR.  
2. Stronger external baselines / harder task (only if seeking broader claims).  
3. Fix documented 6B.2 implementation footnotes (c_obs align, probe leakage) only if revisiting HistC — **not recommended**.

## Deliverables checklist

- [x] `reports/stage6c_protocol.md`  
- [x] `reports/stage6c_saturation_analysis.md`  
- [x] `reports/stage6c_pathdir_results.md`  
- [x] `reports/stage6c_research_decision.md`  
- [x] Figures under `reports/figures/stage6c/`  
- [x] Raw JSON/CSV in `runs/stage6c0_audit`, `stage6c1_lr_ablation`, `stage6c2_pathdir`  
- [x] Unit tests for FD boundary + budget accounting  
