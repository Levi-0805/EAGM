# Stage 6C Protocol — Path-Aware Gradient Consistency & Policy Saturation

**Branch:** `stage6c-path-aware-saturation`  
**Parent commit:** `1f25a971bf42d2dbe600e4b62d9870e8e4a870d7`  
**Date:** 2026-11-13  

## Research questions

1. Does θ-box saturation actually restrict policy return, or is it a benign consequence of the constrained parameterization?
2. Can direction supervision along the real policy-optimization path (`Dir-Path`) beat matched-budget random probes (`Dir-Random`) on physical return?

## Non-goals

- No HistC / ExecState / extra controller-state inputs.
- Do not modify `CVPR2027/` or overwrite Stage 5 / 6A / 6B artifacts.
- Do not auto-launch Stage 6D.

## Preserved locks

From `configs/stage6_frozen.json` / Stage 5:

| Hyper | Value |
|-------|-------|
| World-model updates | 3000 |
| Policy updates (primary) | 50 |
| Imagined / eval horizon | 8 / 16 |
| λ_u / λ_d | 1.0 / 0.1 |
| Batch / lr_model | 128 / 1e-3 |
| Default lr_policy | 0.03 |
| θ box | [-1, 1] |

New runs only under `runs/stage6c_*`.

## 6C.0 — Audit (before large training)

- Verify paired Base/Dir init + minibatch stream (`training/paired_train.py`).
- Add **feasible FD** (`sim_grad_masked(..., feasible=True)`): interior central; active bounds one-sided; stability mask ≠ boundary mask.
- Default `feasible=False` keeps Stage 3–6B label semantics unchanged.
- Projection diagnostics in `policy_optimize`: `ρ_proj`, first-hit update, outward gradient pressure, optional physical curve.
- Smoke: reproduce Dir>Base signal on one seed at reduced updates; unit tests for FD/budget.

## 6C.1 — Policy LR ablation (frozen WM)

Models: Mc, EAGM-Base, EAGM-Dir from Stage 6B.1 checkpoints (seeds 10–14).  
Policy LRs: **0.03 / 0.015 / 0.0075**, updates=50, identical starts within seed.  
Selection: validation / train imagined curves only — **not** interp/extrap for hyper choice.  
Report final physical return, ΔR vs θ₀, FailRate, saturation trajectory, ρ_proj, action clip (separate).

## 6C.2 — Path-Aware Dir (retrain)

Three EAGM twins (same arch):

| Model | Direction loss |
|-------|----------------|
| Base | none |
| Dir-Random | random θ probes ~ N(0,0.3²) ∪ {0} |
| Dir-Path | θ checkpoints from frozen pilot policy path |

Matched: architecture, init, minibatch stream, updates, λ, FD ε/scales/horizon, **number of dir labels**, train-only starts for labels.  
Path pilot: train starts only; checkpoints at fractions 0,0.2,…,1.0; feasible FD for path labels.  
Also train Mc for secondary comparison.

## 6C.3 — Confirmatory seeds 15–19

Primary: `Dir-Path − Dir-Random` physical return (seed as statistical unit).  
Secondary: Path−Base, Path−Mc.  
Pools: interpolation and single-factor extrapolation (separate).  
Report per-seed Δ, mean, median, 95% bootstrap CI, #positive seeds.

## Go / No-Go

| Outcome | Decision |
|---------|----------|
| Path > Random on physical return, stable across seeds | Keep Path-Dir |
| Cosine ↑ but return flat | Do not claim optimization gain |
| Lower LR only delays sat, same final return | Not a method contribution |
| High sat but return unaffected | Report as diagnostic only |
| Path ≯ Random | Stop path-supervision development |

## Known caveats carried from 6B.2 (documentation only)

- HistC c_obs train/imagine alignment mismatch (does not change DROP_HISTC).
- HistCShuf was time-**reverse**, not random shuffle.
- Exec-state probe had train/test leakage.
- HistC cross-arch seeds were not init/batch paired.

These are recorded so Stage 6C does not inherit the same mistakes for Path vs Random.
