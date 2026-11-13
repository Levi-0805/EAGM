# Stage 6C.2 / 6C.3 — Path-Aware Dir Results

**Branch:** `stage6c-path-aware-saturation`  
**Parent:** `1f25a97`  
**Seeds:** 15–19 (confirmatory)  
**Artifacts:** `runs/stage6c2_pathdir/`  
**Wall:** ≈2545 s (~42 min)  
**Decision:** **STOP_PATHDIR**

## Setup (matched)

| Item | Value |
|------|-------|
| Models | Mc, Base, Dir-Random, Dir-Path |
| Arch / init / batch_seed | Identical for Dir-Random & Dir-Path (`seed+40`) |
| WM updates / λ_u / λ_d | 3000 / 1.0 / 0.1 |
| Dir labels | 12 each; FD steps **16128 = 16128** (matched) |
| FD | `feasible=True` (bound-aware); ε=0.05; scales 0.5/1/2; H=8 |
| Path probes | Pilot on frozen trained Base; θ at 0,0.2,…,1.0 |
| Random probes | θ=0 ∪ N(0,0.3²) clipped |
| Policy | Adam lr=0.03, 50 updates (locked) |

## Primary contrast: Dir-Path − Dir-Random (physical return)

| Pool | Mean | Median | n+ / 5 | 95% CI |
|------|------|--------|--------|--------|
| **Interpolation** | **−0.00039** | −0.00052 | **2/5** | [−0.00109, +0.00033] |
| Extrapolation | −0.00054 | −0.00083 | 2/5 | [−0.00138, +0.00036] |

Per-seed interp Δ(Path−Random): `+0.00026, −0.00155, −0.00052, +0.00076, −0.00092`

**Gate:** mean>0 ∧ CI_lo>0 ∧ n+≥4 → KEEP. Observed: **STOP_PATHDIR**.

A better gradient cosine alone would be insufficient; here physical return also fails the gate.

## Secondary contrasts (Path still a strong Dir)

| Contrast | Interp mean | n+ | CI |
|----------|-------------|----|----|
| Path − Base | **+0.00235** | **5/5** | >0 |
| Path − Mc | **+0.00123** | **5/5** | >0 |
| Random − Base | **+0.00274** | **5/5** | >0 |

Extrapolation mirrors interpolation. Direction supervision (either probe scheme) continues to beat Base and Mc; **path sampling does not beat random probes**.

## Budget / protocol notes

- Train-only starts for path pilot and all dir labels.
- OOD FD labels are diagnostic only (not in train budget for selection).
- Feasible FD used for Stage 6C labels; legacy `feasible=False` preserved as default for older scripts.
- Unit tests: `tests/test_stage6c_fd_boundary.py` (PASS).

## Figures

- `reports/figures/stage6c/stage6c3_paired_returns.png`
- `reports/figures/stage6c/stage6c_seed15_sat_curves.png`

## Claim language (allowed)

> Under a matched simulator-query budget, direction labels collected along a frozen policy-optimization path do **not** yield a reproducible physical-return gain over the existing random-θ probe protocol (2/5 seeds, CI contains 0).

Do **not** claim Path-Aware Dir as a method contribution.
