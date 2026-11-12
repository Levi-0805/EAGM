# EAGM training results

> **Update 2026-11-12:** Stage 3 exploratory `stage3_b_*` runs are **deprecated** (protocol bugs).  
> Authoritative single-seed numbers: `reports/stage3_repaired_results.md` / `runs/stage3_1_seed0/`.

## EAGM training results (early / deprecated exploratory)

## Stage 1 regression (Controller A, exact)

See `baseline_reproduction.md`. Both 1k and 3k five-seed tables match the CPU archive; stop condition on exact Mu+Dir @ 3k reconfirmed.

## Stage 3 Controller B smoke (seed 0, 800 updates)

`runs/stage3_b_min_seed0/results.json`

Plant: gain=1.05, extra_lag=1, rate_limit=0.045, smooth_beta=0.25 (fixed).  
Data: 8×150 steps, 840 train / 360 test transfers. **Dynamics/cmd only — no policy return yet.**

First 400-update smoke had Δ last-layer randomly initialized and \(\lambda_u=0.1\); cmd MAE **regressed** to 0.173 vs U0 0.0165. Fixed by zero-init Δ and \(\lambda_u=1\) (normalized by `ustd`), then reran:

| Model | Test dyn NMSE | Cmd MAE |
|---|---:|---:|
| Mc (full c) | 0.236 | — |
| Mu oracle (true u) | 0.245 | — |
| EAGM-Base (U0+Δ on c_obs) | **0.199** | **0.0121** |
| U0 only (no Δ) | — | 0.0165 |

**Honest reading:** After the init/loss fix, residual improves cmd MAE over U0 and beats Mc/Mu on this small IID split. Still **not** a method claim: no held-out execution config, no Dir/CF losses, no policy return, single seed, tiny data.

## Stage 3 Controller B + masked Dir (seed 0, 1000 updates)

`runs/stage3_b_dir_seed0/results.json` (wall ~95 s)

| Model | Dyn NMSE | Cmd MAE | Grad cosine | Policy return |
|---|---:|---:|---:|---:|
| Mc (full c) | 0.284 | — | 0.256 | -0.00563 |
| Mu oracle | 0.280 | — | — | — |
| EAGM-Base | 0.275 | 0.00318 | 0.576 | **-0.00386** |
| EAGM-Dir (masked) | 0.276 | 0.00308 | **0.833** | -0.00432 |

**Reading (single seed, exploratory):**
- Masked Dir **clearly lifts gradient cosine** (0.58 → 0.83) under Controller B.
- Best **policy return** on this seed is EAGM-Base, not EAGM-Dir (same qualitative pattern as exact-Mu Stage 2: Dir helps geometry, not always return).
- EAGM-Base/Dir both beat Mc on return here — promising for partial-unknown setting, but needs five-seed + held-out B configs before any claim.

Next before Stage 4 scale-up:
1. History encoder for lag identification.
2. Policy-region θ direction labels + CF loss.
3. Five-seed paired policy returns + held-out randomized Controller B.
