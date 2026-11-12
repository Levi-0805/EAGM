# EAGM training results (early)

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

**Honest reading:** After the init/loss fix, residual improves cmd MAE over U0 and beats Mc/Mu on this small IID split. Still **not** a method claim: no held-out execution config, no Dir/CF losses, no policy return, single seed, tiny data. Next before Stage 4:

1. History encoder + longer context for lag identification.
2. Masked Dir + policy-region θ labels; CF intervention loss.
3. Matched-budget policy return on Controller B (and held-out randomized B).
4. Answer Q1–Q6 only after five-seed policy stats.
