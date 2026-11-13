# Stage 5E — Corrected Statistics (Stage 6A)

Stage 5 raw runs under `runs/stage5e_ood/` are **frozen and unchanged**.  
Corrected re-evaluation: `runs/stage6a_stage5e_corrected/` (loads Stage 5E checkpoints; paired ΔR / fail rate / saturation / budget).

## What was wrong

1. **`fail_frac`** used `mean(return > 0)`, which is not “worse than θ₀”.
2. **Budget** counted all 36 collected episodes as train (`7200`), mixing train/val/IID.
3. **θ saturation** vs **action clipping** were not separated in OOD eval.

## Corrected budget (per seed)

| Use | Episodes | Control steps |
|-----|----------|---------------|
| Train dynamics | 24 | 4,800 |
| Validation | 6 | 1,200 |
| IID holdout | 6 | 1,200 |
| Collect total | 36 | 7,200 |
| Dir-label train FD | — | 16,128 |
| **Train interaction budget** | — | **20,928** |
| OOD FD diagnostic (interp+extrap) | — | 32,256 (not in train budget) |

## Corrected fail rate

Definition: \(\mathrm{FailRate}=\frac1N\sum_i\mathbf{1}[R_i(\theta_{\mathrm{final}})-R_i(\theta_0)<0]\) on identical starts.

| Pool | Model | mean FailRate | mean θ-param sat | mean action clip |
|------|-------|---------------|------------------|------------------|
| Interp | Mc | 0.00 | 0.95 | 0.00 |
| Interp | Base | 0.35 | 0.45 | 0.00 |
| Interp | Dir | 0.00 | 0.86 | 0.00 |
| Interp | Hist | 0.05 | 0.87 | 0.00 |
| Interp | Hist+Dir | 0.00 | 0.84 | 0.00 |
| Extrap | Mc | 0.13 | 0.95 | 0.00 |
| Extrap | Base | 0.42 | 0.45 | 0.00 |
| Extrap | Dir | 0.07 | 0.86 | 0.00 |
| Extrap | Hist | 0.20 | 0.87 | 0.00 |
| Extrap | Hist+Dir | 0.08 | 0.84 | 0.00 |

Policy return contrasts (recomputed) match Stage 5E within numerical noise: Dir−Base **5/5**; HistDir−Mc **3/5**, CI includes 0.

## Saturation note (also Stage 5D)

On fixed-B Stage 5D path dumps: final θ-param sat Base ≈ **73.6%**, Dir ≈ **92.1%** (see `reports/figures/stage6a_theta_saturation.png`).  
On OOD re-eval, action clip ≈ **0** while θ-param sat remains high for Dir/Hist — boundary saturation is a **parameter** phenomenon, not actor clipping.

## Implication for Stage 6

- Keep Stage 5 qualitative OOD conclusions; use corrected fail rates going forward.
- Val / IID holdout must be used for model/policy selection in Stage 6B+.
- Stage 6C should test whether lowering θ saturation preserves mid-path cosine *and* return.
