# Stage 6B.1 — Analysis for Human Review

## 1. Did Dir−Base reproduce?

- Interp: mean=0.00234256 ± 0.00085053, median=0.00243448, 95% CI [0.00168283, 0.00299457], n_pos=5/5
- Extrap: mean=0.00244405 ± 0.000993583, median=0.00264576, 95% CI [0.00164945, 0.00318889], n_pos=5/5

## 2. Did Hist+Dir beat Mc?

- Interp: mean=0.000525766 ± 0.00181415, median=0.000116466, 95% CI [-0.000747485, 0.00208577], n_pos=3/5
- Extrap: mean=0.000604422 ± 0.00191421, median=0.000193488, 95% CI [-0.000791418, 0.00227863], n_pos=4/5

## 3. FailRate (Stage 6A definition)

- Interp Dir fail=0.00 vs Base 0.65 vs Mc 0.20
- Extrap Dir fail=0.13 vs Base 0.78 vs Mc 0.30

## 4. Saturation

- Interp mean θ-sat: Base=0.69, Dir=0.89, HistDir=0.88, Mc=0.94
- Action clip ≈ 0.00 (Dir)

If Dir θ-sat remains high while FailRate stays low, the two problems remain separable: gradient reliability vs optimization sustainability.

## 5. Gradient full vs stable cosine

- Interp Dir cos_full=0.678, cos_stable=0.686
- Interp Base cos_full=0.214, cos_stable=0.222

## 6. Decision

**Case C.** Dir−Base reproduced, but Hist+Dir still does not stably beat Mc. Focus next on execution-state / history representation (Stage 6B.2 HistC), not more Dir complexity. Do not auto-start without human review.

## 7. Recommended next step (await approval)

- After review: Stage 6B.2 HistC / execution-state estimation; keep Dir as fixed module.

Do **not** start Stage 6B.2 automatically from this report.

## 8. Failure / negative seeds

- seed 10: interp HistDir−Mc=-6.27932e-05
- seed 12: interp HistDir−Mc=-0.00148533; extrap HistDir−Mc=-0.00157757

