# Stage 6B.1 — Independent Confirmation (seeds 10–14)

Protocol: Stage 5E legacy (2 episodes/config). Stage 6A metrics. No model/hyperparameter changes. Compound OOD excluded from main tables.

Artifacts: `runs/stage6b1_confirm/` · wall ≈ 3175s

## Confirmatory seed contrasts (policy return)

### Interpolation

| seed | Dir−Base | Dir−Mc | HistDir−Mc | HistDir−Hist |
|------|----------|--------|------------|--------------|
| 10 | 0.00333798 | 0.000673296 | -6.27932e-05 | -1.56122e-05 |
| 11 | 0.00243448 | 0.000618038 | 0.000116466 | -0.000553668 |
| 12 | 0.00123689 | -0.000490709 | -0.00148533 | 0.000133847 |
| 13 | 0.00177223 | 0.000662721 | 0.000602098 | 0.000535044 |
| 14 | 0.00293122 | 0.00373717 | 0.00345839 | 0.00165026 |

- Dir−Base: mean=0.00234256 ± 0.00085053, median=0.00243448, 95% CI [0.00168283, 0.00299457], n_pos=5/5
- Dir−Mc: mean=0.0010401 ± 0.00158687, median=0.000662721, 95% CI [-2.72221e-05, 0.00249845], n_pos=4/5
- HistDir−Mc: mean=0.000525766 ± 0.00181415, median=0.000116466, 95% CI [-0.000747485, 0.00208577], n_pos=3/5
- HistDir−Hist: mean=0.000349974 ± 0.000824913, median=0.000133847, 95% CI [-0.000228314, 0.00104369], n_pos=3/5

### Extrapolation

| seed | Dir−Base | Dir−Mc | HistDir−Mc | HistDir−Hist |
|------|----------|--------|------------|--------------|
| 10 | 0.00366851 | 0.000713134 | 0.000193488 | 0.00021253 |
| 11 | 0.00264576 | 0.000649515 | 0.000136257 | -0.000752611 |
| 12 | 0.00115107 | -0.000885264 | -0.00157757 | 0.000372873 |
| 13 | 0.00177405 | 0.000560581 | 0.000582142 | 0.000511204 |
| 14 | 0.00298084 | 0.00400994 | 0.0036878 | 0.00169498 |

- Dir−Base: mean=0.00244405 ± 0.000993583, median=0.00264576, 95% CI [0.00164945, 0.00318889], n_pos=5/5
- Dir−Mc: mean=0.00100958 ± 0.00180359, median=0.000649515, 95% CI [-0.000258628, 0.0026483], n_pos=4/5
- HistDir−Mc: mean=0.000604422 ± 0.00191421, median=0.000193488, 95% CI [-0.000791418, 0.00227863], n_pos=4/5
- HistDir−Hist: mean=0.000407794 ± 0.000873702, median=0.000372873, 95% CI [-0.000302418, 0.00116173], n_pos=4/5

## Pool summaries (means over seeds 10–14)

### Interpolation

| model | return | FailRate | θ-sat | action-clip | NMSE | cos_full | cos_stable |
|-------|--------|----------|-------|-------------|------|----------|------------|
| mc | -0.004949 | 0.20 | 0.94 | 0.00 | 0.267 | 0.535 | 0.540 |
| eagm_base | -0.006252 | 0.65 | 0.69 | 0.00 | 0.257 | 0.214 | 0.222 |
| eagm_dir_matched | -0.003909 | 0.00 | 0.89 | 0.00 | 0.258 | 0.678 | 0.686 |
| eagm_hist | -0.004773 | 0.07 | 0.85 | 0.00 | 0.237 | 0.601 | 0.604 |
| eagm_hist_dir_matched | -0.004423 | 0.00 | 0.88 | 0.00 | 0.245 | 0.609 | 0.612 |

### Extrapolation

| model | return | FailRate | θ-sat | action-clip | NMSE | cos_full | cos_stable |
|-------|--------|----------|-------|-------------|------|----------|------------|
| mc | -0.006312 | 0.30 | 0.94 | 0.00 | 0.506 | 0.526 | 0.526 |
| eagm_base | -0.007746 | 0.78 | 0.69 | 0.00 | 0.494 | 0.178 | 0.189 |
| eagm_dir_matched | -0.005302 | 0.13 | 0.89 | 0.00 | 0.495 | 0.670 | 0.670 |
| eagm_hist | -0.006115 | 0.22 | 0.85 | 0.00 | 0.456 | 0.589 | 0.589 |
| eagm_hist_dir_matched | -0.005708 | 0.10 | 0.88 | 0.00 | 0.470 | 0.594 | 0.606 |

## Pre-registered decision

**Case C.** Dir−Base reproduced, but Hist+Dir still does not stably beat Mc. Focus next on execution-state / history representation (Stage 6B.2 HistC), not more Dir complexity. Do not auto-start without human review.

Stage 6B.2 was **not** auto-started.

## Comparison to Stage 5E corrected (seeds 5–9)

Stage 5E corrected (exploratory OOD set) vs 6B.1 (confirmatory):

| contrast | Stage5E corr. n_pos | Stage6B.1 n_pos |
|----------|---------------------|-----------------|
| interp Dir−Base | 5/5 | 5/5 |
| interp HistDir−Mc | 3/5 | 3/5 |
| extrap Dir−Base | 5/5 | 5/5 |
| extrap HistDir−Mc | 3/5 | 4/5 |

