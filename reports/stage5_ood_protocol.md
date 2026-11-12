# Stage 5E — OOD Controller-B Protocol & Results

Status: **executed** after Stage 5A–5C passed stop-conditions.  
Artifacts: `runs/stage5e_ood/` · wall ≈ 2308 s · seeds 5–9.

## Protocol (frozen before looking at outcomes)

- **Train:** `episode_randomize=True` with ranges gain∈[0.92,1.08], lag∈{0,1,2}, rate∈[0.03,0.08], smooth∈[0.0,0.35]; 36 episodes / seed (24 train / 6 val / 6 iid-hold).
- **Models share** the same randomized trajectories. Deployable models **do not** read `known_config_vector` / hidden B state.
- **Interpolation test:** 6 frozen configs inside train ranges × 2 eps each (never used for tuning).
- **Extrapolation test:** 6 frozen configs outside ranges (gain 0.85/1.15, lag 3, rate 0.02/0.10, smooth 0.50).
- Prefer History + Dir; also report Base/Dir/Mc. **No CF loss.**

Interp and extrap are reported **separately** (never averaged into one OOD score).

## Results — policy return

### Interpolation

| seed | Mc | Base | Dir | Hist | Hist+Dir | HistDir−Mc | Dir−Base |
|------|----|------|-----|------|----------|------------|----------|
| 5 | -0.00535 | -0.00561 | -0.00459 | -0.00377 | -0.00367 | +1.67e-3 | +1.01e-3 |
| 6 | -0.00430 | -0.00516 | -0.00271 | -0.00407 | -0.00356 | +0.74e-3 | +2.45e-3 |
| 7 | -0.00515 | -0.00465 | -0.00274 | -0.00473 | -0.00288 | +2.27e-3 | +1.91e-3 |
| 8 | -0.00310 | -0.00718 | -0.00484 | -0.00540 | -0.00411 | −1.00e-3 | +2.34e-3 |
| 9 | -0.00400 | -0.00544 | -0.00363 | -0.00411 | -0.00452 | −0.51e-3 | +1.81e-3 |

Paired: Dir−Base mean=+1.90e-3 (**5/5**); HistDir−Mc mean=+0.63e-3 (**3/5**, CI includes 0); HistDir−Hist mean=+0.67e-3 (4/5).

### Extrapolation

| seed | Mc | Base | Dir | Hist | Hist+Dir | HistDir−Mc | Dir−Base |
|------|----|------|-----|------|----------|------------|----------|
| 5 | -0.00670 | -0.00694 | -0.00570 | -0.00528 | -0.00514 | +1.56e-3 | +1.24e-3 |
| 6 | -0.00545 | -0.00659 | -0.00387 | -0.00535 | -0.00479 | +0.66e-3 | +2.72e-3 |
| 7 | -0.00656 | -0.00605 | -0.00393 | -0.00591 | -0.00404 | +2.53e-3 | +2.11e-3 |
| 8 | -0.00446 | -0.00877 | -0.00614 | -0.00680 | -0.00523 | −0.77e-3 | +2.63e-3 |
| 9 | -0.00549 | -0.00701 | -0.00504 | -0.00542 | -0.00605 | −0.56e-3 | +1.97e-3 |

Paired: Dir−Base mean=+2.14e-3 (**5/5**); HistDir−Mc mean=+0.68e-3 (**3/5**, CI includes 0).

### Dynamics NMSE (approx. seed means)

- Interp: ~0.17–0.25 (vs Stage 3.1 fixed→held-out ~2.4–2.8)
- Extrap: ~0.30–0.49 (~1.5–2× worse than interp)

## Q5 — Does train-time B randomization improve unseen-config generalization?

**Partially.**

1. **Coverage fix:** Train-time randomization collapses the previous catastrophic OOD dynamics failure (NMSE ~2.5 → ~0.2 interp / ~0.4 extrap). That is a real protocol win.
2. **Dir vs Base under OOD:** Matched-θ Dir still improves physical return vs Base on **both** interp and extrap (5/5). Direction supervision remains useful when B varies.
3. **Vs Mc:** Hist+Dir does **not** reliably beat Mc on frozen OOD returns (3/5; CI includes 0). So randomization + history is not yet a settled OOD SOTA claim against a strong memory baseline.
4. **Extrapolation remains harder** (higher NMSE, worse absolute returns). Do not pool with interpolation.

CF loss is **not** justified yet: residual errors may still be explained by coverage / capacity under history encoding; no clean counterfactual-gradient residual analysis beyond this pre-experiment.
