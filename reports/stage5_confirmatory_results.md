# Stage 5A — Confirmatory Results (seeds 5–9)

Frozen from Stage 4 commit `c386ed3193bea0ea676b4d014be33a3bfe9fe24b`. Hyperparameters locked in `configs/stage5_frozen.json`. Stage 4 seeds 0–4 are **exploratory**; seeds 5–9 are an **independent confirmatory** set with freshly generated trajectories.

Primary metric: **EAGM-Dir-Matched − EAGM-Base** physical policy return.  
Secondary: Dir − Mc. Bootstrap CIs use n=5 seeds and are **unstable**; all seeds are shown.

## Confirmatory seed table

| seed | Mc | Base | Dir | Dir−Base | Dir−Mc | cos Mc | cos Base | cos Dir |
|------|----|------|-----|----------|--------|--------|----------|---------|
| 5 | -0.002566 | -0.001831 | -0.001578 | +2.53e-4 | +9.88e-4 | 0.403 | 0.765 | 0.840 |
| 6 | -0.002710 | -0.002006 | -0.001630 | +3.77e-4 | +1.08e-3 | 0.434 | 0.717 | 0.856 |
| 7 | -0.002001 | -0.002533 | -0.001509 | +1.02e-3 | +4.92e-4 | 0.498 | 0.648 | 0.880 |
| 8 | -0.001886 | -0.002378 | -0.001624 | +7.54e-4 | +2.62e-4 | 0.526 | 0.712 | 0.846 |
| 9 | -0.003007 | -0.001694 | -0.001531 | +1.63e-4 | +1.48e-3 | 0.332 | 0.767 | 0.935 |

### Paired seed stats (confirmatory only)

- **Dir − Base:** mean=+5.14e-4, median=+3.77e-4, 95% CI [+2.42e-4, +8.16e-4], **5/5 positive**
- **Dir − Mc:** mean=+8.60e-4, 95% CI [+4.72e-4, +1.22e-3], **5/5 positive**
- **Base − Mc:** mean=+3.46e-4, 95% CI [−2.70e-4, +9.60e-4], 3/5 positive (CI includes 0)

Wall ≈ 1856 s. Artifacts: `runs/stage5a_confirm/`.

## Exploratory Stage 4 (seeds 0–4) — not confirmatory

Previously: Dir−Base mean ≈ +1.01e-3 (5/5). Seed 0 participated in Stage 3.2 selection.

## Merged descriptive (do not use alone as confirmation)

Pooling exploratory+confirmatory strengthens the sign pattern but mixes selection-exposed and clean seeds. Confirmatory-only evidence is the claim-bearing set.

## Q1 — Confirmatory reproduction

**Yes.** On new seeds 5–9 and new trajectories, Dir-Matched again beats paired Base on physical policy return in **5/5** seeds; the seed-level CI excludes 0 (with the usual n=5 caveat). Gradient cosine also rises (Base ~0.65–0.77 → Dir ~0.84–0.93).

## Stage 5C baselines (run on same confirmatory seeds)

| seed | ObsMLP | AuxMLP | Base | Dir | Base−Obs | Base−Aux | Dir−Aux |
|------|--------|--------|------|-----|----------|----------|---------|
| 5 | -0.003034 | -0.003372 | -0.001831 | -0.001578 | +1.20e-3 | +1.54e-3 | +1.79e-3 |
| 6 | -0.003069 | -0.002544 | -0.002006 | -0.001630 | +1.06e-3 | +5.38e-4 | +9.14e-4 |
| 7 | -0.002972 | -0.002625 | -0.002533 | -0.001509 | +4.39e-4 | +9.20e-5 | +1.12e-3 |
| 8 | -0.003051 | -0.002703 | -0.002378 | -0.001624 | +6.73e-4 | +3.25e-4 | +1.08e-3 |
| 9 | -0.003036 | -0.002361 | -0.001694 | -0.001531 | +1.34e-3 | +6.67e-4 | +8.30e-4 |

Params: EAGM-Base/Dir 40 874 · ObsMLP 41 788 · AuxMLP 53 002.

- Base−Obs: mean=+9.41e-4 (5/5)
- Base−Aux: mean=+7.44e-4 (5/5)
- Dir−Aux: mean=+1.26e-3 (5/5)
- Aux−Obs: mean=+1.97e-4 (3/5; CI includes 0)

**Q3:** Structured U0+Δ bottleneck beats capacity-matched ObsMLP and AuxCommandMLP under the same observables / u_exec labels. Direction loss adds a further lift on top of the bottleneck.
