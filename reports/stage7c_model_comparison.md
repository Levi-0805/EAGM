# Stage 7C — Reach Single-Seed Smoke (NOT confirmatory)

**Artifact:** `runs/stage7c_reach_smoke/results.json`  
**Seed:** 20  
**WM updates:** **400** (smoke; locked full budget remains 3000 for 7D)  
**Status:** Pipeline gate — models train, labels valid, returns finite.  
**Do not treat as evidence that Dir wins or loses on Reach.**

## Setup

- Task: Reach (random EE goals, hold refs after one-shot IK)
- Models: Mc, EAGM-Base, EAGM-Dir-Random, AuxCommandMLP (matched init/batch where applicable)
- FK surrogate RMSE ≈ 0.039 m
- Dir labels: 8 × feasible FD; primary return uses surrogate EE (matched imagined/physical)

## Smoke numbers (seed 20 only)

| Model | Cosine (med) | Policy return | Δ vs θ₀ |
|-------|--------------|---------------|---------|
| Mc | 0.038 | −0.0699 | −0.0001 |
| Base | 0.163 | −0.0648 | +0.0050 |
| Dir-Random | 0.100 | −0.0654 | +0.0044 |
| Aux | 0.084 | −0.0666 | +0.0032 |

Contrasts (return): Dir−Base ≈ **−6.4e−4**; Dir−Mc ≈ **+4.5e−3**; Dir−Aux ≈ **+1.2e−3**.

## Interpretation

1. Pipeline works: collect → FK fit → train → Dir labels → policy opt → physical eval.  
2. Under **reduced** updates, Dir did **not** beat Base on this one seed. That is **not** a Stage-7 failure — confirmatory requires locked 3000 updates and ≥10 seeds after human approval.  
3. `recommend_proceed_to_7d` left **false** in JSON; human decides whether to run full-budget 7D.

## Gate checklist

- [x] All models train  
- [x] Direction labels valid  
- [x] Physical returns finite  
- [x] No privileged exec-state inputs  
- [ ] Full-budget multi-seed (7D) — **awaiting human review**
