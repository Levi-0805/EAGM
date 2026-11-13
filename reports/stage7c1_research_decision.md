# Stage 7C.1 — Research Decision

**Branch:** `stage7-cross-task-generalization`  
**Date:** 2026-11-13  
**Stop for human review. Do not auto-start Stage 7D / 7E / 7F.**

## Answers to the four hypotheses from the 400-update smoke

| Hypothesis | Evidence after 7C.1 |
|------------|---------------------|
| 1. Reach needs little exec memory; Base enough | Partial — Base dynamics best (NMSE), but Dir still wins return/FailRate at 3000 |
| 2. Dir not adapted to new objective | Weakened — Dir cos@θ₀≈0.40 ≫ Base; labels not broken |
| 3. **400 updates too few** | **Supported** — Dir−Base flips −6.4e−4 → **+5.8e−4** at 3000 |
| 4. Reward/grad protocol bug | **Largely ruled out** — reward parity PASS |

## Decisions

| Item | Decision |
|------|----------|
| Stage 7D (10×3000) | **NOT approved** — only one full-budget seed |
| Seeds 20–22 engineering pilot | **Optional / human-gated** — script decision flag true, but margins small |
| Pick-and-Place / Vision | **Do not start** |
| New modules | **Do not add** |
| Dir-Random as frozen method | **Retain** |

## How to read the full-budget Dir edge

- Positive but **small** (Dir−Base ≈ +5.8e−4 on ~−0.031 returns).  
- Dir improves ΔR vs θ₀; Base worsens.  
- Gradient story: Dir learns better directions early; path cosine decays.  
- Dynamics: Dir NMSE worse than Base — same pattern as tracking stages.

This is **not** yet “Dir generalizes to Reach.” It is: “smoke was undertrained; with locked budget Dir is competitive / slightly ahead on one seed.”

## Recommended next step (human chooses)

**A (preferred).** Run **7C.3 three-seed pilot** (seeds 20–22, 3000 updates, same protocol).  
Approve 7D only if Dir−Base is positive on ≥2/3 with coherent gradient diagnostics.

**B.** Pause Reach; write a short “task dependence” note and design Pick-and-Place where exec coupling is stronger — without claiming Reach victory.

**C.** Dig deeper on FK error / true-MuJoCo secondary metric if absolute EE accuracy is a concern for paper claims.

## Explicit non-claims

- Do not claim Stage 4–6 Dir advantage automatically transfers to Reach.  
- Do not claim SOTA or multi-task superiority.  
- Do not merge into CVPR main manuscript from this branch yet.
