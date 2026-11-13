# Stage 7 — Cross-Task Generalization of Directional Supervision

**Branch:** `stage7-cross-task-generalization`  
**Parent:** `43cfc79` (`stage6c-path-aware-saturation`)  
**Frozen config:** `configs/stage7_frozen.json`

## Research question

Does directional consistency supervision (Dir-Random) improve model-based policy optimization across distinct robot manipulation objectives and execution configurations — not only within OpenArm joint tracking?

## What is frozen (not redesigned)

| Keep | Drop |
|------|------|
| EAGM-Dir-Random (primary) | HistC / ExecState |
| Mc, EAGM-Base, AuxCommandMLP | Path-Dir |
| lr_policy=0.03, θ∈[-1,1] | LR-desaturation as a “method” |
| λ_u=1, λ_d=0.1, 3000 WM updates, 50 policy updates | Module stacking for novelty |

θ-sat, ρ_proj, action clip remain **diagnostics only**.

## Stage map & gates

| Stage | Content | Gate before next |
|-------|---------|------------------|
| **7A** | Baseline audit + Path-Dir pilot doc fix | PASS audit |
| **7B** | Reach env + unit tests | Tests PASS |
| **7C** | Single-seed matched Mc/Base/Dir/Aux on Reach | Smoke OK |
| **7D** | ≥10-seed confirmatory (human-approved) | Human review |
| **7E** | Pick-and-Place | Reach confirmatory gate |
| **7F** | Visual validation (separate) | Separate decision |
| External baselines | PETS/MBPO/TD-MPC2/… | Only if runnable + matched budget |

**Do not auto-launch 7D/E/F.** No SOTA claims without external benchmarks.

## Paper separation (important)

Stage 6C / Stage 7 Dir-Random is **training-time gradient consistency**, not the manuscript’s R5 policy-update screening. Do not merge Stage 7 into `main(10).pdf` until a separate method write-up is planned. Do not treat R5 negative results as refuting Dir-Random (or vice versa).

## Artifact rules

- New runs only under `runs/stage7_*`
- Do not modify `CVPR2027/` or overwrite Stage 5/6 artifacts
- Record git commit, MuJoCo, torch, GPU in each audit/results JSON
