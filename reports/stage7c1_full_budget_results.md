# Stage 7C.1 — Full-Budget Seed20 Reach Results

**Artifact:** `runs/stage7c1_diagnosis/results.json`  
**Updates:** **3000** WM + 50 policy @ lr=0.03  
**Models:** Mc / Base / Dir-Random (paired init + batch for Base/Dir)  
**Smoke 400-update results preserved** at `runs/stage7c_reach_smoke/`

## Headlines

| Contrast | 400-update smoke | **3000 full budget** |
|----------|------------------|----------------------|
| Dir − Base | **−6.4e−4** | **+5.8e−4** |
| Dir − Mc | +4.5e−3 | +1.6e−4 |
| Base − Mc | (implied large +) | **−4.2e−4** |

**Budget flip:** At 400 updates Dir lagged Base; at 3000 Dir edges Base. Smoke under-training was a real confounder.

## Per-model (test starts, surrogate EE return)

| Model | NMSE | cosF | Return | Δ vs θ₀ | FailRate | θ-sat |
|-------|------|------|--------|---------|----------|-------|
| Mc | 1.03 | −0.063 | −0.03145 | −8e−6 | 0.67 | (pol) |
| Base | **0.85** | 0.103 | −0.03187 | −4.3e−4 | 0.83 | high |
| Dir | 1.16 | 0.113 | **−0.03129** | **+1.6e−4** | **0.50** | high |

Notes:
- Dir has **worse** dynamics NMSE than Base but **better** physical ΔR / FailRate.  
- Absolute returns are ~−0.031 (EE-scale); deltas are small.  
- True-MuJoCo EE returns logged per model on start0 (see JSON).

## Smoke vs full (same seed20)

| | Smoke Dir ret | Smoke Base ret | Full Dir | Full Base |
|--|---------------|----------------|----------|-----------|
| Policy return | −0.0654 | −0.0648 | −0.0313 | −0.0319 |

Scales differ (episode/horizon/collection setup tightened in 7C.1: 36×200 vs smoke 18×120) — **do not compare absolute returns across protocols**; only within-protocol contrasts.

## Gate flags (auto)

- `dir_beats_base_full_budget`: true  
- `dir_beats_mc_full_budget`: true  
- `recommend_three_seed_pilot`: true  
- `recommend_stage7d`: **false** (human gate; single seed only)
