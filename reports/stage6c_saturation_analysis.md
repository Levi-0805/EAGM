# Stage 6C.1 — Policy Saturation Analysis

**Branch:** `stage6c-path-aware-saturation`  
**Parent:** `1f25a97`  
**Artifacts:** `runs/stage6c1_lr_ablation/`  
**Frozen WMs:** Stage 6B.1 seeds 10–14 (Mc / Base / Dir)  
**Wall:** ≈965 s  

## Protocol

- Freeze world-model checkpoints; only policy Adam updates (50 steps).
- LRs: 0.03 / 0.015 / 0.0075 (identical starts, init, eval within seed).
- Diagnostics: θ saturation, ρ_proj, outward pressure, action clip (separate).
- LR narrative uses **validation** physical return; OOD reported but not used for selection.

## Headline result

**For EAGM-Dir, lowering LR removes saturation but worsens physical return.**  
Saturation at lr=0.03 is therefore **not** shown to be the binding constraint on Dir policy quality under this budget.

## Aggregate (5 seeds)

| Model | LR | Val return (mean) | Final θ-sat | Mean ρ_proj | Interp ΔR vs θ₀ |
|-------|-----|-------------------|-------------|-------------|-----------------|
| Mc | 0.03 | −0.00466 | 0.94 | 0.30 | +0.00088 |
| Mc | 0.015 | −0.00458 | 0.00 | 0.00 | +0.00092 |
| Mc | 0.0075 | −0.00482 | 0.00 | 0.00 | +0.00065 |
| Base | 0.03 | −0.00600 | 0.69 | 0.25 | −0.00043 |
| Base | 0.015 | −0.00553 | 0.00 | 0.00 | −0.00003 |
| Base | 0.0075 | −0.00534 | 0.00 | 0.00 | +0.00012 |
| **Dir** | **0.03** | **−0.00367** | **0.89** | **0.31** | **+0.00192** |
| Dir | 0.015 | −0.00386 | 0.00 | 0.00 | +0.00169 |
| Dir | 0.0075 | −0.00446 | 0.00 | 0.00 | +0.00104 |

### Paired interp return: lower LR − lr=0.03

| Model | 0.015 − 0.03 | 0.0075 − 0.03 |
|-------|--------------|---------------|
| Mc | +4e−5 (2/5+, CI∋0) | −2.3e−4 (1/5+) |
| Base | **+4.0e−4 (5/5+, CI>0)** | +5.5e−4 (4/5+) |
| **Dir** | **−2.3e−4 (0/5+, CI<0)** | **−8.8e−4 (0/5+, CI<0)** |

## Interpretation

1. **Dir (primary):** High sat (~89%) and nontrivial ρ_proj (~0.31) coexist with the **best** returns. Cutting LR to eliminate sat **reliably hurts** Dir (0/5 seeds improve). → Do **not** claim that reducing saturation (via LR) is a method contribution for Dir.
2. **ρ_proj:** At lr=0.03, ~30% of the raw Adam step is removed by the box — yet return still improves along the trajectory (see 6C.0 physical curve). High ρ_proj alone does not prove a failed optimizer.
3. **Base:** Weaker gradients; lower LR helps Base catch up slightly. This is model-quality limited, not a reason to retune Dir’s locked lr=0.03.
4. **Mc:** Also highly saturated at 0.03 (~94%); milder LR effect, CI includes zero for 0.015.

## Decision (saturation track)

| Criterion | Outcome |
|-----------|---------|
| Lower LR only delays sat without improving Dir return | **Yes — do not keep as method** |
| Saturation high but Dir return still best | **Report as diagnostic; do not force an architectural “desaturation” fix** |
| Keep Stage-5 locked lr_policy=0.03 for Dir | **Yes** |

## 6C.0 audit cross-check

`runs/stage6c0_audit/audit.json` (PASS):

- Feasible FD at θ=+1 → all backward one-sided; stability mask separate.
- Paired Base/Dir init digest match; diverge after train.
- Path probes at fractions 0…1.0; pilot final sat ≈0.96 on frozen Dir.
- Physical return improves through u=40 even as sat jumps ~0→0.89.

## Next

Stage 6C.2/6C.3 tests whether **Path-Aware** direction labels beat random probes under matched FD budget — without changing the policy LR.
