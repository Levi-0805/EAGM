# Stage 7C.1 — Gradient Audit (Reach, seed20, full-budget models)

**Artifact:** `runs/stage7c1_diagnosis/gradient_audit.json`  
**Reference:** physical feasible FD; stability mask ≠ boundary mask  
**Path thetas:** Dir policy path (θ₀ / mid / final), plus each model’s own final

## Cosine vs physical FD (Dir path θ)

| Model | θ₀ cosF | mid cosF | final cosF | final dir-agree | final sat |
|-------|---------|----------|------------|-----------------|-----------|
| Mc | −0.064 | −0.057 | −0.063 | 0.22 | 0.68 |
| Base | 0.116 | 0.127 | 0.145 | 0.50 | 0.68 |
| **Dir** | **0.396** | **0.239** | **0.192** | 0.40 | 0.68 |

## Key readings

1. **Dir does learn better direction at θ₀** (cos≈0.40 vs Base≈0.12).  
2. Along the policy path, Dir cosine **decays** (0.40→0.19) but remains ≥ Base at final.  
3. Mc gradients are anti-aligned / near-zero quality on Reach FD labels.  
4. Magnitude error stays ~1.0 for Dir (order-matched); Base ~1.12.  
5. Boundary saturation at final is high (~0.68) for all — diagnostic only.

## Probe-label cosines (random θ bank, from full-budget eval)

| Model | cosF med | cosS med |
|-------|----------|----------|
| Mc | −0.063 | −0.072 |
| Base | 0.103 | 0.157 |
| Dir | 0.113 | 0.132 |

Dir’s probe-bank cosine is only slightly above Base; the **path θ₀** gap is much larger. This matches the Stage-6C theme: probe distribution vs optimization path can differ — here Dir still helps at θ₀ under Reach FD.

## Interpretation (for decision report)

Dir **does** improve gradient consistency relative to Base on Reach (especially near θ₀). Combined with full-budget physical returns (see results report), this is closer to “learned better directions + small return lift” than “labels broken.” Absolute cosines remain far below Stage 5–6 tracking levels → task is harder / signal weaker.
