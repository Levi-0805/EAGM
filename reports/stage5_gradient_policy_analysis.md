# Stage 5D — Gradient Quality on the Policy Path

For Base and Dir-Matched, at θ∈{0, mid (u=25), final}, compare model ∇θJ to independent sim FD on the same policy-opt starts. Also record true return change for the actual θ0→θ_final update.

| seed | Δcos θ0 | Δcos mid | Δcos final | Δtrue-return (Dir−Base) | probe cos Base | probe cos Dir |
|------|---------|----------|------------|-------------------------|----------------|---------------|
| 5 | +0.140 | −0.044 | −0.198 | +2.53e-4 | 0.765 | 0.840 |
| 6 | +0.254 | +0.234 | +0.142 | +3.77e-4 | 0.717 | 0.856 |
| 7 | +0.386 | +0.509 | +0.376 | +1.02e-3 | 0.648 | 0.880 |
| 8 | +0.358 | +0.259 | +0.135 | +7.54e-4 | 0.712 | 0.846 |
| 9 | +0.255 | +0.001 | −0.107 | +1.63e-4 | 0.767 | 0.935 |

### Paired

- path Δcos θ0: mean=+0.278, **5/5**, CI excludes 0  
- path Δcos mid: mean=+0.192, **4/5**, CI excludes 0  
- path Δcos final: mean=+0.069, 3/5, CI **includes** 0  
- update return Dir−Base: mean=+5.14e-4, **5/5**

## Q4 — Do better gradients sit on the policy path?

**Mostly yes early/mid-path; weaker at θ_final.**

Dir clearly improves cosine at θ0 and usually at mid-trajectory checkpoints where Adam still moves. At θ_final the cosine gap shrinks or flips on some seeds (saturation / local geometry), yet **true physical return of the Dir update still beats Base on 5/5 seeds**. So the policy gain is not “probe-only”: it tracks better gradients in the region the optimizer actually traverses early, even if final-θ probe fidelity is imperfect.

Artifacts: `runs/stage5d_gradpath/` (per-seed `*_path.json`).
