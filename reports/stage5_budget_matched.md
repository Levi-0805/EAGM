# Stage 5B — Budget-Matched Comparison

Train interaction budget = dynamics collection + **train** direction labels.  
Test FD diagnostics listed separately and excluded from train budget.

Methods:

- **A** EAGM-Base, original dynamics budget (24×200 = 4800 steps)
- **B** EAGM-Base, Dir’s train FD budget converted to extra dynamics trajectories (+81 eps ≈ 16200 steps)
- **C** EAGM-Dir-Matched (4800 dynamics + 16128 train FD label steps)

Same confirmatory seeds 5–9 and same primary test starts (deterministic recollect).

| seed | A Base | B Base+dyn | C Dir | C−A | C−B | B−A |
|------|--------|------------|-------|-----|-----|-----|
| 5 | -0.001831 | -0.001855 | -0.001578 | +2.53e-4 | +2.77e-4 | −2.35e-5 |
| 6 | -0.002006 | -0.001914 | -0.001630 | +3.77e-4 | +2.84e-4 | +9.26e-5 |
| 7 | -0.002533 | -0.001706 | -0.001509 | +1.02e-3 | +1.97e-4 | +8.27e-4 |
| 8 | -0.002378 | -0.001817 | -0.001624 | +7.54e-4 | +1.93e-4 | +5.61e-4 |
| 9 | -0.001694 | -0.001877 | -0.001531 | +1.63e-4 | +3.45e-4 | −1.82e-4 |

### Paired stats

- **C−A:** mean=+5.14e-4, 5/5, CI excludes 0  
- **C−B:** mean=+2.59e-4, **5/5**, CI [+2.11e-4, +3.07e-4]  
- **B−A:** mean=+2.55e-4, 3/5, CI includes 0  

Budget match note: B train budget 21000 vs C 20928 control steps (target FD 16128).

## Q2 — Is the extra direction-supervision cost worth it?

**Yes, under this matched-budget protocol.** Converting Dir’s FD label budget into more dynamics trajectories (B) helps Base somewhat on average, but **Dir still beats B on all five confirmatory seeds**. Direction labels are therefore not explained away as “just more environment interaction.”

Caveats: matched on **control-step count**, not wall-clock or GPU FLOPs; FD labels are structured gradient queries, not i.i.d. rollouts; n=5 remains small.
